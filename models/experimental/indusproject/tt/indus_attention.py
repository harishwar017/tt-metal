# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear
import math
from models.common.utility_functions import is_blackhole
import os
from typing import Tuple


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024
        self.pad_id = config.eos_token_id
        self.dim = config.n_embd  # Use config dimension, not hardcoded!
        self.tile_size = 32

        self.device = device
        self.dram_grid_size = device.dram_grid_size() if device else None
        self.cluster_shape = list(device.shape) if device is not None else None
        # Get the weights
        self.tt_weight_c_attn = ttnn.load_tensor(
            tt_cache_path + base_address + ".c_attn.weight" + str(dtype) + ".tensorbin",
            device=device,
        )
        self.dram_weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(self.dram_grid_size.x - 1, self.dram_grid_size.y - 1),
                )
            }
        )
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.num_devices = 1
        self.n_kv_heads = self.n_head
        self.qkv_size = self.head_dim * (3 * self.n_head)
        # # Convert to torch to check shape and prepare for DRAM sharding
        self.tt_weight_c_attn_temp = ttnn.to_torch(self.tt_weight_c_attn)

        shape = self.tt_weight_c_attn.shape

        # Create shard config matching the final shape
        wqkv_mem_config = self.create_dram_sharded_mem_config(shape[-2], shape[-1])

        self.wqkv = ttnn.as_tensor(
            self.tt_weight_c_attn_temp,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(2, 3), mesh_shape=self.cluster_shape),
            memory_config=wqkv_mem_config,
        )

        self.tt_weight_c_attn_decode = self.wqkv

        # Transpose the original weight for prefill mode Linear helper
        # Linear expects weight as [out_features, in_features] = [6144, 2048]
        self.tt_weight_c_attn = ttnn.transpose(self.tt_weight_c_attn, -2, -1)

        self.tt_weight_c_proj = ttnn.load_tensor(
            tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".tensorbin",
            device=device,
        )
        self.tt_weight_c_proj = ttnn.transpose(self.tt_weight_c_proj, -2, -1)

        # Load biases
        self.tt_bias_c_attn = ttnn.load_tensor(
            tt_cache_path + base_address + ".c_attn.bias" + str(dtype) + ".tensorbin",
            device=device,
        )

        self.tt_bias_c_proj = ttnn.load_tensor(
            tt_cache_path + base_address + ".c_proj.bias" + str(dtype) + ".tensorbin",
            device=device,
        )

        ones = ttnn.ones([1, 1, self.block_size, self.block_size], device=self.device, dtype=dtype)
        ones = ttnn.to_layout(ones, ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.tril(ones)

        self.c_attn = Linear(
            self.config.n_embd,
            3 * config.n_embd,
            self.tt_weight_c_attn,
            self.tt_bias_c_attn,
        )
        # self.c_attn_decode = Linear(
        #     self.config.n_embd,
        #     3 * config.n_embd,
        #     self.tt_weight_c_attn,
        #     self.tt_bias_c_attn,
        #     output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # )
        self.c_proj = Linear(
            self.config.n_embd,
            self.config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
        )
        self.c_proj_decode = Linear(
            self.config.n_embd,
            self.config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        self.init_kv_cache()

        # KV position (host-side)
        self.cur_pos = 0

        # KV position (device-side tensor)
        self.cur_pos_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        is_bh = is_blackhole()

        self.model_config = {}
        self.model_config[
            "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
        ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
            shape=(math.ceil(self.n_head / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
            core_grid=ttnn.CoreRangeSet({self.num_to_corerange(batch_size_per_device_group)}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["CREATE_QKV_DECODE_SHARD"] = (
            (
                lambda: ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )
            if is_bh
            else (lambda: ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
        )

        self.model_config["SDPA_DECODE_PROGCFG"] = lambda: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=128 if is_bh else 256,
            k_chunk_size=128 if is_bh else 256,
        )
        self.max_batch_size = 1

        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

        # For DRAM-sharded matmul, use the DRAM grid size (not compute grid)
        # The weight is sharded across DRAM cores (8 cores in a row)
        dram_num_cores = self.dram_grid_size.x  # 8 cores

        self.model_config["XQKV_DECODE_PROGCFG"] = lambda: (
            self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.qkv_size // self.num_devices,
                num_cores=dram_num_cores,  # Must match weight's shard grid
            )
        )

        residual_grid = self.dram_shard_core_grid_for_k(self.dim // self.num_devices)
        self.model_config["DECODE_RESIDUAL_MEMCFG"] = lambda: (
            ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // residual_grid.num_cores // self.num_devices,
                ),
                residual_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        )

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors"""
        dram_cores = self.dram_grid_size.x  # WH has 12 dram cores, P150 has 8, P100 has 7
        assert self.dram_grid_size.y == 1, "Current dram sharding assumes y dim is 1"
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def find_largest_divisor(self, n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1  # Fallback to 1 if no divisor found

    def dram_shard_core_grid_for_k(self, k: int) -> Tuple[int, int]:
        rows, cols = self.find_grid(k // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid_k_n(self, K, N):
        max_rows = 8
        max_cols = 8  # Maximum number of rows or columns
        max_cores = max_rows * max_cols  # Maximum number of cores

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
        possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
        )

    def find_grid(self, N):
        max_rows = 8
        max_cols = 8
        max_cores = max_rows * max_cols

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        target = 32
        possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
        possible_cores.sort(key=lambda x: abs(x - target))  # Sort by closest to target

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration for {N} tiles that evenly divides into {max_cores} cores of max size {max_rows}x{max_cols}."
        )

    def dram_shard_core_grid_for_k_and_n(self, k: int, n: int) -> Tuple[int, int]:
        rows, cols = self.find_grid_k_n(k // self.tile_size, n // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def dram_matmul_config(self, m: int, k: int, n: int, num_cores=None, fused_activation=None):
        # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
        if num_cores is None:
            # num_cores = self.dram_shard_core_grid_for_k(k).num_cores
            num_cores = self.dram_shard_core_grid_for_k_and_n(k, n).num_cores
            assert (
                k % (self.tile_size * num_cores) == 0
            ), f"k must be divisible by tile_size * num_cores: {k} % {self.tile_size * num_cores} != 0"
            # assert n % (self.tile_size * num_cores) == 0, f"n must be divisible by tile_size * num_cores: {n} % {self.tile_size * num_cores} != 0"
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.find_largest_divisor(k // (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=fused_activation,
        )

    def _transform_decode_inputs_device(self, tokens):
        # tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tokens)
        mem_config = self.model_config["DECODE_RESIDUAL_MEMCFG"]()
        tt_tokens = ttnn.to_memory_config(
            tt_tokens,
            mem_config,
        )
        return tt_tokens

    def num_to_corerange(self, x):
        assert x < 8 or x % 8 == 0
        num_x = min(x, 8)
        num_y = x // num_x
        assert num_x * num_y == x
        return ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(num_x - 1, num_y - 1),
        )

    def init_kv_cache(self):
        B = 1
        H = self.n_head
        T = self.config.block_size
        D = self.n_embd // self.n_head

        cache_k = torch.zeros((B, H, T, D))
        cache_v = torch.zeros((B, H, T, D))

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                # mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def dump_ttnn_tensor(self, tt_tensor, name, step, out_dir):
        dir = f"./models/experimental/indusproject/{out_dir}"
        os.makedirs(dir, exist_ok=True)

        path = os.path.join(dir, f"{name}_{step}")

        # TTNN-native dump (writes metadata + binary)
        ttnn.dump_tensor(tensor=tt_tensor, file_name=f"{path}.tensorbin")

        print(f"[DUMP] {name} -> {path} | shape={tt_tensor.shape}")

    def dump_tensor(self, tt_tensor, name, step, out_dir):
        dir = f"./models/experimental/indusproject/{out_dir}"
        os.makedirs(dir, exist_ok=True)

        path = os.path.join(dir, f"{name}_{step}.tensorbin")

        # TTNN-native dump (writes metadata + binary)
        tensor = ttnn.to_torch(ttnn.from_device(tt_tensor))
        torch.save(tensor, path)

        print(f"[DUMP] {name} -> {path} | shape={tensor.shape}")

    def forward_prefill(self, x):
        x1 = self.c_attn(x)
        # self.dump_tensor(x1, "prefill_embed", step="0", out_dir="after_l1/embed")

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            input=x1,
            num_heads=self.n_head,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            transpose_k_heads=False,
        )

        # self.dump_tensor(q, "only_prefill_q", step="0")
        # self.dump_tensor(k, "only_prefill_k", step="0")
        # self.dump_tensor(v, "only_prefill_v", step="0")

        ttnn.deallocate(x1)
        B = k.shape[0]

        keys = self.layer_past[0]
        values = self.layer_past[1]

        for b in range(B):
            ttnn.fill_cache(
                keys,
                k,
                batch_idx=b,
            )

            ttnn.fill_cache(
                values,
                v,
                batch_idx=b,
            )

        # Normal SDPA (optional)
        tt_y = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        tt_y = ttnn.experimental.nlp_concat_heads(tt_y, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x2 = self.c_proj(tt_y)
        ttnn.deallocate(tt_y)

        return x2

    def forward(self, x: ttnn.Tensor, current_pos_tensor) -> ttnn.Tensor:
        x_embed = self._transform_decode_inputs_device(x)

        # QKV linear layer with DRAM-sharded weight
        prog_config = self.model_config["XQKV_DECODE_PROGCFG"]()
        xqkv_fused_sharded = ttnn.linear(
            x_embed,
            self.tt_weight_c_attn_decode,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog_config,
            dtype=ttnn.bfloat16,
        )

        # Add bias - expand bias to match batch dimension
        # Get the actual batch size from the sharded output
        batch_size = xqkv_fused_sharded.shape[-2]
        bias_expanded = ttnn.to_torch(self.tt_bias_c_attn).unsqueeze(0).expand(batch_size, -1)
        bias_tensor = ttnn.from_torch(
            bias_expanded,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        xqkv_fused_sharded = xqkv_fused_sharded + bias_tensor
        ttnn.deallocate(bias_tensor)

        step = int(ttnn.to_torch(current_pos_tensor).item())
        # self.dump_tensor(xqkv_fused_sharded, "decode_embed", step, out_dir="after_l1/embed")

        # Convert from sharded to interleaved for reshape
        xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(xqkv_fused_sharded)

        # Reshape for nlp_create_qkv_heads_decode: track true unpadded batch in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
        # self.dump_tensor(xqkv_fused, "decode_embed_reshaped", step, out_dir="after_l1/embed")

        decode_cfg = self.model_config["CREATE_QKV_DECODE_SHARD"]()

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            input_tensor=xqkv_fused,
            num_heads=self.n_head,
            num_kv_heads=self.n_head,
            memory_config=decode_cfg,
        )

        # self.dump_tensor(q, "decode_q_final", step)
        # self.dump_tensor(k, "decode_k_final", step)
        # self.dump_tensor(v, "decode_v_final", step)

        # ttnn.deallocate(x1)
        keys = self.layer_past[0]
        values = self.layer_past[1]

        ttnn.experimental.paged_update_cache(keys, k, batch_offset=0, update_idxs_tensor=current_pos_tensor)
        ttnn.experimental.paged_update_cache(values, v, batch_offset=0, update_idxs_tensor=current_pos_tensor)

        ttnn.deallocate(k)
        ttnn.deallocate(v)

        sdpa_dec_cfg = self.model_config["SDPA_DECODE_PROGCFG"]()
        tt_y = ttnn.transformer.scaled_dot_product_attention_decode(
            input_tensor_q=q,
            input_tensor_k=keys,
            input_tensor_v=values,
            is_causal=True,
            cur_pos_tensor=current_pos_tensor,
            program_config=sdpa_dec_cfg,
        )

        ttnn.deallocate(q)

        B = q.shape[0]
        mem_config = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](B)

        tt_y = ttnn.to_memory_config(
            tt_y,
            memory_config=mem_config,
        )

        tt_y = ttnn.experimental.nlp_concat_heads_decode(tt_y, num_heads=self.n_head)

        x2 = self.c_proj_decode(tt_y)

        ttnn.deallocate(tt_y)

        return x2
