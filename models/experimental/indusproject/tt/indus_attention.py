# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear
import math
from models.common.utility_functions import is_blackhole


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024
        self.pad_id = config.eos_token_id

        self.device = device
        # Get the weights
        self.tt_weight_c_attn = ttnn.load_tensor(
            tt_cache_path + base_address + ".c_attn.weight" + str(dtype) + ".tensorbin",
            device=device,
        )

        self.tt_weight_c_proj = ttnn.load_tensor(
            tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".tensorbin",
            device=device,
        )

        self.tt_weight_c_attn = ttnn.transpose(self.tt_weight_c_attn, -2, -1)
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

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

        ones = ttnn.ones([1, 1, self.block_size, self.block_size], device=self.device, dtype=dtype)
        ones = ttnn.to_layout(ones, ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.tril(ones)

        self.c_attn = Linear(
            self.config.n_embd,
            3 * config.n_embd,
            self.tt_weight_c_attn,
            self.tt_bias_c_attn,
        )
        self.c_proj = Linear(
            self.config.n_embd,
            self.config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
        )

        self.init_kv_cache()
        self.head_dim = self.n_embd // self.n_head

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

        # self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
        #         m=num_rows(seq_len),
        #         k=k_dim,
        #         n=n_dim,
        #         grid_size=self.find_prefill_grid(prefill_rows, k_dim // self.tile_size),
        #         in0_block_w=1 if self.is_galaxy else None,
        #         fuse_batch=seq_len <= 1024,
        #         per_core_N=math.ceil(n_dim / (self.tile_size * dram_shard_grid_width)) if dram_sharded_wo else None,
        #     )

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

    def forward_prefill(self, x):
        x1 = self.c_attn(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            input=x1,
            num_heads=self.n_head,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            transpose_k_heads=False,
        )
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
        xqkv_fused = self.c_attn(x)

        xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)

        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, 1, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))

        decode_cfg = self.model_config["CREATE_QKV_DECODE_SHARD"]()
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            input_tensor=xqkv_fused,
            num_heads=self.n_head,
            num_kv_heads=self.n_head,
            memory_config=decode_cfg,
        )

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

        x2 = self.c_proj(tt_y)

        ttnn.deallocate(tt_y)

        return x2
