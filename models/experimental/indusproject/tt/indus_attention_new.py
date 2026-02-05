# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear


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

    def init_kv_cache(self):
        B = 1
        H = self.n_head
        T = self.config.block_size
        D = self.n_embd // self.n_head

        cache_k = torch.zeros((B, H, T, D))
        cache_v = torch.zeros((B, H, T, D))

        # self.k_cache = ttnn.from_torch(
        #     cache_k,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT,
        #     device=self.device,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )

        # self.v_cache = ttnn.from_torch(
        #     cache_v,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT,
        #     device=self.device,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        self.mesh_device = (1, 1)

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
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

        # Make sure layout is interleaved/tile
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

        keys = self.layer_past[0]
        values = self.layer_past[1]

        for b in range(B):
            ttnn.fill_cache(
                keys,
                k[b : b + 1],
                batch_idx=b,
            )

            ttnn.fill_cache(
                values,
                v[b : b + 1],
                batch_idx=b,
            )

        seq_len = q.shape[2]  # =14
        self.cur_pos += seq_len

        self.cur_pos_tensor = ttnn.from_torch(
            torch.tensor([self.cur_pos], dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Normal SDPA (optional)
        tt_y = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        tt_y = ttnn.experimental.nlp_concat_heads(tt_y, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # output projection
        x2 = self.c_proj(tt_y)

        ttnn.deallocate(tt_y)

        return x2

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x1 = self.c_attn(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            input_tensor=x1,
            num_heads=self.n_head,
            num_kv_heads=self.n_head,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ttnn.deallocate(x1)

        pos = self.cur_pos

        keys = self.layer_past[0]
        values = self.layer_past[1]

        # k = ttnn.sharded_to_interleaved(k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # v = ttnn.sharded_to_interleaved(v, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # k = ttnn.permute(k, (0, 2, 1, 3))
        # v = ttnn.permute(v, (0, 2, 1, 3))

        ttnn.experimental.paged_update_cache(keys, k, update_idxs=[self.cur_pos], batch_offset=0)
        ttnn.experimental.paged_update_cache(values, v, update_idxs=[self.cur_pos], batch_offset=0)

        self.cur_pos += 1

        ttnn.deallocate(k)
        ttnn.deallocate(v)

        tt_y = ttnn.transformer.scaled_dot_product_attention_decode(
            input_tensor_q=q,
            input_tensor_k=keys,
            input_tensor_v=values,
            is_causal=True,
            cur_pos=[pos],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(q)

        tt_y = ttnn.permute(tt_y, (1, 2, 0, 3))

        tt_y = ttnn.experimental.nlp_concat_heads(tt_y, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x2 = self.c_proj(tt_y)

        ttnn.deallocate(tt_y)

        return x2

    # def forward_1(self, x: ttnn.Tensor) -> ttnn.Tensor:
    #     assert x.shape[1] == 1, "Decode forward expects seq_len == 1"
    #     x1 = self.c_attn(x)

    #     q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    #         input=x1, input_kv=None, num_heads=self.n_head, num_kv_heads=None, transpose_k_heads=False,
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG
    #     )
    #     ttnn.deallocate(x1)

    #     # Update KV cache (single token)
    #     ttnn.update_cache(
    #         self.k_cache,
    #         k,
    #         update_idx=self.cur_pos,
    #     )

    #     ttnn.update_cache(
    #         self.v_cache,
    #         v,
    #         update_idx=self.cur_pos,
    #     )

    #     self.cur_pos += 1
    #     self.cur_pos_tensor = ttnn.from_torch(
    #         torch.tensor([self.cur_pos], dtype=torch.int32),
    #         device=self.device,
    #         dtype=ttnn.int32,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #     )

    #     # Load KV slice for attention
    #     k_cache_slice = ttnn.experimental.nlp_kv_cache_load_slice(
    #         self.k_cache,
    #         seq_len_start=0,
    #         seq_len_end=((self.cur_pos + 31) // 32) * 32,  # round up to tile
    #     )

    #     v_cache_slice = ttnn.experimental.nlp_kv_cache_load_slice(
    #         self.v_cache,
    #         seq_len_start=0,
    #         seq_len_end=((self.cur_pos + 31) // 32) * 32,
    #     )

    #     # Convert to interleaved for SDPA
    #     k_cache_slice = ttnn.sharded_to_interleaved(
    #         k_cache_slice,
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
    #     )

    #     v_cache_slice = ttnn.sharded_to_interleaved(
    #         v_cache_slice,
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
    #     )
    #     q = ttnn.permute(q, (2, 0, 1, 3))
    #     tt_y = ttnn.transformer.scaled_dot_product_attention_decode(
    #         q, k_cache_slice, v_cache_slice, is_causal=True, cur_pos=[self.cur_pos - 1],
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG
    #     )
    #     tt_y = ttnn.permute(tt_y, (0, 2, 1, 3))

    #     # Free early
    #     ttnn.deallocate(q)
    #     ttnn.deallocate(k_cache_slice)
    #     ttnn.deallocate(v_cache_slice)

    #     tt_y = ttnn.experimental.nlp_concat_heads(tt_y, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    #     # output projection
    #     x2 = self.c_proj(tt_y)

    #     ttnn.deallocate(tt_y)

    #     return x2
