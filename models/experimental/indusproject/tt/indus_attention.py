# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype, B):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024
        self.pad_id = config.eos_token_id
        self.B = B

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

        # Initialize KV cache
        self.head_dim = self.n_embd // self.n_head
        self.init_kv_cache()

    def init_kv_cache(self):
        """Initialize KV cache for incremental decoding"""
        batch_size = self.B  # change manually

        cache_k = torch.zeros((batch_size, self.n_head, 1, self.head_dim))
        cache_v = torch.zeros((batch_size, self.n_head, 1, self.head_dim))

        # Convert to TTNN tensors in DRAM
        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_prefill(self, x: ttnn.Tensor, current_pos) -> ttnn.Tensor:
        x1 = self.c_attn(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            input=x1, input_kv=None, num_heads=self.n_head, num_kv_heads=None, transpose_k_heads=False
        )
        ttnn.deallocate(x1)

        # Fill KV cache with the prompt K,V
        keys = self.layer_past[0]
        values = self.layer_past[1]

        keys_updated = ttnn.concat([keys, k], dim=2)
        values_updated = ttnn.concat([values, v], dim=2)

        # Update the cache references
        self.layer_past[0] = keys_updated
        self.layer_past[1] = values_updated

        # SDPA with full prompt (causal masking for prompt)
        tt_y = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Free early
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        tt_y = ttnn.experimental.nlp_concat_heads(tt_y)

        # output projection
        x2 = self.c_proj(tt_y)

        ttnn.deallocate(tt_y)

        return x2

    def pad_idx_to_32(self, idx):
        B, n, S, d = idx.shape

        S_new = ((S + 31) // 32) * 32
        pad_len = S_new - S

        if pad_len == 0:
            return idx, S

        pad_extra = ttnn.full(
            (B, n, pad_len, d), self.pad_id, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
        )
        idx_padded = ttnn.concat([idx, pad_extra], dim=2)

        return idx_padded

    def build_base_padding_mask(self, seq_lens):
        max_len = int(ttnn.to_torch(seq_lens).max().item())
        NEG = -1e4  # TT-safe
        B = self.B

        # positions: [B, max_len]
        positions = ttnn.arange(
            start=0,
            end=max_len,
            step=1,
            device=self.device,
            dtype=ttnn.int32,
        )
        positions = ttnn.unsqueeze(positions, 0)  # [1, max_len]
        positions = ttnn.repeat(positions, (B, 1))  # [B, max_len]

        seq_lens_u = ttnn.unsqueeze(seq_lens, 1)  # [B, 1]
        valid = ttnn.lt(positions, seq_lens_u)  # bool [B, max_len]
        valid = ttnn.to_layout(valid, ttnn.TILE_LAYOUT)

        # base mask: [B, 1, 1, max_len]
        mask = ttnn.zeros((B, 1, 1, max_len), dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)

        # invalid positions → NEG
        valid = ttnn.unsqueeze(valid, 1)  # [B, 1, max_len]
        valid = ttnn.unsqueeze(valid, 1)  # [B, 1, 1, max_len]

        neg_tensor = ttnn.full(mask.shape, NEG, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        mask = ttnn.where(valid, mask, neg_tensor)
        return mask

    def expand_mask(self, base_mask, pos_int):
        B, _, _, max_len = base_mask.shape
        NEG = -1e4

        S_new = ((pos_int + 31) // 32) * 32

        mask = base_mask

        # pad sequence dim
        if pos_int > max_len:
            extra_seq = ttnn.zeros(
                (B, 1, 1, pos_int - max_len), dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
            )
            mask = ttnn.concat([mask, extra_seq], dim=3)

        # pad head/row dim to S_new (NEG)
        if mask.shape[2] < S_new:
            extra_rows = ttnn.full(
                (B, 1, S_new - mask.shape[2], mask.shape[3]),
                NEG,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
            mask = ttnn.concat([mask, extra_rows], dim=2)

        # pad col dim to S_new (NEG)
        if mask.shape[3] < S_new:
            extra_cols = ttnn.full(
                (B, 1, mask.shape[2], S_new - mask.shape[3]),
                NEG,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
            mask = ttnn.concat([mask, extra_cols], dim=3)

        return mask

    def forward_decode(self, x: ttnn.Tensor, current_pos, base_mask) -> ttnn.Tensor:
        x1 = self.c_attn(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            input=x1, input_kv=None, num_heads=self.n_head, num_kv_heads=self.n_head, transpose_k_heads=False
        )
        ttnn.deallocate(x1)

        # Get current position as integer for cache update
        pos_int = ttnn.to_torch(current_pos)[0].item()
        pos_int += 1

        # Update cache at current position with new K,V
        keys = self.layer_past[0]
        values = self.layer_past[1]

        keys_updated = ttnn.concat([keys, k], dim=2)
        values_updated = ttnn.concat([values, v], dim=2)

        # Update the cache references
        self.layer_past[0] = keys_updated
        self.layer_past[1] = values_updated

        # SDPA with cached K,V (causal not needed since we only have past)
        attn_mask = self.expand_mask(base_mask, pos_int)
        q_new = self.pad_idx_to_32(q)
        k_new = self.pad_idx_to_32(keys_updated)
        v_new = self.pad_idx_to_32(values_updated)

        tt_y = ttnn.transformer.scaled_dot_product_attention(q_new, k_new, v_new, is_causal=False, attn_mask=attn_mask)

        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(q)

        # Regular concat (works for decode too)
        tt_y = ttnn.experimental.nlp_concat_heads(tt_y)

        # output projection
        x2 = self.c_proj(tt_y)

        ttnn.deallocate(tt_y)

        return x2
