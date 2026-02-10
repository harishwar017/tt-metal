# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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

        # Initialize KV cache
        self.head_dim = self.n_embd // self.n_head
        self.init_kv_cache()

    def init_kv_cache(self):
        """Initialize KV cache for incremental decoding"""
        batch_size = 2  # Single user initially

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

    def const_tensor(self, shape, value):
        return ttnn.full(shape, value, device=self.device, dtype=ttnn.bfloat16)

    def forward_prefill(self, x: ttnn.Tensor, current_pos: int = 0) -> ttnn.Tensor:
        """
        Prefill: Process entire prompt and populate KV cache
        x: [batch, seq_len, dim] - entire prompt sequence
        current_pos: starting position (usually 0)
        """
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

    def build_causal_mask(self, idx):
        NEG = -1e4  # safer than -inf on TT
        B = idx.shape[0]
        S = idx.padded_shape[2]

        # Lower triangular matrix
        causal = torch.tril(torch.ones((S, S), dtype=torch.bool))
        causal = causal.unsqueeze(0).unsqueeze(0)
        causal = causal.expand(B, 1, S, S)

        # Convert to float mask
        mask = torch.zeros((B, 1, S, S), dtype=torch.bfloat16)
        mask[~causal] = NEG

        return mask

    def forward_decode(self, x: ttnn.Tensor, current_pos: ttnn.Tensor) -> ttnn.Tensor:
        """
        Decode: Process single new token using cached K,V
        x: [batch, 1, dim] - single new token
        current_pos: [batch] - current position tensor
        """
        x1 = self.c_attn(x)

        # Use regular QKV head creation (works for single token)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            input=x1, input_kv=None, num_heads=self.n_head, num_kv_heads=None, transpose_k_heads=False
        )
        ttnn.deallocate(x1)
        # Get current position as integer for cache update
        pos_int = ttnn.to_torch(current_pos)[0].item()

        # Update cache at current position with new K,V
        keys = self.layer_past[0]
        values = self.layer_past[1]

        keys_updated = ttnn.concat([keys, k], dim=2)
        values_updated = ttnn.concat([values, v], dim=2)

        # Update the cache references
        self.layer_past[0] = keys_updated
        self.layer_past[1] = values_updated

        # SDPA with cached K,V (causal not needed since we only have past)
        tt_y = ttnn.transformer.scaled_dot_product_attention(q, keys_updated, values_updated, is_causal=False)

        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(q)

        # ttnn.deallocate(keys)
        # ttnn.deallocate(values)
        # ttnn.deallocate(keys_updated)
        # ttnn.deallocate(values_updated)

        # Regular concat (works for decode too)
        tt_y = ttnn.experimental.nlp_concat_heads(tt_y)

        # output projection
        x2 = self.c_proj(tt_y)

        ttnn.deallocate(tt_y)

        return x2
