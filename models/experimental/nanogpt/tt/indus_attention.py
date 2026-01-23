# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
import math
from models.common.helper_funcs import Linear


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024

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

    def const_tensor(self, shape, value):
        return ttnn.full(shape, value, device=self.device, dtype=ttnn.bfloat16)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Convert to ROW_MAJOR for reshape operations
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        (_, B, T, C) = x.padded_shape  # batch size, sequence length, embedding dimensionality (n_embd)

        x1 = self.c_attn(x)
        # Ensure x1 is ROW_MAJOR for the split operation
        x1 = ttnn.to_layout(x1, ttnn.ROW_MAJOR_LAYOUT)

        x1 = ttnn.squeeze(x1, dim=0)
        q = x1[:, :, 0 : self.n_embd]
        k = x1[:, :, self.n_embd : 2 * self.n_embd]
        v = x1[:, :, 2 * self.n_embd : 3 * self.n_embd]

        q = ttnn.reshape(q, (B, T, self.n_head, C // self.n_head))
        q = ttnn.permute(q, (0, 2, 1, 3))  # (B, n_head, T, head_dim)
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)

        k = ttnn.reshape(k, (B, T, self.n_head, C // self.n_head))
        k = ttnn.permute(k, (0, 2, 1, 3))  # (B, n_head, T, head_dim)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)

        v = ttnn.reshape(v, (B, T, self.n_head, C // self.n_head))
        v = ttnn.permute(v, (0, 2, 1, 3))  # (B, n_head, T, head_dim)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

        # manual implementation of attention
        key_layer_transposed = ttnn.transpose(k, -2, -1)
        att = ttnn.matmul(q, key_layer_transposed)
        ttnn.deallocate(key_layer_transposed)
        ttnn.deallocate(q)

        scale_factor = 1.0 / math.sqrt(k.padded_shape[-1])
        att = ttnn.multiply(att, scale_factor)

        causal_mask = ttnn.slice(self.tt_bias, [0, 0, 0, 0], [1, 1, T, T])
        causal_mask = ttnn.gt(causal_mask, 0.0)
        neg_inf = ttnn.full(att.shape, -1e9, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        tt_att = ttnn.where(causal_mask, att, neg_inf)
        ttnn.deallocate(neg_inf)
        ttnn.deallocate(causal_mask)
        ttnn.deallocate(att)

        tt_att = ttnn.softmax(tt_att)  # Using ttnn.softmax reduces pcc from 0.99 to 0.98 for whole model
        # Convert to TILE for matmul with v
        tt_att = ttnn.to_layout(tt_att, ttnn.TILE_LAYOUT)

        tt_y = ttnn.matmul(tt_att, v)
        ttnn.deallocate(tt_att)
        ttnn.deallocate(v)

        tt_y = ttnn.transpose(tt_y, 1, -2)
        tt_y = ttnn.to_layout(tt_y, ttnn.ROW_MAJOR_LAYOUT)
        tt_y = ttnn.reshape_on_device(tt_y, 1, B, T, C)
        tt_y = ttnn.to_layout(tt_y, ttnn.TILE_LAYOUT)

        ttnn.deallocate(k)
        ttnn.deallocate(x1)  # - H should look for the most optimal place to deallocate this

        # output projection
        x2 = self.c_proj(tt_y)
        ttnn.deallocate(tt_y)
        return x2
