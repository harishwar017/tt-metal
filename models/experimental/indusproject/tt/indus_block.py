# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch.nn as nn
import ttnn
import models.experimental.indusproject.tt.indus_mlp as indus_mlp
import models.experimental.indusproject.tt.indus_attention as indus_attention


class TtBlock(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()

        self.device = device
        self.config = config

        self.beta_1 = ttnn.load_tensor(
            tt_cache_path + base_address + ".ln_1.bias" + str(dtype) + ".tensorbin", device=device
        )
        self.beta_1 = ttnn.to_layout(self.beta_1, ttnn.TILE_LAYOUT)

        self.gamma_1 = ttnn.load_tensor(
            tt_cache_path + base_address + ".ln_1.weight" + str(dtype) + ".tensorbin", device=device
        )
        self.gamma_1 = ttnn.to_layout(self.gamma_1, ttnn.TILE_LAYOUT)

        self.ln_1 = ttnn.layer_norm

        self.attn = indus_attention.TtCausalSelfAttention(config, f"{base_address}.attn", device, tt_cache_path, dtype)

        self.beta_2 = ttnn.load_tensor(
            tt_cache_path + base_address + ".ln_2.bias" + str(dtype) + ".tensorbin", device=device
        )
        self.beta_2 = ttnn.to_layout(self.beta_2, ttnn.TILE_LAYOUT)

        self.gamma_2 = ttnn.load_tensor(
            tt_cache_path + base_address + ".ln_2.weight" + str(dtype) + ".tensorbin", device=device
        )
        self.gamma_2 = ttnn.to_layout(self.gamma_2, ttnn.TILE_LAYOUT)

        self.ln_2 = ttnn.layer_norm

        self.mlp = indus_mlp.TtMLP(f"{base_address}.mlp", self.config, device, tt_cache_path, dtype)

    def forward_prefill(
        self, x: ttnn.Tensor, current_pos: int = 0, idx: Optional = None, pad_mask: Optional = None
    ) -> ttnn.Tensor:
        tmp = self.attn.forward_prefill(
            self.ln_1(x, epsilon=1e-5, weight=self.gamma_1, bias=self.beta_1), current_pos=current_pos
        )
        x = ttnn.add(x, tmp)

        tmp = self.mlp.forward(self.ln_2(x, epsilon=1e-5, weight=self.gamma_2, bias=self.beta_2))
        x = ttnn.add(x, tmp)

        return x

    def forward_decode(
        self, x: ttnn.Tensor, current_pos: ttnn.Tensor, idx: Optional = None, pad_mask: Optional = None
    ) -> ttnn.Tensor:
        tmp = self.attn.forward_decode(  # ← FIX: was calling forward_prefill!
            self.ln_1(x, epsilon=1e-5, weight=self.gamma_1, bias=self.beta_1), current_pos=current_pos
        )
        x = ttnn.add(x, tmp)

        tmp = self.mlp.forward(self.ln_2(x, epsilon=1e-5, weight=self.gamma_2, bias=self.beta_2))
        x = ttnn.add(x, tmp)

        return x
