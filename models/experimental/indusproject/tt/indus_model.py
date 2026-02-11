# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear
from typing import Optional
import ttnn

import models.experimental.indusproject.tt.indus_block as indus_block
from models.experimental.indusproject.indusproject_utils import unpad_from_zero


class TtGPT(nn.Module):
    def __init__(self, config, device, tt_cache_path, dtype, B):
        super().__init__()

        assert config.vocab_size is not None

        self.config = config
        self.config.block_size = 1024
        base_address = f"transformer"
        self.device = device
        self.B = B

        self.beta = ttnn.load_tensor(tt_cache_path + base_address + ".ln_f.bias" + str(dtype) + ".tensorbin")
        self.beta = ttnn.to_device(self.beta, device)

        self.gamma = ttnn.load_tensor(tt_cache_path + base_address + ".ln_f.weight" + str(dtype) + ".tensorbin")
        self.gamma = ttnn.to_device(self.gamma, device)

        wte_torch = torch.load(tt_cache_path + "transformer.wte.weight.pt")
        wpe_torch = torch.load(tt_cache_path + "transformer.wpe.weight.pt")
        # Keep embeddings in ROW_MAJOR as they are lookup tables
        self.tt_wte_weight = ttnn.from_torch(
            wte_torch, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )
        self.tt_wpe_weight = ttnn.from_torch(
            wpe_torch, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

        blocks = []

        for i in range(config.n_layer):
            block = indus_block.TtBlock(self.config, f"{base_address}.h.{i}", self.device, tt_cache_path, dtype, B)
            blocks.append(block)

        self.h = nn.ModuleList(blocks)

        self.ln_f = ttnn.layer_norm

        tt_lm_weight = ttnn.load_tensor(tt_cache_path + "lm_head.weight" + str(dtype) + ".tensorbin")

        weight = unpad_from_zero(tt_lm_weight, (1, 1, self.config.vocab_size, self.config.n_embd))
        weight_torch = weight
        weight = ttnn.from_torch(weight, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.lm_head = Linear(self.config.n_embd, self.config.vocab_size, weight)

        self.tt_pos_cache = ttnn.arange(
            start=0,
            end=self.config.block_size,
            step=1,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ).reshape((1, self.config.block_size))

        self.pos_cache = ttnn.embedding(self.tt_pos_cache, self.tt_wpe_weight)

    def forward_prefill(self, idx, current_pos) -> ttnn.Tensor:
        """
        Prefill: Process entire prompt and populate KV cache
        idx: [batch, seq_len] token indices (TTNN tensor)
        current_pos: starting position (usually 0)
        """
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = ttnn.embedding(idx, self.tt_wte_weight)
        pos_emb = self.pos_cache[:, :t, :]

        x = ttnn.add(tok_emb, pos_emb)
        x = ttnn.unsqueeze(x, dim=1)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Pass through transformer blocks
        for block in self.h:
            x = block.forward_prefill(x, current_pos=current_pos)

        x = self.ln_f(x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(x)

        return logits

    def forward_decode(self, idx, current_pos, seq_lens) -> ttnn.Tensor:
        """
        Decode: Process single new token using cached K,V
        idx: [batch, 1] single token index (TTNN tensor)
        current_pos: [batch] position tensor
        """
        b, t = idx.shape
        assert t == 1, "Decode should process exactly one token"

        # Get position index for embedding lookup
        pos_idx = ttnn.to_torch(current_pos)[0].item()

        # Token and position embeddings for single token
        tok_emb = ttnn.embedding(idx, self.tt_wte_weight)
        pos_emb = self.pos_cache[:, pos_idx : pos_idx + 1, :]

        x = ttnn.add(tok_emb, pos_emb)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.unsqueeze(x, dim=1)

        base_mask = self.h[0].attn.build_base_padding_mask(seq_lens)
        # Pass through transformer blocks with position tracking
        for block in self.h:
            x = block.forward_decode(x, current_pos=current_pos, base_mask=base_mask)

        x = self.ln_f(x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        idx,
        seq_lens: Optional = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        eos_id: int = None,
        do_sample: bool = False,  # keep greedy for now
        top_k=None,
    ):
        # Convert to TT once
        if not isinstance(idx, ttnn.Tensor):
            idx = ttnn.from_torch(
                idx.to(torch.uint32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
            )

        B = idx.shape[0]
        prompt_len = idx.shape[1]

        # Track which sequences are finished
        finished = torch.zeros(B, dtype=torch.bool)

        # PREFILL PHASE: Process entire prompt and populate cache
        tt_logits = self.forward_prefill(idx, current_pos=0)

        # Get logits for last token of prompt
        tt_logits = ttnn.squeeze(tt_logits, dim=1)
        last_logits = tt_logits[:, -1, :]  # [B, vocab_size]

        # Sample first token
        idx_next = ttnn.argmax(last_logits, dim=-1)  # [B]
        idx_next = ttnn.unsqueeze(idx_next, dim=1)  # [B, 1]
        idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

        # Concatenate generated token
        idx = ttnn.concat([idx, idx_next], dim=1)
        current_pos = ttnn.full(
            (1,), prompt_len + 1, dtype=ttnn.int32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        one = ttnn.full((1,), 1, dtype=ttnn.int32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # DECODE PHASE: Generate remaining tokens one at a time
        for _ in range(max_new_tokens - 1):
            # Forward decode with single token
            tt_logits = self.forward_decode(idx_next, current_pos=current_pos, seq_lens=seq_lens)

            tt_logits = ttnn.squeeze(tt_logits, dim=1)
            last_logits = tt_logits[:, -1, :]

            # Sample next token
            idx_next = ttnn.argmax(last_logits, dim=-1)
            idx_next = ttnn.unsqueeze(idx_next, dim=1)
            idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

            # Concatenate
            idx = ttnn.concat([idx, idx_next], dim=1)
            current_pos = ttnn.add(current_pos, one)

            # EOS handling
            if eos_id is not None:
                next_tok = ttnn.to_torch(idx_next).squeeze(1)
                finished |= next_tok == eos_id
                if finished.all():
                    break

        return idx
