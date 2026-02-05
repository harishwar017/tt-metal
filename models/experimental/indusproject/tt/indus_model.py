# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear

import ttnn

import models.experimental.indusproject.tt.indus_block as indus_block
from models.experimental.indusproject.indusproject_utils import unpad_from_zero


class TtGPT(nn.Module):
    def __init__(self, config, device, tt_cache_path, dtype):
        super().__init__()

        assert config.vocab_size is not None

        self.config = config
        self.config.block_size = 1024
        base_address = f"transformer"
        self.device = device

        self.beta = ttnn.load_tensor(tt_cache_path + base_address + ".ln_f.bias" + str(dtype) + ".tensorbin")
        self.beta = ttnn.to_device(self.beta, device)

        self.gamma = ttnn.load_tensor(tt_cache_path + base_address + ".ln_f.weight" + str(dtype) + ".tensorbin")
        self.gamma = ttnn.to_device(self.gamma, device)

        self.pad_id = config.eos_token_id

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
            block = indus_block.TtBlock(self.config, f"{base_address}.h.{i}", self.device, tt_cache_path, dtype)
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

    def forward(self, idx, seq_lens) -> ttnn.Tensor:
        b, t = idx.shape

        # assert (
        #     t <= self.config.block_size
        # ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = ttnn.embedding(idx, self.tt_wte_weight)
        pos_emb = self.pos_cache[:, :t, :]

        x = ttnn.add(tok_emb, pos_emb)
        x = ttnn.unsqueeze(x, dim=1)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # pad_mask = self.h[0].attn.make_pad_mask_torch(idx)
        for block in self.h:
            x = block.forward(x, idx=idx, seq_lens=seq_lens)
        x = self.ln_f(x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(x)

        return logits

    def pad_idx_to_32(self, idx):
        B, S = idx.shape

        S_new = ((S + 31) // 32) * 32
        pad_len = S_new - S

        if pad_len == 0:
            return idx, S

        pad_extra = ttnn.full((B, pad_len), self.pad_id, dtype=ttnn.uint32, device=self.device, layout=ttnn.TILE_LAYOUT)
        idx_padded = ttnn.concat([idx, pad_extra], dim=1)

        return idx_padded

    def generate_1(
        self,
        idx,
        seq_lens,
        max_new_tokens: int = 20,
        eos_id: int = None,
    ):
        vocab_size = int(self.config.vocab_size)

        # Convert to TT once
        if not isinstance(idx, ttnn.Tensor):
            idx = ttnn.from_torch(
                idx.to(torch.uint32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
            )

        B = idx.shape[0]

        # Track which sequences are finished
        finished = torch.zeros(B, dtype=torch.bool)

        for i in range(max_new_tokens):
            print(f"predicting: {i}")
            # idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]

            # Forward

            tt_logits = self.forward(idx, seq_lens)

            tt_logits = ttnn.squeeze(tt_logits, dim=1)

            # Take last timestep: [B, V]
            last_logits = tt_logits[:, -1, :]

            # Argmax: [B]
            idx_next = ttnn.argmax(last_logits, dim=-1)

            # Make [B,1]
            idx_next = ttnn.unsqueeze(idx_next, dim=1)
            idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

            idx = ttnn.concat([idx, idx_next], dim=1)

            # EOS handling (CPU)
            if eos_id is not None:
                next_tok = ttnn.to_torch(idx_next).squeeze(1)  # [B]

                finished |= next_tok == eos_id

                # Stop if all finished
                if finished.all():
                    break

        return idx

    def benchmark_generate(
        self,
        idx,
        max_new_tokens,
        eos_id,
        runs=5,
    ):
        ttfts = []
        tpss = []

        # Convert to TT once (TILE layout)
        if not isinstance(idx, ttnn.Tensor):
            base_idx = ttnn.from_torch(
                idx.to(torch.uint32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            base_idx = idx

        for _ in range(runs):
            # Reset for each run
            idx = ttnn.clone(base_idx)

            B = idx.shape[0]
            # Track which sequences are finished
            finished = torch.zeros(B, dtype=torch.bool)

            start = time.perf_counter()

            first_token_time = None
            tokens_generated = 0

            for step in range(max_new_tokens):
                idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]

                tt_logits = self.forward(idx_cond)

                tt_logits = ttnn.squeeze(tt_logits, dim=1)
                last_logits = tt_logits[:, -1, :]

                idx_next = ttnn.argmax(last_logits, dim=-1)
                idx_next = ttnn.unsqueeze(idx_next, 1)

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                tokens_generated += B

                # Safe: both TILE
                idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)
                idx = ttnn.concat([idx, idx_next], dim=1)

                # EOS handling (CPU)
                if eos_id is not None:
                    next_tok = ttnn.to_torch(idx_next).squeeze(1)  # [B]

                    finished |= next_tok == eos_id

                    # Stop if all finished
                    if finished.all():
                        break

            end = time.perf_counter()

            ttft = first_token_time - start
            total_decode = end - first_token_time
            tps = tokens_generated / total_decode

            ttfts.append(ttft)
            tpss.append(tps)

        return {
            "ttft_avg": sum(ttfts) / len(ttfts),
            "tps_avg": sum(tpss) / len(tpss),
            "ttft_runs": ttfts,
            "tps_runs": tpss,
            "idx": idx,
        }
