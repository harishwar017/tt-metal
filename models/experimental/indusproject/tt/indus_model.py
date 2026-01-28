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

    def forward(self, idx) -> ttnn.Tensor:
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tt_pos = ttnn.arange(
            start=0, end=t, step=1, device=self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        ).reshape((1, t))

        # 1. Token Embedding Lookup (on-device)
        tok_emb = ttnn.embedding(idx, self.tt_wte_weight)
        # print("tok_emb shape:", tok_emb.shape)

        # 2. Position Embedding Lookup (on-device)
        pos_emb = ttnn.embedding(tt_pos, self.tt_wpe_weight)
        # print("pos_emb shape:", pos_emb.shape)

        tt_tok_emb = ttnn.permute(tok_emb, (0, 2, 1))
        tt_pos_emb = ttnn.permute(pos_emb, (0, 2, 1))
        # 3. Combine [Batch, 1, Seq_Len, Hidden]
        x = ttnn.add(tok_emb, pos_emb)
        # print("x shape after adding pos and tok emb:", x.shape)

        # shape = x.padded_shape
        # x = ttnn.reshape(x, (1, shape[0], shape[1], shape[2]))
        x = ttnn.unsqueeze(x, dim=1)

        pad_mask = self.h[0].attn.make_pad_mask(idx)
        for block in self.h:
            x = block.forward(x, idx, pad_mask)
        x = self.ln_f(x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(x)

        tt_tok_emb.deallocate()
        tt_pos_emb.deallocate()

        return logits

    def generate(
        self,
        idx: None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        eos_id: int = None,
        do_sample: bool = True,
        top_k=None,
    ) -> torch.Tensor:
        # B = idx.shape[0]
        vocab_size = int(self.config.vocab_size)

        # PRE-CALCULATE reciprocal temperature to avoid ttnn.reciprocal in loop
        if not isinstance(idx, ttnn.Tensor):
            idx = ttnn.from_torch(
                idx.to(torch.uint32), device=self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]

            # 1. Forward pass (Keep on device)
            tt_logits = self.forward(idx_cond)

            tt_logits = ttnn.squeeze(tt_logits, dim=1)
            tt_logits = tt_logits[:, -1, :vocab_size]
            tt_logits = ttnn.squeeze(tt_logits, dim=1)

            if do_sample:
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, idx_next), dim=1)
            else:
                # Greedy decoding
                idx_next = ttnn.argmax(tt_logits, dim=-1, keepdim=True)
                idx = ttnn.concat([idx, idx_next], dim=1)

                next_tok = ttnn.to_torch(idx_next)[0, 0].item()
                if next_tok == eos_id:
                    break

            # CRITICAL: If you use any intermediate TT tensors,
            # deallocate them or ensure they are overwritten.
            # del tt_logits

        return idx
        # return ttnn.to_torch(idx[:, prompt_len:])

    def generate_1(
        self,
        idx,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        eos_id: int = None,
        do_sample: bool = False,  # keep greedy for now
        top_k=None,
    ):
        print(f"eos id: {eos_id}")

        vocab_size = int(self.config.vocab_size)

        # Convert to TT once
        if not isinstance(idx, ttnn.Tensor):
            idx = ttnn.from_torch(
                idx.to(torch.uint32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        B = idx.shape[0]

        # Track which sequences are finished
        finished = torch.zeros(B, dtype=torch.bool)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]

            # Forward
            tt_logits = self.forward(idx_cond)

            tt_logits = ttnn.squeeze(tt_logits, dim=1)

            # Take last timestep: [B, V]
            last_logits = tt_logits[:, -1, :]

            # Argmax: [B]
            idx_next = ttnn.argmax(last_logits, dim=-1)

            # Make [B,1]
            idx_next = ttnn.unsqueeze(idx_next, dim=1)

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

        # Convert to TT once
        if not isinstance(idx, ttnn.Tensor):
            idx = ttnn.from_torch(
                idx.to(torch.uint32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        for _ in range(runs):
            B = idx.shape[0]

            start = time.perf_counter()

            # finished = torch.zeros(B, dtype=torch.bool)
            first_token_time = None
            tokens_generated = 0

            # cur_idx = idx.clone()

            for step in range(max_new_tokens):
                step_start = time.perf_counter()
                idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]
                tt_logits = self.forward(idx_cond)
                tt_logits = ttnn.squeeze(tt_logits, dim=1)
                last_logits = tt_logits[:, -1, :]
                idx_next = ttnn.argmax(last_logits, dim=-1)
                idx_next = ttnn.unsqueeze(idx_next, 1)

                # Force sync
                # next_tok = ttnn.to_torch(idx_next)

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                tokens_generated += B

                idx = ttnn.concat([idx, idx_next], dim=1)

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
        }
