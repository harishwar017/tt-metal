# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
from models.common.helper_funcs import Linear

import ttnn

import models.experimental.nanogpt.tt.indus_block as indus_block
from models.experimental.nanogpt.nanogpt_utils import unpad_from_zero



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
        # Convert TTNN tensor to PyTorch if needed
        # if isinstance(idx, ttnn.Tensor):
        #     idx = ttnn.to_torch(idx).to(dtype=torch.int64)
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)
        tt_pos = ttnn.from_torch(pos, device=self.device)

        # 1. Token Embedding Lookup (on-device)
        tok_emb = ttnn.embedding(idx, self.tt_wte_weight)

        # 2. Position Embedding Lookup (on-device)
        pos_emb = ttnn.embedding(tt_pos, self.tt_wpe_weight)

        tt_tok_emb = ttnn.permute(tok_emb, (0, 2, 1))
        tt_pos_emb = ttnn.permute(pos_emb, (0, 2, 1))
        # 3. Combine [Batch, 1, Seq_Len, Hidden]
        x = ttnn.add(tok_emb, pos_emb)

        # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        # tt_x = ttnn.permute(tt_x, (0, 2, 1))

        shape = x.padded_shape
        x = ttnn.reshape(x, (1, shape[0], shape[1], shape[2]))

        for block in self.h:
            x = block.forward(x)
        x = self.ln_f(x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(x)

        tt_tok_emb.deallocate()
        tt_pos_emb.deallocate()

        return logits

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k=None,
    ) -> torch.Tensor:
        """
        idx: LongTensor (B, T)
        Returns: LongTensor (B, T + max_new_tokens)
        """
        B = idx.size(0)
        vocab_size = int(self.config.vocab_size)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            # Forward TT model -> logits on TT
            tt_logits = self.forward(idx_cond)

            # Convert to torch first
            logits = ttnn.to_torch(tt_logits)  # shape: [1, 1, T, vocab_size]

            # Get last token logits: [1, 1, 1, vocab_size] -> [1, vocab_size]
            logits = logits[:, :, -1, :B]  # Last time step, first B vocab entries
            logits = logits.squeeze(1).squeeze(0)  # Remove batch dims if they exist

            # Ensure logits is [B, vocab_size] or [vocab_size]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # Add batch dim if missing

            # Slice to true vocab size
            logits = logits[:, :vocab_size]

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Optional top-k
            if top_k is not None:
                k = min(int(top_k), logits.size(-1))
                if k > 0:
                    v, _ = torch.topk(logits, k, dim=-1)
                    threshold = v[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Softmax
            probs = torch.softmax(logits, dim=-1)

            # Safety checks
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def generate_1(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k=None,
    ) -> torch.Tensor:
        B = idx.size(0)
        vocab_size = int(self.config.vocab_size)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            # 1. Forward pass (Keep on device)
            tt_logits = self.forward(idx_cond)

            # 2. Slice the last token's logits
            # Instead of fallback_ops, use ttnn.slice if possible,
            # or move to torch ONLY ONCE here.
            logits = ttnn.to_torch(tt_logits)
            # print("Logits shape:", logits.shape)

            # [Batch, 1, seq_len, hidden] -> Get last token
            # Adjust these indices based on your actual model output shape
            logits = logits[:, :, -1, :]
            logits = logits.view(B, -1)
            logits = logits[:, :vocab_size]

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
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)

            # CRITICAL: If you use any intermediate TT tensors,
            # deallocate them or ensure they are overwritten.
            # del tt_logits

        return idx

    def generate_2(
        self,
        idx: None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k=None,
    ) -> torch.Tensor:
        B = idx.shape[0]
        vocab_size = int(self.config.vocab_size)

        # PRE-CALCULATE reciprocal temperature to avoid ttnn.reciprocal in loop

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

            # CRITICAL: If you use any intermediate TT tensors,
            # deallocate them or ensure they are overwritten.
            # del tt_logits

        return idx

    def generate_timed(
        self,
        idx: None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k=None,
        return_timing: bool = True,
    ):
        B = idx.shape[0]
        vocab_size = int(self.config.vocab_size)

        # Ensure we start timing after previous async work finishes
        ttnn.synchronize_device(self.device)
        t_start = time.perf_counter()

        ttft_s = None
        decode_start = None

        for i in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]

            # ---- Forward pass ----
            tt_logits = self.forward(idx_cond)

            # logits shape handling
            tt_logits = ttnn.squeeze(tt_logits, dim=1)
            tt_logits = tt_logits[:, -1, :vocab_size]
            tt_logits = ttnn.squeeze(tt_logits, dim=1)

            # ---- Next token ----
            if do_sample:
                # NOTE: your sampling branch currently uses torch tensors named `logits`
                # but `tt_logits` is TT tensor. Unless you convert, this path is wrong.
                # For benchmarking speed, always use greedy decoding.
                raise NotImplementedError("Sampling path mixes torch/ttnn tensors; benchmark with do_sample=False.")
            else:
                idx_next = ttnn.argmax(tt_logits, dim=-1, keepdim=True)
                idx = ttnn.concat([idx, idx_next], dim=1)

            # ---- TTFT measurement ----
            if i == 0:
                # First token has been produced, wait for device to finish this iteration
                ttnn.synchronize_device(self.device)
                t_after_first = time.perf_counter()
                ttft_s = t_after_first - t_start

                # Start decode timing AFTER first token
                decode_start = time.perf_counter()

        # final sync so total timing is correct
        ttnn.synchronize_device(self.device)
        t_end = time.perf_counter()

        # ---- metrics ----
        total_time_s = t_end - t_start
        decode_time_s = (t_end - decode_start) if decode_start is not None else 0.0

        decode_tokens = max(max_new_tokens - 1, 0)
        decode_tps = (decode_tokens / decode_time_s) if decode_time_s > 0 else float("inf")
        overall_tps = (max_new_tokens / total_time_s) if total_time_s > 0 else float("inf")

        timing = {
            "ttft_s": ttft_s,
            "decode_tps": decode_tps,
            "overall_tps": overall_tps,
            "total_time_s": total_time_s,
            "decode_time_s": decode_time_s,
            "max_new_tokens": max_new_tokens,
            "batch_size": B,
        }

        return (idx, timing) if return_timing else idx

    def generate_full(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 20,
        num_beams: int = 5,
        do_sample: bool = True,
        early_stopping: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        """
        Complex generation loop for Tenstorrent Blackhole.
        Forward pass happens on TT Device; Search logic happens on Host CPU.
        """
        vocab_size = int(self.config.vocab_size)
        batch_size = idx.size(0)

        # If num_beams > 1, we need to replicate the input for each beam
        if num_beams > 1:
            idx = idx.repeat_interleave(num_beams, dim=0)

        cur_len = idx.size(1)

        for _ in range(max_new_tokens):
            # 1. Forward pass on Blackhole
            # We crop to block_size to ensure TT kernels don't overflow L1/DRAM
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            tt_logits = self.forward(idx_cond)

            # 2. Move only the last token logits to Host CPU
            # Shape: [B*num_beams, 1, seq_len, Hidden] -> [B*num_beams, vocab_size]
            logits = ttnn.to_torch(tt_logits)
            logits = logits[:, 0, -1, :vocab_size]

            # 3. Apply Repetition Penalty
            for i in range(logits.size(0)):
                for token in set(idx[i].tolist()):
                    if logits[i, token] < 0:
                        logits[i, token] *= repetition_penalty
                    else:
                        logits[i, token] /= repetition_penalty

            # 4. Apply No-Repeat N-Gram Penalty
            if no_repeat_ngram_size > 0:
                for i in range(logits.size(0)):
                    tokens = idx[i].tolist()
                    ngram_prev = tokens[-(no_repeat_ngram_size - 1) :]
                    if len(ngram_prev) == no_repeat_ngram_size - 1:
                        for j in range(len(tokens) - (no_repeat_ngram_size - 1)):
                            if tokens[j : j + no_repeat_ngram_size - 1] == ngram_prev:
                                forbidden_token = tokens[j + no_repeat_ngram_size - 1]
                                logits[i, forbidden_token] = float("-inf")

            # 5. Temperature Scaling
            logits = logits / max(temperature, 1e-5)

            # 6. Top-K and Top-P (Nucleus) Sampling
            if do_sample:
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    for i in range(logits.size(0)):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        logits[i, indices_to_remove] = float("-inf")

            # 7. Selection (Beam vs Sample)
            if not do_sample and num_beams > 1:
                # Simple Beam Search placeholder (Argmax for now)
                # Full Beam Search requires tracking multiple sequences;
                # Most users use Sample for LLM performance on TT
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            # Early Stopping check (if next token is EOS, would go here)
            # if (idx_next == eos_token_id).all(): break

        return idx
