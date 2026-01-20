# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.common.helper_funcs import Linear

import ttnn

import models.experimental.nanogpt.tt.nanogpt_block as nanogpt_block
from models.experimental.nanogpt.nanogpt_utils import unpad_from_zero

from models.common.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


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

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(self.config.block_size, config.n_embd)

        self.wte.weight = torch.nn.Parameter(torch.load(tt_cache_path + "transformer.wte.weight.pt"))

        self.wpe.weight = torch.nn.Parameter(torch.load(tt_cache_path + "transformer.wpe.weight.pt"))

        blocks = []

        for i in range(config.n_layer):
            block = nanogpt_block.TtBlock(self.config, f"{base_address}.h.{i}", self.device, tt_cache_path, dtype)
            blocks.append(block)

        self.h = nn.ModuleList(blocks)

        self.ln_f = ttnn.layer_norm

        tt_lm_weight = ttnn.load_tensor(tt_cache_path + "lm_head.weight" + str(dtype) + ".tensorbin")

        weight = unpad_from_zero(tt_lm_weight, (1, 1, self.config.vocab_size, self.config.n_embd))
        weight_torch = weight
        weight = torch_to_tt_tensor_rm(weight, device=self.device)

        self.lm_head = Linear(self.config.n_embd, self.config.vocab_size, weight)

        self.wte.weight = nn.Parameter(weight_torch.squeeze())  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx) -> ttnn.Tensor:
        # Convert TTNN tensor to PyTorch if needed
        if isinstance(idx, ttnn.Tensor):
            idx = tt_to_torch_tensor(idx).to(dtype=torch.int64)

        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = ttnn.arange(0, t, 1)
        pos = tt_to_torch_tensor(pos)
        pos = pos.squeeze(0).squeeze(0)
        pos = pos.to(dtype=torch.int64)
        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        tt_tok_emb = torch_to_tt_tensor_rm(tok_emb, self.device)
        tt_pos_emb = torch_to_tt_tensor_rm(pos_emb, self.device)
        tt_tok_emb = ttnn.permute(tt_tok_emb, (0, 2, 1, 3))
        tt_pos_emb = ttnn.permute(tt_pos_emb, (0, 2, 1, 3))
        tt_x = ttnn.add(tt_tok_emb, tt_pos_emb)
        tt_tok_emb.deallocate()
        tt_pos_emb.deallocate()
        tt_x = ttnn.permute(tt_x, (0, 2, 1, 3))
        for block in self.h:
            tt_x = block.forward(tt_x)
        tt_x = self.ln_f(tt_x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(tt_x)

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
            logits = tt_to_torch_tensor(tt_logits)  # shape: [1, 1, T, vocab_size]

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
        top_k=None,
    ) -> torch.Tensor:
        B = idx.size(0)
        vocab_size = int(self.config.vocab_size)

        # PRE-CALCULATE reciprocal temperature to avoid ttnn.reciprocal in loop
        inv_temp = 1.0 / temperature

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            # 1. Forward pass (Keep on device)
            tt_logits = self.forward(idx_cond)
            print(f"tt_logit: {tt_logits.shape}")

            # 2. Slice the last token's logits
            # Instead of fallback_ops, use ttnn.slice if possible,
            # or move to torch ONLY ONCE here.
            logits = tt_to_torch_tensor(tt_logits)
            # print("Logits shape:", logits.shape)

            # [Batch, 1, seq_len, hidden] -> Get last token
            # Adjust these indices based on your actual model output shape
            logits = logits[:, :, -1, :]
            logits = logits.view(B, -1)
            logits = logits[:, :vocab_size]

            # 3. Perform math in Torch (much faster for small vectors than H2D/D2H overhead)
            if temperature != 1.0:
                logits = logits * inv_temp

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)

            # 4. Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # 5. Append
            idx = torch.cat((idx, idx_next), dim=1)

            # CRITICAL: If you use any intermediate TT tensors,
            # deallocate them or ensure they are overwritten.
            # del tt_logits

        return idx

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
            logits = tt_to_torch_tensor(tt_logits)
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
