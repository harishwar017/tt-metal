# def build_padding_mask(self, x, lengths, pos_int):
#     max_len = lengths.max().item()

#     NEG = -1e4  # TT-safe

#     B = lengths.shape[0]
#     S = pos_int
#     S_new = ((S + 31) // 32) * 32

#     # [B, max_len]
#     positions = torch.arange(max_len).unsqueeze(0).expand(B, max_len)

#     # True = valid token
#     valid = positions < lengths.unsqueeze(1)

#     # Convert to additive mask
#     mask = torch.zeros((B, 1, 1, max_len), dtype=torch.bfloat16)
#     mask[~valid.unsqueeze(1).unsqueeze(1)] = NEG

#     extra = torch.zeros(B, 1, 1, pos_int - max_len)
#     mask = torch.cat((mask, extra), dim=3)

#     extra = torch.zeros(B, 1, S_new - mask.shape[2], mask.shape[3]) + NEG
#     mask = torch.cat((mask, extra), dim=2)

#     extra = torch.zeros(B, 1, mask.shape[2], S_new - mask.shape[3]) + NEG
#     mask = torch.cat((mask, extra), dim=3)

#     return mask

# def build_padding_mask_ttnn(self, x, lengths, pos_int):
#     """
#     x        : unused (kept for signature parity)
#     lengths  : ttnn.Tensor, shape [B], int32 / int64 on device
#     pos_int  : python int (current sequence length S)
#     returns  : ttnn.Tensor, shape [B, 1, S_new, S_new], bfloat16
#     """

#     max_len = int(ttnn.to_torch(lengths).max().item())

#     # -------------------------
#     # constants / sizes
#     # -------------------------
#     NEG = -1e4  # TT-safe
#     B = lengths.shape[0]
#     S = pos_int
#     S_new = ((S + 31) // 32) * 32

#     # -------------------------
#     # positions: [B, max_len]
#     # -------------------------
#     # torch.arange(max_len).unsqueeze(0).expand(B, max_len)
#     positions = ttnn.arange(
#         start=0,
#         end=max_len,
#         step=1,
#         device=self.device,
#         dtype=ttnn.int32,
#     )
#     positions = ttnn.unsqueeze(positions, 0)              # [1, max_len]
#     positions = ttnn.repeat(positions, (B, 1))             # [B, max_len]

#     # -------------------------
#     # valid = positions < lengths.unsqueeze(1)
#     # -------------------------
#     lengths_u = ttnn.unsqueeze(lengths, 1)                  # [B, 1]
#     valid = ttnn.lt(positions, lengths_u)                   # bool [B, max_len]
#     valid = ttnn.to_layout(valid, ttnn.TILE_LAYOUT)

#     # -------------------------
#     # base mask: [B, 1, 1, max_len]
#     # -------------------------
#     mask = ttnn.zeros(
#         (B, 1, 1, max_len),
#         dtype=ttnn.bfloat16,
#         device=self.device,
#         layout = ttnn.TILE_LAYOUT
#     )

#     # invalid positions → NEG
#     valid = ttnn.unsqueeze(valid, 1)        # [B, 1, max_len]
#     valid = ttnn.unsqueeze(valid, 1)        # [B, 1, 1, max_len]

#     neg_tensor = ttnn.full(
#         mask.shape,
#         NEG,
#         dtype=ttnn.bfloat16,
#         device=self.device,
#         layout = ttnn.TILE_LAYOUT
#     )
#     mask = ttnn.where(valid, mask, neg_tensor)

#     # -------------------------
#     # pad sequence dim to pos_int
#     # torch.cat((mask, extra), dim=3)
#     # -------------------------
#     if pos_int > max_len:
#         extra_seq = ttnn.zeros(
#             (B, 1, 1, pos_int - max_len),
#             dtype=ttnn.bfloat16,
#             device=self.device,
#             layout = ttnn.TILE_LAYOUT
#         )
#         mask = ttnn.concat([mask, extra_seq], dim=3)

#     # -------------------------
#     # pad head/row dim to S_new (NEG)
#     # torch.cat((mask, extra), dim=2)
#     # -------------------------
#     if mask.shape[2] < S_new:
#         extra_rows = ttnn.full(
#             (B, 1, S_new - mask.shape[2], mask.shape[3]),
#             NEG,
#             dtype=ttnn.bfloat16,
#             device=self.device,
#             layout = ttnn.TILE_LAYOUT
#         )
#         mask = ttnn.concat([mask, extra_rows], dim=2)

#     # -------------------------
#     # pad col dim to S_new (NEG)
#     # torch.cat((mask, extra), dim=3)
#     # -------------------------
#     if mask.shape[3] < S_new:
#         extra_cols = ttnn.full(
#             (B, 1, mask.shape[2], S_new - mask.shape[3]),
#             NEG,
#             dtype=ttnn.bfloat16,
#             device=self.device,
#             layout = ttnn.TILE_LAYOUT
#         )
#         mask = ttnn.concat([mask, extra_cols], dim=3)

#     return mask

# def build_causal_mask(self, idx):
#     NEG = -1e4  # safer than -inf on TT
#     B = idx.shape[0]
#     S = idx.padded_shape[2]

#     # Lower triangular matrix
#     causal = torch.tril(torch.ones((S, S), dtype=torch.bool))
#     causal = causal.unsqueeze(0).unsqueeze(0)
#     causal = causal.expand(B, 1, S, S)

#     # Convert to float mask
#     mask = torch.zeros((B, 1, S, S), dtype=torch.bfloat16)
#     mask[~causal] = NEG

#     return mask


#     def generate(
#     self,
#     idx,
#     seq_lens: Optional = None,
#     max_new_tokens: int = 20,
#     temperature: float = 1.0,
#     eos_id: int = None,
#     do_sample: bool = False,  # keep greedy for now
#     top_k=None,
# ):

#     vocab_size = int(self.config.vocab_size)

#     # Convert to TT once
#     if not isinstance(idx, ttnn.Tensor):
#         idx = ttnn.from_torch(
#             idx.to(torch.uint32),
#             device=self.device,
#             dtype=ttnn.uint32,
#             layout=ttnn.TILE_LAYOUT,
#         )

#     B = idx.shape[0]
#     prompt_len = idx.shape[1]

#     # Track which sequences are finished
#     finished = torch.zeros(B, dtype=torch.bool)

#     # PREFILL PHASE: Process entire prompt and populate cache
#     tt_logits = self.forward_prefill(idx, current_pos=0)

#     # Get logits for last token of prompt
#     tt_logits = ttnn.squeeze(tt_logits, dim=1)
#     last_logits = tt_logits[:, -1, :]  # [B, vocab_size]

#     # Sample first token
#     idx_next = ttnn.argmax(last_logits, dim=-1)  # [B]
#     idx_next = ttnn.unsqueeze(idx_next, dim=1)  # [B, 1]
#     idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

#     # Concatenate generated token
#     idx = ttnn.concat([idx, idx_next], dim=1)

#     # Initialize current_pos tensor for decode
#     # current_pos_torch = torch.tensor([prompt_len + 1], dtype=torch.int32)
#     # current_pos = ttnn.from_torch(
#     #     current_pos_torch,
#     #     device=self.device,
#     #     dtype=ttnn.int32,
#     #     layout=ttnn.ROW_MAJOR_LAYOUT,
#     # )
#     current_pos = ttnn.full((1,), prompt_len + 1, dtype=ttnn.int32,device = self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
#     one = ttnn.full((1,), 1, dtype=ttnn.int32,device = self.device, layout=ttnn.ROW_MAJOR_LAYOUT)

#     # DECODE PHASE: Generate remaining tokens one at a time
#     for _ in range(max_new_tokens - 1):
#         # Forward decode with single token
#         tt_logits = self.forward_decode(idx_next, current_pos=current_pos, seq_lens=seq_lens)

#         tt_logits = ttnn.squeeze(tt_logits, dim=1)
#         last_logits = tt_logits[:, -1, :]

#         # Sample next token
#         idx_next = ttnn.argmax(last_logits, dim=-1)
#         idx_next = ttnn.unsqueeze(idx_next, dim=1)
#         idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

#         # Concatenate
#         idx = ttnn.concat([idx, idx_next], dim=1)

#         # Increment position
#         # current_pos_torch += 1
#         # current_pos = ttnn.from_torch(
#         #     current_pos_torch,
#         #     device=self.device,
#         #     dtype=ttnn.int32,
#         #     layout=ttnn.ROW_MAJOR_LAYOUT,
#         # )
#         current_pos = ttnn.add(current_pos, one)

#         # EOS handling
#         if eos_id is not None:
#             next_tok = ttnn.to_torch(idx_next).squeeze(1)
#             finished |= next_tok == eos_id
#             if finished.all():
#                 break

#     return idx


#     def sample(self, logits, temperature, top_p):
#     if temperature > 0:
#         probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
#         next_token = sample_top_p(probs, top_p)
#     else:
#         idx_next = ttnn.argmax(logits, dim=-1)
#         idx_next = ttnn.unsqueeze(idx_next, dim=1)
#         idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

#     return idx_next

# def generate_1(
#     self,
#     idx,
#     seq_lens,
#     max_new_tokens: int = 20,
#     temperature: float = 1.0,
#     eos_id: int = None,
#     do_sample: bool = False,
#     top_p=None,
# ):

#     vocab_size = int(self.config.vocab_size)

#     if not isinstance(idx, ttnn.Tensor):
#         idx = ttnn.from_torch(
#             idx.to(torch.uint32),
#             device=self.device,
#             dtype=ttnn.uint32,
#             layout=ttnn.TILE_LAYOUT,
#         )

#     B = idx.shape[0]
#     prompt_len = idx.shape[1]

#     # -------------------------
#     # DEVICE finished flag
#     # -------------------------
#     finished = ttnn.zeros(
#         (B,),
#         dtype=ttnn.int32,
#         device=self.device,
#         layout=ttnn.ROW_MAJOR_LAYOUT,
#     )

#     NEG_INF = -1e9

#     pad_logits = ttnn.full(
#         (B, vocab_size),
#         NEG_INF,
#         dtype=ttnn.bfloat16,
#         device=self.device,
#         layout=ttnn.TILE_LAYOUT,
#     )

#     # allow only PAD token
#     pad_logits = ttnn.to_torch(pad_logits)
#     pad_logits[:, eos_id] = 0
#     pad_logits = ttnn.from_torch(
#         pad_logits,
#         device=self.device,
#         dtype=ttnn.bfloat16,
#         layout=ttnn.TILE_LAYOUT,
#     )

#     # -------------------------
#     # PREFILL
#     # -------------------------
#     tt_logits = self.forward_prefill(idx, current_pos=0)
#     tt_logits = ttnn.squeeze(tt_logits, dim=1)
#     last_logits = tt_logits[:, -1, :]

#     idx_next = ttnn.argmax(last_logits, dim=-1)
#     idx_next = ttnn.unsqueeze(idx_next, dim=1)
#     idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

#     idx = ttnn.concat([idx, idx_next], dim=1)

#     # position
#     current_pos = ttnn.full((1,), prompt_len + 1, dtype=ttnn.int32,device = self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
#     one = ttnn.full((1,), 1, dtype=ttnn.int32,device = self.device, layout=ttnn.ROW_MAJOR_LAYOUT)

#     # -------------------------
#     # DECODE LOOP
#     # -------------------------
#     for _ in range(max_new_tokens - 1):

#         tt_logits = self.forward_decode(
#             idx_next,
#             current_pos=current_pos,
#             seq_lens=seq_lens,
#         )

#         tt_logits = ttnn.squeeze(tt_logits, dim=1)
#         last_logits = tt_logits[:, -1, :]   # [B, vocab]

#         # MASK LOGITS FOR FINISHED SEQS
#         if eos_id is not None:
#             finished_b = ttnn.unsqueeze(finished, dim=1)   # [B, 1]

#             finished_b = ttnn.to_layout(finished_b, ttnn.TILE_LAYOUT)
#             last_logits = ttnn.to_layout(last_logits, ttnn.TILE_LAYOUT)

#             last_logits = ttnn.where(
#                 finished_b,
#                 pad_logits,
#                 last_logits,
#             )

#         # sample
#         idx_next = ttnn.argmax(last_logits, dim=-1)
#         idx_next = ttnn.unsqueeze(idx_next, dim=1)
#         idx_next = ttnn.to_layout(idx_next, ttnn.TILE_LAYOUT)

#         idx = ttnn.concat([idx, idx_next], dim=1)

#         # UPDATE finished ON DEVICE
#         if eos_id is not None:
#             next_tok = ttnn.squeeze(idx_next, dim=1)
#             eos_tensor = ttnn.full(
#                 next_tok.shape,
#                 eos_id,
#                 dtype=ttnn.uint32,
#                 device=self.device,
#             )
#             is_eos = ttnn.eq(next_tok, eos_tensor)
#             finished = ttnn.logical_or(finished, is_eos)

#         current_pos = ttnn.add(current_pos, one)


#     return idx
