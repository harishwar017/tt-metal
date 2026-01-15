# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import transformers
import ttnn
from loguru import logger

from ...encoders.transformer import MISTRAL3_CONVERSION, RopeConfig, TransformerEncoder
from ...layers.module import Module
from ...utils import cache, tensor
from .system_messages import SYSTEM_MESSAGE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from ...parallel.config import EncoderParallelConfig
    from ...parallel.manager import CCLManager


class PromptEncoder:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch_encoder: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        self._tokenizer = transformers.LlamaTokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer")
        assert isinstance(self._tokenizer, transformers.LlamaTokenizerFast)

        if use_torch_encoder:
            self._encoder = _load_torch_encoder(checkpoint_name)
            return

        self._encoder = TransformerEncoder(
            vocab_size=131072,
            head_size=128,
            embed_size=5120,
            ff_size=32768,
            num_layers=40,
            num_heads=32,
            num_kv_heads=8,
            norm_eps=1e-05,
            attn_qkv_bias=False,
            attn_out_bias=False,
            rope_config=RopeConfig(theta=1000000000),
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        def get_torch_state_dict() -> dict[str, torch.Tensor]:
            return MISTRAL3_CONVERSION.convert(_load_torch_encoder(checkpoint_name).state_dict())

        cache.load_model(
            self._encoder,
            model_name="flux2",
            subfolder="text_encoder",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            dtype="bf16",
            get_torch_state_dict=get_torch_state_dict,
        )

    def encode(
        self, prompts: Sequence[str], *, num_images_per_prompt: int, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            sequence_length=sequence_length,
            output_from_layers=[10, 20, 30],
            mesh_device=self._device,
        )


def _load_torch_encoder(checkpoint_name: str) -> transformers.Mistral3ForConditionalGeneration:
    return transformers.Mistral3ForConditionalGeneration.from_pretrained(checkpoint_name, subfolder="text_encoder")


def _get_prompt_embeds(
    *,
    text_encoder: Module | torch.nn.Module,
    prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    num_images_per_prompt: int,
    output_from_layers: Sequence[int],
    mesh_device: ttnn.MeshDevice | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    conversation = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in prompts
    ]

    tokenizer_out = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_dict=True,
    )
    tokens = tokenizer_out.input_ids
    attention_mask = tokenizer_out.attention_mask

    untruncated_out = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        padding="longest",
        return_dict=True,
    )
    untruncated_tokens = untruncated_out.input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("input text was truncated")

    if isinstance(text_encoder, Module):
        assert mesh_device is not None

        tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
        tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device)

        tt_hidden_states = text_encoder.forward(
            tt_tokens,
            mask=tt_attention_mask,
            skip_final_linear=True,
            output_hidden_states=True,
        )
        tt_prompt_embeds = ttnn.concat([tt_hidden_states[k] for k in output_from_layers], dim=-1)

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)

        with torch.no_grad():
            output = text_encoder.forward(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        prompt_embeds = torch.concat([output.hidden_states[k] for k in output_from_layers], dim=-1).to("cpu")

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, attention_mask
