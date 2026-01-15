# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.pipelines.flux2.pipeline_flux2
import pytest
import torch
import transformers
import transformers.models.mistral.modeling_mistral
import ttnn
from loguru import logger

# from ....encoders.mistral.encoder_pair import MistralTokenizerEncoderPair
from ....encoders.mistral.model_mistral import (
    MistralAttention,
    MistralContext,
    MistralTextEncoder,
    create_rope_tensors,
    prepare_attention_bias,
)
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "batch_size", "skip_layers"),
    [
        # pytest.param((1, 1), 2, 4, id="1x1"),
        pytest.param((1, 2), 10, 0, id="1x2"),
        pytest.param((1, 8), 10, 0, id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        pytest.param(False, id="unmasked"),
    ],
)
def test_mistral_text_encoder(*, mesh_device: ttnn.MeshDevice, batch_size: int, skip_layers: int, masked: bool) -> None:
    torch.manual_seed(0)

    sequence_length = 512
    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = (
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        )
        if tp_axis is not None
        else None
    )

    torch_model = transformers.Mistral3ForConditionalGeneration.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="text_encoder"
    )
    torch_text_model = torch_model.model.language_model
    config = torch_text_model.config

    mid = len(torch_text_model.layers) // 2
    del torch_text_model.layers[mid - skip_layers // 2 : mid - (-skip_layers // 2)]

    model = MistralTextEncoder(
        vocab_size=config.vocab_size,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        num_hidden_layers=config.num_hidden_layers - skip_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        # mrope_section=config.rope_scaling["mrope_section"],
        mrope_section=[],
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    model.load_torch_state_dict(torch_text_model.state_dict())

    tokens = torch.randint(0, config.vocab_size, [batch_size, sequence_length])
    m = torch.randint(0, sequence_length + 1, [batch_size])
    attention_mask = torch.arange(sequence_length) < m.unsqueeze(1) if masked else None
    cos, sin = model.create_rope_tensors(batch_size, sequence_length, attention_mask)

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device) if attention_mask is not None else None
    tt_pos_embeds_cos = tensor.from_torch(cos, device=mesh_device)
    tt_pos_embeds_sin = tensor.from_torch(sin, device=mesh_device)

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        attention_mask=tt_attention_mask,
        pos_embeds=(tt_pos_embeds_cos, tt_pos_embeds_sin),
    )
    tt_prompt_embeds = tt_hidden_states[-1]
    tt_prompt_embeds_torch = tensor.to_torch(tt_prompt_embeds)

    logger.info("running torch model...")
    with torch.no_grad():
        out = torch_model.forward(
            tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = out.hidden_states[-1]

    assert len(out.hidden_states) == len(tt_hidden_states)

    if masked:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.952, relative_rmse=0.31)
    else:
        assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.991, relative_rmse=0.14)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 2), id="1x2"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompts",
    [
        [
            "",
            "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot",
        ],
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_mistral_encoder_pair(*, mesh_device: ttnn.MeshDevice, prompts: list[str]) -> None:
    # There is a bug in the HF implementation where the prompt_embeds_mask is incorrectly repeated
    # if num_images_per_prompt != 1.
    # https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L262
    # is
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    # but should be
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
    num_images_per_prompt = 1

    checkpoint = "black-forest-labs/FLUX.2-dev"

    torch_pipeline = diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.from_pretrained(checkpoint)

    template = torch_pipeline.prompt_template_encode
    start_idx = torch_pipeline.prompt_template_encode_start_idx
    sequence_length = 512

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_encoder_pair = MistralTokenizerEncoderPair(
        checkpoint,
        tokenizer_subfolder="tokenizer",
        encoder_subfolder="text_encoder",
        use_torch=False,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    logger.info("running torch model...")
    with torch.no_grad():
        embeds, mask = torch_pipeline.encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=sequence_length,
        )
        embeds = torch.nn.functional.pad(embeds, [0, 0, 0, sequence_length - embeds.shape[1]], value=0)
        mask = torch.nn.functional.pad(mask, [0, sequence_length - mask.shape[1]], value=0)

    logger.info("running TT model...")
    formatted_prompts = [template.format(e) for e in prompts]
    tt_embeds, tt_mask = tt_encoder_pair.encode(
        formatted_prompts,
        num_images_per_prompt=num_images_per_prompt,
        sequence_length=sequence_length + start_idx,
    )
    tt_embeds = tt_embeds[:, start_idx:]
    tt_mask = tt_mask[:, start_idx:]
    tt_embeds *= tt_mask.unsqueeze(-1)

    assert torch.allclose(mask, tt_mask)
    assert_quality(embeds, tt_embeds, pcc=0.983, relative_rmse=0.19)
