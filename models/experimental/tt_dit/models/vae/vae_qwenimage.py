# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import ttnn

from ...blocks.vae import VaeContext, VaeConv2d, VaeMidBlock, VaeNormDescRms, VaeRmsNorm, VaeUpBlock
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class QwenImageVaeDecoder(Module):
    """Qwen-Image VAE decoder without support for temporal dimension."""

    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        parallel_config: VAEParallelConfig | None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = VaeContext(
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            device=device,
            ccl_manager=ccl_manager,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        dims = [base_dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        eps = 1e-12

        self.post_quant_conv = Linear(z_dim, z_dim, mesh_device=device)
        self.conv_in = VaeConv2d(z_dim, dims[0], kernel_size=3, padding=1, ctx=ctx)
        self.mid_block = VaeMidBlock(num_channels=dims[0], norm=VaeNormDescRms(eps=eps), ctx=ctx)

        self.up_blocks = ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            up_block = VaeUpBlock(
                in_channels=in_dim // 2 if i > 0 else in_dim,
                out_channels=out_dim,
                upsampler_out_channels=out_dim // 2,
                num_layers=num_res_blocks + 1,
                upsample=i != len(dim_mult) - 1,
                norm=VaeNormDescRms(eps=eps),
                ctx=ctx,
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = VaeRmsNorm(out_dim, eps=eps, ctx=ctx)
        self.conv_out = VaeConv2d(out_dim, 3, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx)

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        _convert_state_dict(state)

        # remove encoder state
        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].squeeze(2, 3)

        rename_substate(state, "decoder", "")

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = self.post_quant_conv.forward(z)
        z = self.conv_in.forward(z)

        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)
        z = ttnn.silu(z)

        if self._ccl_manager is not None:
            z = self._ccl_manager.all_gather(z, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        z = self.conv_out.forward(z)

        return ttnn.clamp(z, min=-1.0, max=1.0)


def _convert_state_dict(state: dict[str, torch.Tensor]) -> None:
    """Convert state dict to diffusers format and remove time dimension."""
    rename = {
        "decoder.norm_out.gamma": "decoder.conv_norm_out.gamma",
        "decoder.up_blocks.0.upsamplers.0.resample.1.weight": "decoder.up_blocks.0.upsampler.conv.weight",
        "decoder.up_blocks.1.upsamplers.0.resample.1.weight": "decoder.up_blocks.1.upsampler.conv.weight",
        "decoder.up_blocks.2.upsamplers.0.resample.1.weight": "decoder.up_blocks.2.upsampler.conv.weight",
        "decoder.up_blocks.0.upsamplers.0.resample.1.bias": "decoder.up_blocks.0.upsampler.conv.bias",
        "decoder.up_blocks.1.upsamplers.0.resample.1.bias": "decoder.up_blocks.1.upsampler.conv.bias",
        "decoder.up_blocks.2.upsamplers.0.resample.1.bias": "decoder.up_blocks.2.upsampler.conv.bias",
        "decoder.mid_block.attentions.0.proj.weight": "decoder.mid_block.attentions.0.to_out.0.weight",
        "decoder.mid_block.attentions.0.proj.bias": "decoder.mid_block.attentions.0.to_out.0.bias",
    }

    for src, dst in rename.items():
        state[dst] = state.pop(src)

    remove = [
        "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
        "decoder.up_blocks.1.upsamplers.0.time_conv.weight",
        "decoder.up_blocks.0.upsamplers.0.time_conv.bias",
        "decoder.up_blocks.1.upsamplers.0.time_conv.bias",
    ]

    for key in remove:
        del state[key]

    conv3d = [
        "decoder.conv_in.weight",
        "decoder.conv_out.weight",
        "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.mid_block.resnets.1.conv2.weight",
        "decoder.up_blocks.0.resnets.0.conv1.weight",
        "decoder.up_blocks.0.resnets.0.conv2.weight",
        "decoder.up_blocks.0.resnets.1.conv1.weight",
        "decoder.up_blocks.0.resnets.1.conv2.weight",
        "decoder.up_blocks.0.resnets.2.conv1.weight",
        "decoder.up_blocks.0.resnets.2.conv2.weight",
        "decoder.up_blocks.1.resnets.0.conv1.weight",
        "decoder.up_blocks.1.resnets.0.conv2.weight",
        "decoder.up_blocks.1.resnets.1.conv1.weight",
        "decoder.up_blocks.1.resnets.1.conv2.weight",
        "decoder.up_blocks.1.resnets.2.conv1.weight",
        "decoder.up_blocks.1.resnets.2.conv2.weight",
        "decoder.up_blocks.2.resnets.0.conv1.weight",
        "decoder.up_blocks.2.resnets.0.conv2.weight",
        "decoder.up_blocks.2.resnets.1.conv1.weight",
        "decoder.up_blocks.2.resnets.1.conv2.weight",
        "decoder.up_blocks.2.resnets.2.conv1.weight",
        "decoder.up_blocks.2.resnets.2.conv2.weight",
        "decoder.up_blocks.3.resnets.0.conv1.weight",
        "decoder.up_blocks.3.resnets.0.conv2.weight",
        "decoder.up_blocks.3.resnets.1.conv1.weight",
        "decoder.up_blocks.3.resnets.1.conv2.weight",
        "decoder.up_blocks.3.resnets.2.conv1.weight",
        "decoder.up_blocks.3.resnets.2.conv2.weight",
        "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
        "post_quant_conv.weight",
    ]

    for key in conv3d:
        state[key] = state[key][:, :, -1, :, :]

    (
        state["decoder.mid_block.attentions.0.to_q.weight"],
        state["decoder.mid_block.attentions.0.to_k.weight"],
        state["decoder.mid_block.attentions.0.to_v.weight"],
    ) = (
        state.pop("decoder.mid_block.attentions.0.to_qkv.weight").squeeze(2, 3).chunk(3)
    )

    (
        state["decoder.mid_block.attentions.0.to_q.bias"],
        state["decoder.mid_block.attentions.0.to_k.bias"],
        state["decoder.mid_block.attentions.0.to_v.bias"],
    ) = state.pop("decoder.mid_block.attentions.0.to_qkv.bias").chunk(3)

    state["decoder.mid_block.attentions.0.to_out.0.weight"] = state[
        "decoder.mid_block.attentions.0.to_out.0.weight"
    ].squeeze(2, 3)
