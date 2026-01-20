# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn


def Linear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.padded_shape == [
        1,
        1,
        out_features,
        in_features,
    ], "weight does not have the expected shape"

    if bias is not None:
        assert bias.padded_shape[-1] == out_features, "bias does not have the expected shape"

    # weight = weight
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    bias = bias
    weight_T = ttnn.transpose(weight, -2, -1)

    def linear_(activation):
        nonlocal bias
        assert activation.padded_shape[-1] == in_features, "activation tensor do not have the expected shape"
        # if bias is not None and bias.get_layout() != ttnn.TILE_LAYOUT:
        if bias is not None:
            bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        # return ttnn.linear(activation, weight_T, bias=bias, memory_config=output_mem_config)
        # Ensure activation is in TILE layout for linear operation
        activation = ttnn.to_layout(activation, ttnn.TILE_LAYOUT)
        output = ttnn.linear(activation, weight_T, bias=bias, memory_config=output_mem_config)
        return output

    return linear_
