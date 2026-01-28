# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModelForCausalLM
from models.experimental.indusproject.tt.indus_model import TtGPT


def _indusproject(config, device, tt_cache_path, dtype):
    return TtGPT(
        config=config,
        device=device,
        tt_cache_path=tt_cache_path,
        dtype=dtype,
    )


def indusproject_model(device, dtype) -> TtGPT:
    model_name = "nickmalhotra/ProjectIndus"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = model.config
    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/IndusProject/"
    model = _indusproject(config, device, tt_cache_path, dtype)
    return model
