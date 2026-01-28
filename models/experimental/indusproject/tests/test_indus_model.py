# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# pytest models/experimental/nanogpt/tests/test_indus_model.py
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer
from models.experimental.indusproject.indusproject_utils import get_tt_cache_path, store_weights
from pathlib import Path
import os

from loguru import logger
import models.experimental.indusproject.tt.indus_model as indus_model

from models.common.utility_functions import tt_to_torch_tensor, comp_allclose, comp_pcc


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "pcc, prompt",
    ((0.98, "Hello, my dog is a little"),),
)
def test_indus_model_real(device, pcc, prompt, dtype, reset_seeds):
    # Prepare input
    model_hf = AutoModelForCausalLM.from_pretrained("nickmalhotra/ProjectIndus")
    tokenizer = AutoTokenizer.from_pretrained("nickmalhotra/ProjectIndus")
    model_hf.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=False)

    pt_model = model_hf
    pt_out = pt_model.forward(inputs.input_ids)

    config_1 = model_hf.config
    print(f"indus gpt config: {config_1}")

    base_address = ""
    model_version = "indusproject"
    tt_cache_path = get_tt_cache_path(model_version)

    if (
        tt_cache_path == (str(Path(f"models/experimental/indusproject/datasets/{model_version}")) + "/")
        and len(os.listdir(f"models/experimental/indusproject/datasets/{model_version}")) < 320
    ):
        store_weights(model_version=model_version, file_name=tt_cache_path, dtype=dtype, base_address=base_address)

    tt_model = indus_model.TtGPT(config_1, device, tt_cache_path, dtype)

    tt_input_ids = ttnn.from_torch(inputs.input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_out = tt_model.forward(tt_input_ids)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze()

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out[0], tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("indus_model_real: Passed!")
    else:
        logger.warning("indus_model_real: Failed!")

    assert does_pass
