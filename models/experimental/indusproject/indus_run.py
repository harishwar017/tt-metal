# Full generate test with chat template - same as HF example
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.experimental.indusproject.indusproject_utils import get_tt_cache_path, store_weights
import models.experimental.nanogpt.tt.indus_model as indus_model
from pathlib import Path
import os

# Load models
model = AutoModelForCausalLM.from_pretrained("nickmalhotra/ProjectIndus")
tokenizer = AutoTokenizer.from_pretrained("nickmalhotra/ProjectIndus")
model.eval()

config = model.config
model_version = "indusproject"
base_address = ""
dtype = ttnn.bfloat16

tt_cache_path = get_tt_cache_path(model_version)

if (
    tt_cache_path == (str(Path(f"models/experimental/indusproject/datasets/{model_version}")) + "/")
    and len(os.listdir(f"models/experimental/indusproject/datasets/{model_version}")) < 320
):
    store_weights(model_version=model_version, file_name=tt_cache_path, dtype=dtype, base_address=base_address)


device = ttnn.device.open_device(device_id=0)

tt_model = indus_model.TtGPT(config, device, tt_cache_path, dtype)
print("✓ TT Model loaded!")

# Test generate with chat template
print("Testing TT generate function with chat template...")


def format_template(user_prompt):
    messages = [
        [
            {"role": "user", "content": user_prompt},
        ]
    ]
    response = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return response


user_prompt = """भारत की राजधानी क्या है?"""

# format_template already returns tokenized input_ids
test_input_ids = format_template(user_prompt)
print(f"Test input shape: {test_input_ids.shape}")
# def format_chat_batch(conversations):
#     """
#     conversations: List[List[Dict]]
#     """

#     return tokenizer.apply_chat_template(
#         conversations,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         padding=True,
#     )

# with open("models/experimental/indusproject/batch32_convs.json", "r", encoding="utf-8") as f:
#     convs = json.load(f)
# test_input_ids = format_chat_batch(convs)
# print("Shape:", test_input_ids.shape)


# eos_id = tokenizer.eos_token_id

# stats = tt_model.benchmark_generate(
#     idx=test_input_ids,
#     max_new_tokens=32,
#     eos_id=tokenizer.eos_token_id,
#     runs=5,
# )

timing = tt_model.generate_timed(
    idx=test_input_ids,
    do_sample=False,
    max_new_tokens=32,
    temperature=1.0,
    # eos_id=tokenizer.eos_token_id
)

print("done")
# output_ids = tt_model.generate_1(
#     idx=test_input_ids,
#     do_sample=False,
#     # max_new_tokens=10,
#     temperature=1.0,
#     eos_id=eos_id
# )
# print(f"Output shape: {output_ids.shape}")
# output_ids = ttnn.to_torch(output_ids)
# text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
# print(f"\nGenerated text:\n{text}")
# print("\n✓ TT Generate function works!")

# texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

# for t in texts:
#     print(t)


# print("TTFT:", stats["ttft_avg"])
# print("TPS :", stats["tps_avg"])
