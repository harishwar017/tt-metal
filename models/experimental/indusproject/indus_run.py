# Full generate test with chat template - same as HF example
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.experimental.indusproject.indusproject_utils import get_tt_cache_path, store_weights
import models.experimental.indusproject.tt.indus_model as indus_model
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


# device = ttnn.device.open_device(device_id=0)
mesh_shape = ttnn.MeshShape(1, 1)

# 2. actually OPEN the device using that shape
# This creates the ttnn._ttnn.multi_device.MeshDevice object
device = ttnn.open_mesh_device(mesh_shape)

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


user_prompt = """भारत के वर्तमान समय में देश की सरकार का नेतृत्व करने वाले, संसद में बहुमत रखने वाली पार्टी के नेता और प्रशासनिक फैसलों में मुख्य भूमिका निभाने वाले प्रधानमंत्री कौन हैं?"""

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


# timing = tt_model.benchmark_generate(
#     idx=test_input_ids,
#     # do_sample=False,
#     max_new_tokens=32,
#     # temperature=1.0,
#     eos_id=tokenizer.eos_token_id
# )


timings = tt_model.generate_kv(
    idx=test_input_ids,
    max_new_tokens=10,
    eos_id=tokenizer.eos_token_id,
)
# print(f"ttft avg: {timings['ttft_avg']}")
# print(f"tps avg: {timings['tps_avg']}")
output_ids = ttnn.to_torch(timings)
text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
print(f"text: {text}")
# print(f"Output shape: {output_ids.shape}")
# output_ids = ttnn.to_torch(output_ids)
# text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)

# print("\n✓ TT Generate function works!")
# print("done")

# texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

# for t in texts:
#     print(t)


# print("TTFT:", stats["ttft_avg"])
# print("TPS :", stats["tps_avg"])
