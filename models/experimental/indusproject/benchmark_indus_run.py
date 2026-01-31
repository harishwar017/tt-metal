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

test_input_ids = format_template(user_prompt)
print(f"Test input: {user_prompt}")


print("Benchmarking indus model...")

timing = tt_model.benchmark_generate(idx=test_input_ids, max_new_tokens=32, eos_id=tokenizer.eos_token_id)

output_ids = ttnn.to_torch(timing["idx"])
text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print("✓ Benchmark complete!")
print(f"Generated output: {text}")
print(f"TT Generate ttft: {timing['ttft_avg']} seconds")
print(f"TT Generate tps: {timing['tps_runs'][-1]} seconds")
