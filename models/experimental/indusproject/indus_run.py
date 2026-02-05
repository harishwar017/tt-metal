# Full generate test with chat template - same as HF example
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.experimental.indusproject.indusproject_utils import get_tt_cache_path, store_weights
import models.experimental.indusproject.tt.indus_model as indus_model
from pathlib import Path
import os
import torch

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


# def format_template(user_prompt):
#     messages = [
#         [
#             {"role": "user", "content": user_prompt},
#         ]
#     ]
#     response = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
#     return response


# # user_prompt = """भारत की राजधानी क्या है?"""
# # user_prompt = """भारत के वर्तमान प्रधानमंत्री कौन हैं?"""
# user_prompt = """दिल्ली किस नदी के किनारे स्थित है?"""
# # user_prompt = """भारत के पहले राष्ट्रपति कौन थे?"""
# # user_prompt = """ताजमहल कहाँ स्थित है?"""

# test_input_ids = format_template(user_prompt)
# print(f"Test input: {user_prompt}")


# output_ids = tt_model.generate_1(idx=test_input_ids, do_sample=False, max_new_tokens=12, eos_id=tokenizer.eos_token_id)
# output_ids = ttnn.to_torch(output_ids)
# text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
# print(f"\nGenerated text:\n{text}")
# print("\n✓ TT Generate function works!")


# def format_template_batch(user_prompts):
#     messages = [
#         [
#             {"role": "user", "content": prompt},
#         ]
#         for prompt in user_prompts
#     ]

#     # Apply chat template in batch
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=False,
#         padding=True,              # IMPORTANT for batching
#         return_tensors="pt"
#     )

#     return input_ids


def format_template_batch(user_prompts):
    messages = [[{"role": "user", "content": prompt}] for prompt in user_prompts]

    # 1) No padding → get true lengths
    unpadded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        padding=False,
        return_tensors=None,  # return list
    )

    seq_lens = torch.tensor([len(x) for x in unpadded], dtype=torch.int32)

    # 2) With padding → get batch tensor
    padded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, padding=True, return_tensors="pt"
    )

    max_len = padded.shape[1]
    seq_lens = max_len - seq_lens
    print(f"seq_len {seq_lens}")
    print(f"paded_shape:{padded.shape} ")

    return unpadded, padded, seq_lens


# --------- Batch of prompts ----------
user_prompts = [
    # "दिल्ली किस नदी के किनारे स्थित है??",
    "ताजमहल कहाँ स्थित है?",
    "भारत के वर्तमान प्रधानमंत्री कौन हैं?",
    "दिल्ली किस नदी के किनारे स्थित है??",
    "भारत की राजधानी क्या है?",
]

# Format batch
unpadded, test_input_ids, seq_lens = format_template_batch(user_prompts)

print("Batch shape:", test_input_ids.shape)
print("Prompts:", user_prompts)


# --------- Generate ----------
output_ids = tt_model.generate_1(
    idx=test_input_ids, max_new_tokens=12, eos_id=tokenizer.eos_token_id, seq_lens=seq_lens
)

# # Convert to torch
output_ids = ttnn.to_torch(output_ids)


# # --------- Decode ----------
# outputs = tokenizer.batch_decode(
#     output_ids,
#     skip_special_tokens=True
# )

# # # Print results
# # for i, (inp, out) in enumerate(zip(user_prompts, outputs)):
# #     # print(f"\n[{i}] Prompt: {inp}")
# #     print(f"    Output: {out}")


# --------- Decode & Print ----------
print("\n--- Model Results ---")
for i, prompt_text in enumerate(user_prompts):
    prompt_len = len(unpadded[i])
    generated_tokens = output_ids[i, prompt_len:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"Prompt {i+1}: {prompt_text}")
    print(f"Output {i+1}: {decoded_output.strip()}")
    print("-" * 30)
