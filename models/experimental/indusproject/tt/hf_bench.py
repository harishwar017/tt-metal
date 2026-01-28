import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def benchmark_hf(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    """
    Returns:
      - ttft_s
      - decode_tps
      - overall_tps
      - total_time_s
      - generated_ids (includes prompt + new tokens)
    """

    device = next(model.parameters()).device
    model.eval()

    input_ids = input_ids.to(device)
    B = input_ids.shape[0]

    # Sync for accurate timing on GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    # =========================
    # 1) PREFILL (prompt pass)
    # =========================
    out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values

    # First next-token logits from last prompt position
    logits = out.logits[:, -1, :]  # [B, vocab]
    if do_sample:
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_after_first = time.perf_counter()
    ttft_s = t_after_first - t_start

    # Generated continuation tokens
    new_tokens = [next_token]

    # =========================
    # 2) DECODE LOOP
    # =========================
    decode_start = time.perf_counter()

    cur_token = next_token
    for _ in range(max_new_tokens - 1):
        out = model(
            input_ids=cur_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

        if do_sample:
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            cur_token = torch.multinomial(probs, num_samples=1)
        else:
            cur_token = torch.argmax(logits, dim=-1, keepdim=True)

        new_tokens.append(cur_token)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    # =========================
    # Metrics
    # =========================
    decode_time_s = t_end - decode_start
    total_time_s = t_end - t_start

    decode_tokens = max(max_new_tokens - 1, 0)
    decode_tps = decode_tokens / decode_time_s if decode_time_s > 0 else float("inf")
    overall_tps = max_new_tokens / total_time_s if total_time_s > 0 else float("inf")

    continuation = torch.cat(new_tokens, dim=1)  # [B, max_new_tokens]
    generated_ids = torch.cat([input_ids, continuation], dim=1)

    stats = {
        "ttft_s": ttft_s,
        "decode_tps": decode_tps,
        "overall_tps": overall_tps,
        "total_time_s": total_time_s,
        "decode_time_s": decode_time_s,
        "max_new_tokens": max_new_tokens,
        "batch_size": B,
    }

    return generated_ids, stats


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "nickmalhotra/ProjectIndus",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("nickmalhotra/ProjectIndus")


def format_template(user_prompt):
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    response = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return response


user_prompt = "भारत के वर्तमान प्रधानमंत्री कौन हैं?"
input_ids = format_template(user_prompt)  # your function returns CPU ids
input_ids = input_ids.to("cuda")

# warmup
for _ in range(3):
    _ = model(input_ids=input_ids, use_cache=True)

gen_ids, s = benchmark_hf(
    model,
    input_ids,
    max_new_tokens=64,
    do_sample=False,  # IMPORTANT for stable benchmark
    temperature=1.0,
)

print("\n--- HF Bench ---")
print(f"TTFT: {s['ttft_s']*1000:.2f} ms")
print(f"Decode tok/s: {s['decode_tps']:.2f}")
print(f"Overall tok/s: {s['overall_tps']:.2f}")

print(tokenizer.decode(gen_ids[0], skip_special_tokens=False))
