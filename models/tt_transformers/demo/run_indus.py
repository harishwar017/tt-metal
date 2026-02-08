#!/usr/bin/env python3

import argparse
import os

import torch

import ttnn
from models.tt_transformers.tt.common import create_tt_model, preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision

os.environ["HF_MODEL"] = "nickmalhotra/ProjectIndus"

# ---------------------------------------------------------
# Args
# ---------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser("Run LLaMA 3.1 8B on Tenstorrent")

    parser.add_argument("--prompt", type=str, default="भारत के वर्तमान प्रधानमंत्री कौन हैं?", help="Input prompt")

    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")

    parser.add_argument("--max_seq_len", type=int, default=8192, help="Max context length")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    args = parse_args()

    # -----------------------------------------------------
    # Device setup (Single TT Card)
    # -----------------------------------------------------

    device_ids = ttnn.get_device_ids()
    assert len(device_ids) >= 1, "No TT devices found"

    # mesh_device = ttnn.open_device(device_id=device_ids[0])
    mesh_device = ttnn.open_device(
        device_id=device_ids[0],
        trace_region_size=50_000_000,  # 50MB (same as TT tests)
    )

    num_devices = 1
    data_parallel = 1
    batch_size = 1
    global_batch = 1

    print("Using device:", device_ids[0])

    # -----------------------------------------------------
    # Model config
    # -----------------------------------------------------

    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    instruct = True

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    submeshes = create_submeshes(mesh_device, data_parallel)

    model_args_list = []
    models = []
    kv_caches = []

    state_dict = None

    for submesh in submeshes:
        model_args, model, kv_cache, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=args.max_seq_len,
            paged_attention_config=None,
            dtype=ttnn.bfloat16,
            state_dict=state_dict,
            num_layers=None,
        )

        model_args_list.append(model_args)
        models.append(model)
        kv_caches.append(kv_cache)

    tokenizer = model_args_list[0].tokenizer
    processor = model_args_list[0].processor

    generator = Generator(
        models,
        model_args_list,
        mesh_device,
        tokenizer=tokenizer,
        processor=processor,
    )

    # -----------------------------------------------------
    # Preprocess prompt
    # -----------------------------------------------------

    prompts = [args.prompt]

    (
        input_tokens,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        prompts,
        tokenizer,
        model_args_list,
        instruct,
        args.max_tokens,
        max_prefill_len=args.max_seq_len,
    )

    input_tokens = torch.stack(input_tokens).view(1, -1)

    # -----------------------------------------------------
    # Sampling
    # -----------------------------------------------------

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # -----------------------------------------------------
    # Prefill
    # -----------------------------------------------------

    print("Running prefill...")

    logits = generator.prefill_forward_text(
        input_tokens,
        kv_cache=kv_caches,
        prompt_lens=decoding_pos,
    )

    # next_tok = torch.argmax(logits, dim=-1)
    _, next_tok = sample_host(
        logits,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # -----------------------------------------------------
    # Decode loop
    # -----------------------------------------------------

    print("Generating...")

    # outputs = encoded_prompts[0][:prefill_lens[0]]
    outputs = []
    outputs.append(int(next_tok[0].item()))
    print("First generated token:", tokenizer.decode(outputs))

    current_pos = torch.tensor([decoding_pos[0]])

    out_tok = next_tok

    for i in range(args.max_tokens):
        out_tok = generator.decode_forward_text(
            out_tok,
            current_pos,
            kv_cache=kv_caches,
            sampling_params=sampling_params,
        )

        # print("Logits shape:", logits.shape)
        # print(f"Generated token {i+1}: {tokenizer.decode([int(out_tok[0].item())])}")

        if out_tok.dim() == 1:
            out_tok = out_tok.unsqueeze(-1)

        tok = int(out_tok[0].item())

        if tok in tokenizer.stop_tokens or tok == tokenizer.eos_token_id:
            break

        outputs.append(tok)

        current_pos += 1

        # -H will try to get the below running, buit not absolutely necessary for now.
        # def sample(logits):
        #     if args.temperature > 0:
        #         probs = torch.softmax(logits[:, -1] / args.temperature, dim=-1)
        #         next_token = sample_top_p(probs, args.top_p)
        #     else:
        #         next_token = torch.argmax(logits[:, -1], dim=-1)
        #     next_token = next_token.reshape(-1)
        #     decoder = tokenizer or processor
        #     return next_token, decoder.decode(next_token.tolist())

        # next_token, text = sample(logits)

    # -----------------------------------------------------
    # Print output
    # -----------------------------------------------------

    text = tokenizer.decode(outputs, skip_special_tokens=True)

    # prompt_with_tags = tokenizer.decode(
    #     model_args_list[0].encode_prompt(args.prompt, instruct=instruct)
    # )

    # final = text.replace(prompt_with_tags, "", 1)

    print("\n================ OUTPUT ================\n")
    print(text.strip())
    print("\n========================================\n")


if __name__ == "__main__":
    main()
