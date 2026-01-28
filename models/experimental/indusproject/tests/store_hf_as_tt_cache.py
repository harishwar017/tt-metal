# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import torch
import ttnn
from transformers import AutoModelForCausalLM


def maybe_transpose_gpt2_conv1d(key: str, t: torch.Tensor) -> torch.Tensor:
    """
    HF GPT2 uses Conv1D: weights stored transposed relative to Linear.
    TT code (and most nanoGPT code) expects Linear-style.
    """
    if key.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
        return t.t().contiguous()
    return t


def dump_tt_tensor(t: torch.Tensor, out_path: str, *, dtype, layout=ttnn.TILE_LAYOUT):
    """
    Convert torch tensor to TT tensor and dump as .tensorbin.
    This does NOT need an explicit device argument for dump/load in your loader.
    """
    # Ensure CPU torch tensor
    t = t.detach().contiguous().cpu()

    # Convert torch -> TT
    tt = ttnn.from_torch(t, dtype=dtype)
    if layout is not None:
        tt = ttnn.to_layout(tt, layout)

    # dump
    ttnn.dump_tensor(out_path, tt)


def store_hf_model_as_tt_cache(repo_id: str, out_dir: str, dtype=ttnn.bfloat16):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.eval()
    sd = model.state_dict()

    # ----------------------------
    # Save embeddings as torch .pt
    # ----------------------------
    torch.save(sd["transformer.wte.weight"].cpu(), out_dir / "transformer.wte.weight.pt")
    torch.save(sd["transformer.wpe.weight"].cpu(), out_dir / "transformer.wpe.weight.pt")

    # ----------------------------
    # Final LN
    # ----------------------------
    dump_tt_tensor(
        sd["transformer.ln_f.bias"],
        str(out_dir / f"transformer.ln_f.bias{dtype}.tensorbin"),
        dtype=dtype,
    )
    dump_tt_tensor(
        sd["transformer.ln_f.weight"],
        str(out_dir / f"transformer.ln_f.weight{dtype}.tensorbin"),
        dtype=dtype,
    )

    # ----------------------------
    # LM head
    # ----------------------------
    lm_w = sd.get("lm_head.weight", sd["transformer.wte.weight"])
    dump_tt_tensor(
        lm_w,
        str(out_dir / f"lm_head.weight{dtype}.tensorbin"),
        dtype=dtype,
    )

    # ----------------------------
    # All block weights (TtBlock loads per-key)
    # ----------------------------
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if not k.startswith("transformer.h."):
            continue

        v2 = maybe_transpose_gpt2_conv1d(k, v)

        # Your loader expects: <key><dtype>.tensorbin
        out_path = out_dir / f"{k}{dtype}.tensorbin"
        dump_tt_tensor(v2, str(out_path), dtype=dtype)

    print(f"[OK] Wrote TT cache for {repo_id} to: {out_dir}")


if __name__ == "__main__":
    repo_id = "/tt-metal/models/experimental/nanogpt/datasets/nickmalhotra--ProjectIndus/snapshots/4f2e365d7a34c63c1e3be66346d563e0a5e31781/"
    out_dir = "/tt-metal/models/experimental/nanogpt/datasets/projectindus/"
    store_hf_model_as_tt_cache(repo_id, out_dir, dtype=ttnn.bfloat16)
