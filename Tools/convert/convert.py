#!/usr/bin/env python3
"""V1 converter: ACE-Step/Ace-Step1.5 @ 19671f4 → MLX-layout bundles for mlx-ace-step-swift (AB-T-0352, E25).

Four bundles, one per upstream folder, every key otherwise verbatim (isomorphic port):
  ACE-Step-1.5-turbo-bf16/      model.safetensors (bf16, as shipped) — `decoder.proj_in.1.*` → `decoder.proj_in.*` with the
                                Conv1d weight (O, I, K) → (O, K, I); `decoder.proj_out.1.*` → `decoder.proj_out.*` with the
                                ConvTranspose1d weight (I, O, K) → (O, K, I) — exactly upstream's dit_convert.py; bf16 transposes
                                are exact. config.json + silence_latent.safetensors ({"silence_latent": [15000, 64] fp32}).
  ACE-Step-1.5-planner-1.7B-bf16/  mlx-lm layout (`model.` prefix on every key, tied embedding, no lm_head) + config.json +
                                the tokenizer files (vocab.json, merges.txt, tokenizer.json, added_tokens.json, ...).
  Qwen3-Embedding-0.6B-bf16/    the same treatment (it is a bare Qwen3Model too).
  ACE-Step-1.5-vae-fp32/        encoder + decoder, weight_norm folded in fp32 numpy exactly as upstream's vae_convert.py does
                                (g · v / (‖v‖ + 1e-9), norm over all axes but the first), Conv1d (O, I, K) → (O, K, I),
                                ConvTranspose1d `conv_t1` (I, O, K) → (O, K, I), Snake alpha/beta [1, C, 1] → [C], biases as is.
Every bundle carries LICENSE (the GitHub repo's MIT text), NOTICE (provenance + the two manifest footnotes) and the HF README.

    .venv/bin/python -I convert.py --weights <Ace-Step1.5 snapshot> --github <ACE-Step-1.5 clone> --out <dir>
                                   [--parts turbo,planner,text,vae]
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from safetensors.numpy import save_file as save_np

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer.json", "added_tokens.json", "special_tokens_map.json",
                   "tokenizer_config.json", "chat_template.jinja")
UPSTREAM = {"weights": "ACE-Step/Ace-Step1.5@19671f4", "code": "ace-step/ACE-Step-1.5@ca1e85f"}


def load_all(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def licences(weights: Path, github: Path, out: Path, part: str) -> None:
    shutil.copyfile(github / "LICENSE", out / "LICENSE")
    if (weights / "README.md").exists():
        shutil.copyfile(weights / "README.md", out / "README.md")
    (out / "NOTICE").write_text(
        f"ACE-Step 1.5 — {part} — by ACE Studio and StepFun (https://huggingface.co/ACE-Step/Ace-Step1.5, {UPSTREAM['weights']}).\n"
        "This bundle is an MLX-layout conversion of the released checkpoint (key prefix / conv layout / weight_norm folding);\n"
        "the model is otherwise unmodified. Code licence: MIT (ace-step/ACE-Step-1.5; LICENSE is that repository's text).\n"
        "Weights: the Hugging Face repository has no LICENSE file; its card tag and README say MIT, and the GitHub LICENSE\n"
        "(Copyright (c) 2026 ACEStep) is the controlling text. Footnote 1: the HF repo carries no LICENSE file — the grant\n"
        "is the card tag + README + the GitHub LICENSE. Footnote 2: the Oobleck VAE (ACE-Step/ace-step-v1.5-1d-vae-\n"
        "stable-audio-format, MIT) is architecturally Stable Audio Open's VAE retargeted to 48 kHz; whether it was\n"
        "initialised from Stability's community-licensed weights is stated nowhere — ACE-Step asserts MIT and ships it.\n"
        "Bases: Qwen3-Embedding-0.6B and the planner's Qwen3 base are Apache-2.0 (Qwen); the fine-tune is ACE-Step's.\n")


def meta(extra: dict) -> dict:
    return {"source_weights": UPSTREAM["weights"], "upstream_code": UPSTREAM["code"], **extra}


def convert_turbo(weights: Path, github: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src = weights / "acestep-v15-turbo"
    sd = load_all(src / "model.safetensors")
    tensors: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("decoder.proj_in.1."):
            nk = k.replace("decoder.proj_in.1.", "decoder.proj_in.")
            if nk.endswith(".weight"):
                v = v.permute(0, 2, 1)          # Conv1d (O, I, K) → (O, K, I)
        elif k.startswith("decoder.proj_out.1."):
            nk = k.replace("decoder.proj_out.1.", "decoder.proj_out.")
            if nk.endswith(".weight"):
                v = v.permute(1, 2, 0)          # ConvTranspose1d (I, O, K) → (O, K, I)
        else:
            nk = k
        assert v.dtype == torch.bfloat16, (k, v.dtype)
        tensors[nk] = v.contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "mlx"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["mlx_ace_step"] = meta({"part": "acestep-v15-turbo", "dtype": "bfloat16", "conv_layout": "OKI",
                                "renamed": {"decoder.proj_in.1": "decoder.proj_in", "decoder.proj_out.1": "decoder.proj_out"}})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    sil = torch.load(src / "silence_latent.pt", map_location="cpu", weights_only=True)   # [1, 64, 15000] fp32
    assert sil.shape[0] == 1 and sil.shape[1] == 64 and sil.dtype == torch.float32, sil.shape
    save_np({"silence_latent": sil[0].transpose(0, 1).contiguous().numpy()}, str(out / "silence_latent.safetensors"),
            metadata={"format": "mlx"})
    licences(weights, github, out, "acestep-v15-turbo (DiT + condition encoder + FSQ tokenizer/detokenizer)")
    print(f"turbo → {out}: {len(tensors)} tensors bf16; silence latent {tuple(sil[0].T.shape)}")


def convert_qwen3(weights: Path, github: Path, out: Path, folder: str, part: str, src: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src = src or weights / folder
    sd = load_all(src / "model.safetensors")
    tensors = {}
    # the 1.7B planner and the embedding model are bare `Qwen3Model` checkpoints (no prefix); the 0.6B planner ships as a
    # `Qwen3ForCausalLM` (keys already `model.*`, tied head so no `lm_head`) — both land in mlx-lm's `model.*` layout.
    prefixed = all(k.startswith("model.") for k in sd)
    assert prefixed or not any(k.startswith("model.") for k in sd), "mixed key prefixes"
    for k, v in sd.items():
        assert v.dtype == torch.bfloat16, (k, v.dtype)
        assert not k.startswith("lm_head."), k
        tensors[k if prefixed else "model." + k] = v.contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "mlx"})
    cfg = json.loads((src / "config.json").read_text())
    assert cfg.get("tie_word_embeddings", False) is True, "the planner / embedding checkpoints tie their embeddings"
    cfg["mlx_ace_step"] = meta({"part": folder, "dtype": "bfloat16", "layout": "mlx-lm (model.* keys, tied embedding)"})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    licences(weights, github, out, part)
    print(f"{folder} → {out}: {len(tensors)} tensors bf16")


def quantize_planner(bf16_bundle: Path, out: Path, group_size: int = 64, bits: int = 8) -> None:
    """ACE-Step-1.5-planner-1.7B-8bit (V4b): the bf16 planner bundle with the layers' Linears quantised — affine, group 64, 8 bits,
    `mx.quantize` (the same core op mlx-swift's `QuantizedLinear.quantize` calls, so the tensors are bit-equal to what the package's
    quantise-at-load path produced for its `.int8` tier); the tied embedding / head and the norms stay bf16, exactly as
    `Qwen3LM.quantizeLayers` keeps them. mlx-lm's bundle contract: `weight` (uint32 packed) + `scales` + `biases` per Linear and
    `"quantization": {"group_size", "bits", "mode"}` in config.json (also mirrored as `quantization_config`)."""
    import mlx.core as mx
    out.mkdir(parents=True, exist_ok=True)
    arrays = mx.load(str(bf16_bundle / "model.safetensors"))
    tensors, n_q = {}, 0
    for k, v in arrays.items():
        # the layers' Linears: 2-D `.weight` tensors under `model.layers.` (norm weights are 1-D; the embedding is not a layer)
        if k.startswith("model.layers.") and k.endswith(".weight") and v.ndim == 2:
            assert v.dtype == mx.bfloat16 and v.shape[-1] % group_size == 0, (k, v.dtype, v.shape)
            wq, scales, biases = mx.quantize(v, group_size=group_size, bits=bits)
            base = k[: -len(".weight")]
            tensors[base + ".weight"] = wq; tensors[base + ".scales"] = scales; tensors[base + ".biases"] = biases
            n_q += 1
        else:
            tensors[k] = v
    mx.eval(*tensors.values())                                   # materialise before saving (lazy arrays save as zeros)
    mx.save_safetensors(str(out / "model.safetensors"), tensors, metadata={"format": "mlx"})
    cfg = json.loads((bf16_bundle / "config.json").read_text())
    q = {"group_size": group_size, "bits": bits, "mode": "affine"}
    cfg["quantization"] = q; cfg["quantization_config"] = q
    cfg["mlx_ace_step"] = meta({"part": "acestep-5Hz-lm-1.7B", "dtype": "bfloat16", "layout": "mlx-lm (model.* keys, tied embedding)",
                                "quantized": f"model.layers.* Linears affine g{group_size} {bits}-bit via mx.quantize; embedding / norms bf16",
                                "from_bundle": bf16_bundle.name})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for name in TOKENIZER_FILES + ("LICENSE", "NOTICE"):
        if (bf16_bundle / name).exists():
            shutil.copyfile(bf16_bundle / name, out / name)
    print(f"{bf16_bundle.name} → {out}: {n_q} Linears quantised ({bits}-bit, group {group_size}), {len(tensors)} tensors")


def fuse_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Upstream's `_fuse_weight_norm` (acestep/models/mlx/vae_convert.py), kept operation for operation."""
    v_flat = weight_v.reshape(weight_v.shape[0], -1)
    norm = np.linalg.norm(v_flat, axis=1).reshape(weight_g.shape)
    return weight_g * weight_v / (norm + eps)


def convert_vae(weights: Path, github: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src = weights / "vae"
    sd = load_all(src / "diffusion_pytorch_model.safetensors")
    tensors: dict[str, np.ndarray] = {}
    folded = 0
    for k in sorted(sd):
        v = sd[k]
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            g = v.float().numpy(); vv = sd[base + ".weight_v"].float().numpy()
            w = fuse_weight_norm(g, vv)
            w = w.transpose(1, 2, 0) if "conv_t1" in base else w.swapaxes(1, 2)
            tensors[base + ".weight"] = np.ascontiguousarray(w, dtype=np.float32); folded += 1
        elif k.endswith(".weight_v"):
            continue
        elif k.endswith(".alpha") or k.endswith(".beta"):
            tensors[k] = np.ascontiguousarray(v.float().numpy().squeeze(), dtype=np.float32)
        else:
            tensors[k] = np.ascontiguousarray(v.float().numpy(), dtype=np.float32)
    save_np(tensors, str(out / "model.safetensors"), metadata={"format": "mlx"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["mlx_ace_step"] = meta({"part": "vae", "dtype": "float32", "conv_layout": "OKI", "weight_norm": "folded (fp32)",
                                "snake_params": "[C]", "encoder_included": True})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    licences(weights, github, out, "vae (AutoencoderOobleck, 48 kHz stereo, 64-d latents at 25 Hz)")
    print(f"vae → {out}: {len(tensors)} tensors fp32 ({folded} weight-norm folds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--github", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", default="turbo,planner,text,vae", help="also: planner-lite (the 0.6B planner, needs --planner-lite-src); planner-8bit (needs --planner-8bit-from)")
    ap.add_argument("--planner-lite-src", type=Path, default=None, help="snapshot of ACE-Step/acestep-5Hz-lm-0.6B (V2b lm06 tier)")
    ap.add_argument("--planner-8bit-from", type=Path, default=None, help="the converted ACE-Step-1.5-planner-1.7B-bf16 bundle to quantise (part planner-8bit, V4b)")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if "planner-8bit" in parts:
        assert a.planner_8bit_from, "--planner-8bit-from is required for the planner-8bit part"
        quantize_planner(a.planner_8bit_from, a.out / "ACE-Step-1.5-planner-1.7B-8bit")
    if "planner-lite" in parts:
        assert a.planner_lite_src, "--planner-lite-src is required for the planner-lite part"
        convert_qwen3(a.weights, a.github, a.out / "ACE-Step-1.5-planner-0.6B-bf16", "acestep-5Hz-lm-0.6B",
                      "acestep-5Hz-lm-0.6B (the lite 5 Hz planner, a Qwen3-0.6B fine-tune with the same 217 204-token vocabulary; "
                      "HF ACE-Step/acestep-5Hz-lm-0.6B @ 148d8ea, MIT)", src=a.planner_lite_src)
    if "turbo" in parts:
        convert_turbo(a.weights, a.github, a.out / "ACE-Step-1.5-turbo-bf16")
    if "planner" in parts:
        convert_qwen3(a.weights, a.github, a.out / "ACE-Step-1.5-planner-1.7B-bf16", "acestep-5Hz-lm-1.7B",
                      "acestep-5Hz-lm-1.7B (the 5 Hz planner, a Qwen3-1.7B fine-tune with 65 536 audio-code tokens)")
    if "text" in parts:
        convert_qwen3(a.weights, a.github, a.out / "Qwen3-Embedding-0.6B-bf16", "Qwen3-Embedding-0.6B",
                      "Qwen3-Embedding-0.6B (text encoder, Apache-2.0, Qwen)")
    if "vae" in parts:
        convert_vae(a.weights, a.github, a.out / "ACE-Step-1.5-vae-fp32")


if __name__ == "__main__":
    main()
