#!/usr/bin/env python3
"""V1 gate (AB-T-0352): the converted bundles equal upstream's OWN conversions bit for bit, and MLX's conv semantics on the
converted layouts reproduce torch.  Run with MLX_ENABLE_TF32=0 and the acestep venv (upstream importable).

  (1) DiT bundle: `decoder.*` == acestep.models.mlx.dit_convert.convert_decoder_weights(torch model) (fp32 compare, max|Δ| 0,
      same key set); every non-decoder key == the checkpoint; silence_latent == silence_latent.pt transposed.
  (2) VAE bundle == acestep.models.mlx.vae_convert.convert_vae_weights(diffusers AutoencoderOobleck) (max|Δ| 0, same keys).
  (3) planner + text encoder: mlx_lm.load(bundle) succeeds; every tensor == checkpoint (bf16 bit-equal).
  (4) MLX conv1d / conv_transpose1d on the converted layouts == torch on proj_in (k2 s2), proj_out (k2 s2), the VAE's first
      decoder block (conv_t1 k20 s10 p5) and a dilated residual conv (k7 d3): relmax ≤ 1e-5.

    gate_v1.py --weights <snapshot> --github <clone> --bundles <converted dir>
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx
from mlx.utils import tree_flatten


def read_all(path: Path):
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def compare(name, got: dict, ref: dict, to_np=lambda t: t.float().numpy()) -> bool:
    missing, extra = sorted(set(ref) - set(got)), sorted(set(got) - set(ref))
    worst, n = 0.0, 0
    for k in ref:
        if k not in got: continue
        a, b = np.asarray(to_np(got[k])), np.asarray(to_np(ref[k]))
        if a.shape != b.shape:
            print(f"  {name}: SHAPE {k} {a.shape} vs {b.shape}"); worst = float("inf"); continue
        worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())); n += 1
    ok = worst == 0.0 and not missing and not extra
    print(f"{name}: {n} tensors compared, max|Δ| = {worst:.3e}, missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]} → {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True); ap.add_argument("--github", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.github))
    from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
    from acestep.models.mlx.dit_convert import convert_decoder_weights
    from acestep.models.mlx.vae_convert import convert_vae_weights
    from diffusers.models import AutoencoderOobleck
    results = []

    # (1) DiT bundle
    turbo_src = a.weights / "acestep-v15-turbo"
    model = AceStepConditionGenerationModel.from_pretrained(str(turbo_src), torch_dtype=torch.bfloat16).eval()
    ref_dec = {k: np.array(v) for k, v in convert_decoder_weights(model)}
    bundle = read_all(a.bundles / "ACE-Step-1.5-turbo-bf16" / "model.safetensors")
    got_dec = {k[len("decoder."):]: v for k, v in bundle.items() if k.startswith("decoder.")}
    results.append(compare("DiT decoder.* vs upstream dit_convert", got_dec, ref_dec, to_np=lambda t: t.float().numpy() if torch.is_tensor(t) else t))
    ckpt = read_all(turbo_src / "model.safetensors")
    got_rest = {k: v for k, v in bundle.items() if not k.startswith("decoder.")}
    ref_rest = {k: v for k, v in ckpt.items() if not k.startswith("decoder.")}
    results.append(compare("DiT non-decoder keys vs checkpoint", got_rest, ref_rest))
    results.append(all(v.dtype == torch.bfloat16 for v in bundle.values()))
    print(f"DiT bundle dtypes all bf16: {results[-1]}")
    with safe_open(str(a.bundles / "ACE-Step-1.5-turbo-bf16" / "silence_latent.safetensors"), "np") as f:
        sil = f.get_tensor("silence_latent")
    sil_ref = torch.load(turbo_src / "silence_latent.pt", weights_only=True).transpose(1, 2)[0].numpy()
    ok = sil.shape == sil_ref.shape and np.array_equal(sil, sil_ref)
    print(f"silence_latent [{sil.shape}] bit-equal: {ok}"); results.append(ok)

    # (2) VAE bundle
    vae = AutoencoderOobleck.from_pretrained(str(a.weights / "vae")).eval()
    ref_vae = {k: np.array(v) for k, v in convert_vae_weights(vae)}
    with safe_open(str(a.bundles / "ACE-Step-1.5-vae-fp32" / "model.safetensors"), "np") as f:
        got_vae = {k: f.get_tensor(k) for k in f.keys()}
    results.append(compare("VAE vs upstream vae_convert", got_vae, ref_vae, to_np=lambda t: t))
    results.append(all(v.dtype == np.float32 for v in got_vae.values())); print(f"VAE bundle dtypes all fp32: {results[-1]}")

    # (3) planner + text encoder through mlx_lm
    from mlx_lm import load as mlx_load
    for folder, bundle_name in [("acestep-5Hz-lm-1.7B", "ACE-Step-1.5-planner-1.7B-bf16"), ("Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B-bf16")]:
        m, tok = mlx_load(str(a.bundles / bundle_name))
        n_params = len(dict(tree_flatten(m.parameters())))
        got = read_all(a.bundles / bundle_name / "model.safetensors")
        ref = {"model." + k: v for k, v in read_all(a.weights / folder / "model.safetensors").items()}
        results.append(compare(f"{folder} bundle vs checkpoint (model.* prefix)", got, ref))
        # the mlx-lm model's parameter tree must consume exactly the bundle's keys
        flat = dict(tree_flatten(m.parameters()))
        ok = set(flat) == set(got)
        print(f"{folder}: mlx_lm.load OK ({n_params} parameters, tokenizer {type(tok).__name__}); parameter keys == bundle keys: {ok}")
        results.append(ok)
        del m

    # (4) MLX conv semantics on the converted layouts
    torch.manual_seed(0)
    dec = model.decoder.float()
    def relmax(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    x = torch.randn(1, 10, 192)
    with torch.no_grad(): y_t = dec.proj_in(x)                        # [1, 5, 2048]
    w = bundle["decoder.proj_in.weight"].float().numpy(); b = bundle["decoder.proj_in.bias"].float().numpy()
    y_m = mx.conv1d(mx.array(x.numpy()), mx.array(w), stride=2, padding=0) + mx.array(b)
    e_in = relmax(np.array(y_m), y_t.numpy())
    h = torch.randn(1, 5, 2048)
    with torch.no_grad(): z_t = dec.proj_out(h)                       # [1, 10, 64]
    w = bundle["decoder.proj_out.weight"].float().numpy(); b = bundle["decoder.proj_out.bias"].float().numpy()
    z_m = mx.conv_transpose1d(mx.array(h.numpy()), mx.array(w), stride=2, padding=0) + mx.array(b)
    e_out = relmax(np.array(z_m), z_t.numpy())
    vae = vae.float()
    blk = vae.decoder.block[0]
    xx = torch.randn(1, 2048, 9)
    with torch.no_grad(): ct = blk.conv_t1(xx)                        # [1, 1024, 90]
    w = got_vae["decoder.block.0.conv_t1.weight"]; b = got_vae["decoder.block.0.conv_t1.bias"]
    cm = mx.conv_transpose1d(mx.array(xx.permute(0, 2, 1).numpy()), mx.array(w), stride=10, padding=5) + mx.array(b)
    e_ct = relmax(np.array(cm).transpose(0, 2, 1), ct.numpy())
    ru = blk.res_unit2.conv1                                           # k7 dilation 3 pad 9
    xr = torch.randn(1, 1024, 40)
    with torch.no_grad(): rt = ru(xr)
    w = got_vae["decoder.block.0.res_unit2.conv1.weight"]; b = got_vae["decoder.block.0.res_unit2.conv1.bias"]
    rm = mx.conv1d(mx.array(xr.permute(0, 2, 1).numpy()), mx.array(w), stride=1, padding=9, dilation=3) + mx.array(b)
    e_ru = relmax(np.array(rm).transpose(0, 2, 1), rt.numpy())
    print(f"MLX conv semantics relmax: proj_in {e_in:.2e}, proj_out {e_out:.2e}, VAE conv_t1(k20,s10,p5) {e_ct:.2e}, res conv(k7,d3) {e_ru:.2e}")
    results.append(max(e_in, e_out, e_ct, e_ru) <= 1e-5)
    print("V1 GATE:", "PASS" if all(results) else "FAIL", results)


if __name__ == "__main__":
    main()
