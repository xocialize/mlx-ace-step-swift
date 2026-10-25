#!/usr/bin/env python3
"""V3b oracle for the ACE-Step 1.5 MLX-Swift port (AB-T-0352): reference-audio timbre, source audio (cover / repaint /
extend) and LRC timestamps, from upstream ca1e85f — the pre-registered lanes in WIP/acestep-port/MEASUREMENTS.md §V3b.

The handler is loaded as the V2 oracle was (`initialize_service(device="cpu")`: torch pieces fp32 on the CPU, the official MLX
DiT and VAE fp32, MLX_ENABLE_TF32=0, VAE decode chunk 512). The planner is NOT loaded: every V3b task is direct-conditioned
(cover / repaint skip the LM upstream) or replays the V2 goldens' codes (the timbre case). One hook makes the VAE encode
reproducible: `_mlx_compiled_encode_sample` samples with `mx.random.key(seed + 1_000_000 + i)`, i counting encode calls in
upstream's order (source windows first, then the reference's); the Swift pipeline draws the same keys in the same order.

  dump_goldens_v3b.py --weights <Ace-Step1.5 snapshot> --github <ACE-Step-1.5 clone> --goldens <V2 goldens dir>
                      --cues <prompts.json> --harness <acestep-eval/harness> --out <goldens-v3b> [--cases a,b,…]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ["MLX_ENABLE_TF32"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import soundfile as sf

SR = 48000
REF_SEED = 1004
LRC_SEED = 42
ALL_CASES = ["resample-44k1", "resample-16k-mono", "timbre-folk-30s", "cover-folk-30s", "cover-half-folk-30s", "repaint-folk-30s",
             "extend-tail-folk-30s", "extend-head-folk-30s", "lrc-folk-30s", "lrc-citypop-20s"]
# declared diagnostics (MEASUREMENTS §V3b amendment): two more repaint fixtures — a different range, a different seed
DIAGNOSTIC_CASES = {"repaint-b-folk-30s": (15.0, 25.0, REF_SEED), "repaint-c-folk-30s": (10.0, 20.0, 1005)}


def save(out: Path, name: str, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    a = np.ascontiguousarray(np.asarray(arr))
    if a.dtype in (np.float64, np.float16):
        a = a.astype(np.float32)
    if a.dtype == np.bool_:
        a = a.astype(np.int32)
    if a.dtype == np.int64:
        a = a.astype(np.int32)
    np.save(out / f"{name}.npy", a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path); ap.add_argument("--github", required=True, type=Path)
    ap.add_argument("--goldens", required=True, type=Path); ap.add_argument("--cues", required=True, type=Path)
    ap.add_argument("--harness", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    args = ap.parse_args()
    sys.path.insert(0, str(args.github)); sys.path.insert(0, str(args.harness))
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(args.weights)
    import common as C  # phys_footprint sampler
    import mlx.core as mx
    import torchaudio
    from acestep.handler import AceStepHandler
    from acestep.constants import TASK_INSTRUCTIONS, DEFAULT_DIT_INSTRUCTION
    import acestep.core.generation.handler.generate_music as gm_mod
    from acestep.core.scoring.dit_alignment import MusicStampsAligner

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    cases = args.cases.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = {p["id"]: p for p in json.loads(args.cues.read_text(encoding="utf-8"))["prompts"]}
    folk = prompts["p04_folk"]
    v2_folk = args.goldens / "folk-30s"; v2_city = args.goldens / "citypop-20s"
    sampler = C.FootprintSampler(0.1)

    # ---- fixed inputs ---------------------------------------------------------------------------------------------
    src_np, sr = sf.read(v2_folk / "audio_final.wav", dtype="float32"); assert sr == SR and src_np.shape == (30 * SR, 2), src_np.shape
    city_np, sr = sf.read(v2_city / "audio_final.wav", dtype="float32"); assert sr == SR and city_np.shape == (20 * SR, 2)
    ref_np = np.tile(city_np, (2, 1))[: 30 * SR]            # exactly 30 s → upstream's segment picker is deterministic (randint(0, 0))
    inputs = args.out / "_inputs"; inputs.mkdir(exist_ok=True)
    src_path = inputs / "source_30s.wav"; ref_path = inputs / "reference_30s.wav"
    sf.write(src_path, src_np, SR, subtype="FLOAT"); sf.write(ref_path, ref_np, SR, subtype="FLOAT")
    src_rt, _ = sf.read(src_path, dtype="float32"); ref_rt, _ = sf.read(ref_path, dtype="float32")
    assert np.array_equal(src_rt, src_np) and np.array_equal(ref_rt, ref_np), "float WAV must round-trip bit for bit"
    save(inputs, "source_30s", src_np); save(inputs, "reference_30s", ref_np)

    t0 = time.perf_counter()
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=str(args.github), config_path="acestep-v15-turbo", device="cpu", offload_to_cpu=False)
    assert ok, msg
    assert dit.use_mlx_dit and dit.use_mlx_vae, (dit.use_mlx_dit, dit.use_mlx_vae)
    assert dit.dtype == torch.float32 and next(dit.model.parameters()).dtype == torch.float32
    dit.mlx_vae_chunk_size = 512
    print(f"loaded in {time.perf_counter()-t0:.1f}s: torch fp32 CPU pieces, MLX DiT fp32, MLX VAE {dit._mlx_vae_dtype}, "
          f"decode chunk 512; phys {sampler.mark()/1e9:.2f} GB", flush=True)

    # ---- the keyed encode hook (the one deliberate divergence from upstream's unseeded sampling) ----------------------
    enc_state = {"seed": REF_SEED, "i": 0, "calls": []}
    def keyed_encode(audio_nlc):
        h = dit.mlx_vae.encoder(audio_nlc)
        mean, scale = mx.split(h, 2, axis=-1)
        std = mx.where(scale > 20.0, scale, mx.log(1.0 + mx.exp(scale))) + 1e-4
        k = enc_state["seed"] + 1_000_000 + enc_state["i"]; enc_state["i"] += 1
        noise = mx.random.normal(mean.shape, key=mx.random.key(k))
        z = mean + std * noise
        mx.eval(h, z)
        enc_state["calls"].append(dict(key=k, samples=int(audio_nlc.shape[1]), frames=int(z.shape[1]),
                                       h=np.array(h, dtype=np.float32), z=np.array(z, dtype=np.float32)))
        return z
    dit._mlx_compiled_encode_sample = keyed_encode
    def reset_encode(seed):
        enc_state["seed"] = seed; enc_state["i"] = 0; enc_state["calls"] = []

    # ---- hooks: batch, refer latents, tokenize / detokenize, prepare_condition, diffusion kwargs, splice, text prompts ----
    rec = {}
    orig_prepare_batch = dit._prepare_batch
    def prepare_batch_hook(*a, **kw):
        b = orig_prepare_batch(*a, **kw)
        rec["batch"] = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in b.items() if k in ("target_latents", "src_latents", "latent_masks", "chunk_masks", "repaint_mask", "spans", "is_covers", "lyric_token_idss", "text_inputs")}
        return b
    dit._prepare_batch = prepare_batch_hook
    orig_refer = dit.infer_refer_latent
    def refer_hook(refer_audioss):
        packed, order = orig_refer(refer_audioss)
        rec["refer_packed"] = packed.detach().cpu().clone(); rec["refer_order"] = order.detach().cpu().clone()
        rec["refer_audio"] = [a.detach().cpu().clone() for a in refer_audioss[0]]
        return packed, order
    dit.infer_refer_latent = refer_hook
    orig_tokenize = dit.model.tokenize
    def tokenize_hook(x, silence_latent, attention_mask):
        q, idx, m = orig_tokenize(x, silence_latent, attention_mask)
        rec.setdefault("tokenize", []).append(dict(x=x.detach().cpu().clone(), quantized=q.detach().cpu().clone(), indices=idx.detach().cpu().clone(), mask=m.detach().cpu().clone()))
        return q, idx, m
    dit.model.tokenize = tokenize_hook
    orig_detok = dit.model.detokenize
    def detok_hook(q):
        h = orig_detok(q); rec.setdefault("detok", []).append(h.detach().cpu().clone()); return h
    dit.model.detokenize = detok_hook
    orig_prep = dit.model.prepare_condition
    def prep_hook(**kw):
        enc, mask, ctx = orig_prep(**kw)
        rec.setdefault("prepare_condition", []).append(dict(enc=enc.detach().cpu().clone(), mask=mask.detach().cpu().clone(), ctx=ctx.detach().cpu().clone(),
                                                            is_covers=kw["is_covers"].detach().cpu().clone(), src=kw["src_latents"].detach().cpu().clone(),
                                                            text_hidden=kw["text_hidden_states"].detach().cpu().clone()))
        return enc, mask, ctx
    dit.model.prepare_condition = prep_hook
    orig_diff = dit._mlx_run_diffusion
    def diff_hook(*a, **kw):
        out = orig_diff(*a, **kw)
        rec["diffusion"] = dict(seed=kw.get("seed"), audio_cover_strength=kw.get("audio_cover_strength"), repaint_crossfade_frames=kw.get("repaint_crossfade_frames"),
                                repaint_injection_ratio=kw.get("repaint_injection_ratio"), shift=kw.get("shift"), infer_steps=kw.get("infer_steps"),
                                repaint_mask=(kw.get("repaint_mask").detach().cpu().clone() if kw.get("repaint_mask") is not None else None),
                                clean_src=(kw.get("clean_src_latents").detach().cpu().clone() if kw.get("clean_src_latents") is not None else None),
                                enc_nc=(kw.get("encoder_hidden_states_non_cover").detach().cpu().clone() if kw.get("encoder_hidden_states_non_cover") is not None else None),
                                ctx_nc=(kw.get("context_latents_non_cover").detach().cpu().clone() if kw.get("context_latents_non_cover") is not None else None),
                                target=out["target_latents"].detach().cpu().clone())
        return out
    dit._mlx_run_diffusion = diff_hook
    orig_splice = gm_mod.apply_repaint_waveform_splice
    def splice_hook(pred_wavs, src_wavs, repainting_starts, repainting_ends, sample_rate=48000, crossfade_duration=0.01):
        rec["splice_in"] = pred_wavs.detach().cpu().clone(); rec["splice_src"] = src_wavs.detach().cpu().clone()
        rec["splice_args"] = dict(starts=list(map(float, repainting_starts)), ends=list(map(float, repainting_ends)), crossfade=float(crossfade_duration))
        out = orig_splice(pred_wavs, src_wavs, repainting_starts, repainting_ends, sample_rate, crossfade_duration)
        rec["splice_out"] = out.detach().cpu().clone()
        return out
    gm_mod.apply_repaint_waveform_splice = splice_hook
    orig_text = dit._prepare_text_conditioning_inputs
    def text_hook(*a, **kw):
        r = orig_text(*a, **kw)
        rec["text"] = dict(text_inputs=r[0], text_ids=r[1].clone(), lyric_ids=r[3].clone(), non_cover_ids=(r[5].clone() if r[5] is not None else None))
        return r
    dit._prepare_text_conditioning_inputs = text_hook

    folk_text = json.loads((v2_folk / "text.json").read_text(encoding="utf-8"))
    folk_planner = json.loads((v2_folk / "planner.json").read_text(encoding="utf-8"))
    folk_codes_str = "".join(f"<|audio_code_{c}|>" for c in folk_planner["codes"])

    def run_case(name, **kw):
        out = args.out / name; out.mkdir(parents=True, exist_ok=True)
        rec.clear(); reset_encode(kw["seed"])
        mx.reset_peak_memory(); sampler.peak = sampler.mark()
        t1 = time.perf_counter()
        res = dit.generate_music(**kw)
        wall = time.perf_counter() - t1
        assert res["success"], res.get("error")
        xo = res["extra_outputs"]
        audio_raw = res["audios"][0]["tensor"].float().numpy().T           # [S, 2], the handler's waveform (peak-clamped, spliced, NOT normalised)
        save(out, "audio_raw", audio_raw)
        save(out, "pred_latents", xo["pred_latents"]); save(out, "encoder_hidden", xo["encoder_hidden_states"]); save(out, "encoder_mask", xo["encoder_attention_mask"])
        save(out, "context", xo["context_latents"]); save(out, "lyric_ids", xo["lyric_token_idss"])
        if xo.get("target_latents") is not None: save(out, "target_latents", xo["target_latents"])
        if xo.get("src_latents") is not None: save(out, "src_latents", xo["src_latents"])
        if xo.get("chunk_masks") is not None: save(out, "chunk_mask", xo["chunk_masks"][..., 0])
        b = rec["batch"]
        if b.get("repaint_mask") is not None: save(out, "repaint_mask", b["repaint_mask"])
        T = int(xo["pred_latents"].shape[1])
        noise = mx.random.normal((1, T, 64), key=mx.random.key(kw["seed"])); mx.eval(noise); save(out, "noise", np.array(noise, dtype=np.float32))
        for i, c in enumerate(enc_state["calls"]):
            save(out, f"enc{i}_h", c["h"]); save(out, f"enc{i}_z", c["z"])
        if "refer_packed" in rec:
            save(out, "refer_latents", rec["refer_packed"]); save(out, "refer_order", rec["refer_order"])
            save(out, "refer_audio", rec["refer_audio"][0].T)
        for i, t in enumerate(rec.get("tokenize", [])):
            save(out, f"tok{i}_in", t["x"]); save(out, f"tok{i}_indices", t["indices"]); save(out, f"tok{i}_quantized", t["quantized"])
        for i, h in enumerate(rec.get("detok", [])):
            save(out, f"detok{i}_hints", h)
        pcs = rec.get("prepare_condition", [])
        for i, pc in enumerate(pcs):
            save(out, f"cond{i}_enc", pc["enc"]); save(out, f"cond{i}_mask", pc["mask"]); save(out, f"cond{i}_ctx", pc["ctx"]); save(out, f"cond{i}_text_hidden", pc["text_hidden"])
        d = rec["diffusion"]
        save(out, "diffusion_target", d["target"])
        if d["repaint_mask"] is not None: save(out, "diffusion_repaint_mask", d["repaint_mask"]); save(out, "diffusion_clean_src", d["clean_src"])
        if d["enc_nc"] is not None: save(out, "enc_non_cover", d["enc_nc"]); save(out, "ctx_non_cover", d["ctx_nc"])
        if "splice_in" in rec:
            save(out, "audio_decoded", rec["splice_in"][0].T); save(out, "audio_spliced", rec["splice_out"][0].T); save(out, "splice_src", rec["splice_src"][0].T)
        tx = rec["text"]
        text_prompt = tx["text_inputs"][0].split("\n\n# Languages\n")[0]
        lyric_prompt = "# Languages\n" + tx["text_inputs"][0].split("\n\n# Languages\n", 1)[1]
        meta = dict(case=name, seed=kw["seed"], frames=T, wall_seconds=wall, phys_peak_bytes=sampler.peak, mx_peak_bytes=mx.get_peak_memory(),
                    task_type=kw.get("task_type"), instruction=kw.get("instruction"), repainting_start=kw.get("repainting_start"), repainting_end=kw.get("repainting_end"),
                    audio_cover_strength=kw.get("audio_cover_strength", 1.0), chunk_mask_mode=kw.get("chunk_mask_mode"), vocal_language=kw.get("vocal_language"),
                    caption=kw.get("captions"), lyrics=kw.get("lyrics"), bpm=kw.get("bpm"), key_scale=kw.get("key_scale"), time_signature=kw.get("time_signature"),
                    audio_duration=kw.get("audio_duration"), text_prompt=text_prompt, text_ids=tx["text_ids"][0].tolist(), lyric_prompt=lyric_prompt, lyric_ids=tx["lyric_ids"][0].tolist(),
                    non_cover_text_ids=(tx["non_cover_ids"][0].tolist() if tx["non_cover_ids"] is not None else None),
                    encode_calls=[dict(key=c["key"], samples=c["samples"], frames=c["frames"]) for c in enc_state["calls"]],
                    diffusion=dict(seed=d["seed"], audio_cover_strength=d["audio_cover_strength"], repaint_crossfade_frames=d["repaint_crossfade_frames"],
                                   repaint_injection_ratio=d["repaint_injection_ratio"], shift=d["shift"], infer_steps=d["infer_steps"]),
                    splice=rec.get("splice_args"), spans=[list(s) for s in b.get("spans", [])], is_covers=b["is_covers"].tolist(),
                    time_costs=res["extra_outputs"].get("time_costs"), upstream_code="ace-step/ACE-Step-1.5@ca1e85f", weights="ACE-Step/Ace-Step1.5@19671f4",
                    source="goldens/folk-30s/audio_final.wav", reference="citypop-20s tiled to 30 s", audio_codes=(kw.get("audio_code_string") or None))
        (out / "case.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        checks = dict(pred_eq_diffusion=float((xo["pred_latents"].float() - d["target"].float()).abs().max()),
                      n_prepare_condition=len(pcs), n_encode_calls=len(enc_state["calls"]), n_tokenize=len(rec.get("tokenize", [])))
        print(f"[{name}] T={T} wall {wall:.1f}s phys peak {sampler.peak/1e9:.2f} GB mx peak {mx.get_peak_memory()/1e9:.2f} GB checks {checks}", flush=True)
        mx.clear_cache()
        return meta

    base = dict(lyrics=folk["lyrics"], vocal_language="en", inference_steps=8, guidance_scale=1.0, use_random_seed=False, batch_size=1,
                shift=1.0, infer_method="ode", use_tiled_decode=True, repaint_mode="balanced", repaint_strength=0.5)

    if "resample-44k1" in cases or "resample-16k-mono" in cases:
        src_t = torch.from_numpy(src_np.T.copy())                                              # [2, S]
        if "resample-44k1" in cases:
            out = args.out / "resample-44k1"; out.mkdir(exist_ok=True)
            clip = src_t[:, : 5 * SR]
            down = torchaudio.functional.resample(clip, SR, 44100)
            up = torchaudio.transforms.Resample(44100, SR)(down)                              # the loader's exact call
            save(out, "input", down.T); save(out, "expected", up.T)
            (out / "case.json").write_text(json.dumps(dict(orig_freq=44100, new_freq=SR, channels=2, input_samples=int(down.shape[1]), expected_samples=int(up.shape[1]))), encoding="utf-8")
            print(f"[resample-44k1] {tuple(down.shape)} → {tuple(up.shape)}", flush=True)
        if "resample-16k-mono" in cases:
            out = args.out / "resample-16k-mono"; out.mkdir(exist_ok=True)
            mono = src_t[:, 5 * SR: 8 * SR].mean(dim=0, keepdim=True)
            down = torchaudio.functional.resample(mono, SR, 16000)
            # the loader: mono → duplicated to stereo, then resampled (the order upstream's _normalize_audio_to_stereo_48k uses)
            up = torchaudio.transforms.Resample(16000, SR)(torch.cat([down, down], dim=0))
            save(out, "input", down.T); save(out, "expected", up.T)
            (out / "case.json").write_text(json.dumps(dict(orig_freq=16000, new_freq=SR, channels=1, input_samples=int(down.shape[1]), expected_samples=int(up.shape[1]))), encoding="utf-8")
            print(f"[resample-16k-mono] {tuple(down.shape)} → {tuple(up.shape)}", flush=True)

    if "timbre-folk-30s" in cases:
        # the V2 folk-30s request exactly (the planner's caption / metas / codes replayed) plus the reference — text2music + codes
        # resolves to the cover instruction with the codes as hints, as V2's W lane ran it
        meta = run_case("timbre-folk-30s", task_type="text2music", audio_code_string=folk_codes_str, reference_audio=str(ref_path),
                        captions=folk_text["caption"], bpm=folk_planner["metadata"]["bpm"], key_scale=folk_planner["metadata"]["keyscale"],
                        time_signature=str(folk_planner["metadata"]["timesignature"]), audio_duration=30, seed=REF_SEED,
                        instruction=DEFAULT_DIT_INSTRUCTION, chunk_mask_mode="auto", **base)
        assert meta["text_prompt"] == folk_text["text_prompt"], "the timbre case must reproduce the V2 folk-30s text prompt"
        assert meta["lyric_prompt"] == folk_text["lyric_prompt"]
        # the diffusers fp32 encoder on the same 30 s reference (the D-row cross-check), and the MLX encoder's mean beside
        out = args.out / "timbre-folk-30s"
        with torch.inference_mode():
            ref_t = torch.from_numpy(ref_np.T.copy())[None]                                 # [1, 2, S]
            dist = dit.vae.encode(ref_t).latent_dist
            save(out, "refer_mean_diffusers", dist.mean[0].T); save(out, "refer_std_diffusers", dist.std[0].T)
        h = dit.mlx_vae.encoder(mx.array(ref_np[None])); mean, scale = mx.split(h, 2, axis=-1); mx.eval(mean)
        d_mean = float(np.linalg.norm(np.array(mean)[0] - dist.mean[0].T.numpy()) / np.linalg.norm(dist.mean[0].T.numpy()))
        print(f"[timbre-folk-30s] MLX encoder mean vs diffusers fp32: relL2 {d_mean:.3e} ({-20*np.log10(max(d_mean,1e-30)):.1f} dB)", flush=True)
        (out / "checks.json").write_text(json.dumps(dict(mlx_encoder_mean_vs_diffusers_relL2=d_mean)), encoding="utf-8")

    for name, strength in (("cover-folk-30s", 1.0), ("cover-half-folk-30s", 0.5)):
        if name in cases:
            run_case(name, task_type="cover", src_audio=str(src_path), audio_code_string="", captions=folk["caption"], bpm=None, key_scale="", time_signature="",
                     audio_duration=None, seed=REF_SEED, instruction=TASK_INSTRUCTIONS["cover"], audio_cover_strength=strength, chunk_mask_mode="auto", **base)

    for name, start, end in (("repaint-folk-30s", 10.0, 20.0), ("extend-tail-folk-30s", 30.0, 40.0), ("extend-head-folk-30s", -10.0, 0.0)):
        if name in cases:
            run_case(name, task_type="repaint", src_audio=str(src_path), audio_code_string="", captions=folk["caption"], bpm=None, key_scale="", time_signature="",
                     audio_duration=None, seed=REF_SEED, instruction=TASK_INSTRUCTIONS["repaint"], repainting_start=start, repainting_end=end,
                     chunk_mask_mode="explicit", **base)
    for name, (start, end, seed) in DIAGNOSTIC_CASES.items():
        if name in cases:
            run_case(name, task_type="repaint", src_audio=str(src_path), audio_code_string="", captions=folk["caption"], bpm=None, key_scale="", time_signature="",
                     audio_duration=None, seed=seed, instruction=TASK_INSTRUCTIONS["repaint"], repainting_start=start, repainting_end=end,
                     chunk_mask_mode="explicit", **base)

    # ---- LRC on the V2 goldens --------------------------------------------------------------------------------------
    lrc_rec = {}
    orig_noise = dit._sample_noise_like
    def noise_hook(reference, seed):
        n = orig_noise(reference, seed); lrc_rec["noise"] = n.detach().cpu().clone(); return n
    dit._sample_noise_like = noise_hook
    orig_align = MusicStampsAligner.stamps_align_info
    def align_hook(self, attention_matrix, lyrics_tokens, total_duration_seconds, custom_config, return_matrices=False, violence_level=2.0, medfilt_width=1):
        lrc_rec["attention"] = attention_matrix.detach().cpu().float().clone(); lrc_rec["custom_config"] = {int(k): list(v) for k, v in custom_config.items()}
        lrc_rec["violence_level"] = violence_level; lrc_rec["medfilt_width"] = medfilt_width
        r = orig_align(self, attention_matrix, lyrics_tokens, total_duration_seconds, custom_config, return_matrices=True, violence_level=violence_level, medfilt_width=medfilt_width)
        lrc_rec["calc_matrix"] = np.array(r["calc_matrix"]); lrc_rec["energy_matrix"] = np.array(r.get("energy_matrix")); lrc_rec["lyrics_tokens"] = list(lyrics_tokens)
        return r
    MusicStampsAligner.stamps_align_info = align_hook
    orig_stamps = MusicStampsAligner.get_timestamps_and_lrc
    def stamps_hook(self, calc_matrix, lyrics_tokens, total_duration_seconds):
        r = orig_stamps(self, calc_matrix, lyrics_tokens, total_duration_seconds)
        lrc_rec["token_stamps"] = [dict(token_id=t.token_id, text=t.text, start=t.start, end=t.end) for t in r["token_timestamps"]]
        lrc_rec["sentence_stamps"] = [dict(text=s.text, start=s.start, end=s.end, confidence=s.confidence) for s in r["sentence_timestamps"]]
        return r
    MusicStampsAligner.get_timestamps_and_lrc = stamps_hook
    for name, gdir in (("lrc-folk-30s", v2_folk), ("lrc-citypop-20s", v2_city)):
        if name not in cases:
            continue
        out = args.out / name; out.mkdir(exist_ok=True); lrc_rec.clear()
        req = json.loads((gdir / "request.json").read_text(encoding="utf-8")); text = json.loads((gdir / "text.json").read_text(encoding="utf-8"))
        lat = torch.from_numpy(np.load(gdir / "latents.npy")); enc = torch.from_numpy(np.load(gdir / "encoder_hidden.npy"))
        mask = torch.from_numpy(np.load(gdir / "encoder_mask.npy")).float(); ctx = torch.from_numpy(np.load(gdir / "context.npy"))
        lyric_ids = torch.tensor([text["lyric_ids"]], dtype=torch.long)
        t1 = time.perf_counter()
        r = dit.get_lyric_timestamp(pred_latent=lat, encoder_hidden_states=enc, encoder_attention_mask=mask, context_latents=ctx, lyric_token_ids=lyric_ids,
                                    total_duration_seconds=float(req["duration"]), vocal_language=text["language"], inference_steps=8, seed=LRC_SEED)
        wall = time.perf_counter() - t1
        assert r["success"], r["error"]
        save(out, "noise", lrc_rec["noise"]); save(out, "attention", lrc_rec["attention"]); save(out, "calc_matrix", lrc_rec["calc_matrix"]); save(out, "energy_matrix", lrc_rec["energy_matrix"])
        header_ids = dit.text_tokenizer.encode(f"# Languages\n{text['language']}\n\n# Lyric\n", add_special_tokens=False)
        (out / "lrc.json").write_text(json.dumps(dict(case=name, source_golden=gdir.name, lrc_text=r["lrc_text"], token_stamps=lrc_rec["token_stamps"], sentence_stamps=lrc_rec["sentence_stamps"],
                                                      lyrics_tokens=lrc_rec["lyrics_tokens"], header_len=len(header_ids), lyric_ids=text["lyric_ids"], vocal_language=text["language"],
                                                      total_duration_seconds=float(req["duration"]), inference_steps=8, seed=LRC_SEED, t_last=1.0 / 8,
                                                      custom_config={str(k): v for k, v in lrc_rec["custom_config"].items()}, violence_level=lrc_rec["violence_level"],
                                                      medfilt_width=lrc_rec["medfilt_width"], attention_shape=list(lrc_rec["attention"].shape), wall_seconds=wall),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{name}] attention {tuple(lrc_rec['attention'].shape)} → {len(lrc_rec['sentence_stamps'])} lines in {wall:.1f}s; first: {r['lrc_text'].splitlines()[:2]}", flush=True)

    print(f"ALL DONE; phys peak {sampler.stop()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
