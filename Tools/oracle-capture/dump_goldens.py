#!/usr/bin/env python3
"""V0/V2 oracle for the ACE-Step 1.5 MLX-Swift port (AB-T-0352): per-case goldens from upstream ca1e85f.

The torch pieces (Qwen3-Embedding text encoder, condition encoder, FSQ quantizer + detokenizer, the torch DiT as the N
reference, the diffusers Oobleck decoder as the D reference) run fp32 on the CPU (`initialize_service(device="cpu")`); the
official Apple-Silicon MLX path (fp32 `MLXDiTDecoder`, `mlx_generate_diffusion` with native-Haar DCW, the chunked fp32
`MLXAutoEncoderOobleck` at 512/64) is what the pipeline runs for the E and D(mlx) goldens; the planner is the bf16 mlx-lm
model for the SAMPLED rollout (reference only, AB-L-0090) and is cast to fp32 for the teacher-forced logits (lane P).
Runs with MLX_ENABLE_TF32=0.

  dump_goldens.py --weights <Ace-Step1.5 snapshot> --github <ACE-Step-1.5 clone> --cues <prompts.json> --out <goldens>
                  [--cases folk-30s,citypop-20s]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ["MLX_ENABLE_TF32"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ACESTEP_LM_BACKEND", "mlx")
import numpy as np
import torch

CASES = {
    "folk-30s": dict(prompt_id="p04_folk", duration=30, seed=1004),
    "citypop-20s": dict(prompt_id="p11_zh_citypop", duration=20, seed=1011),
}
TAIL = 16          # lane P keeps full logits for the last TAIL positions (argmax for all)
P_CODES = 64       # lane P teacher-forces the codes prompt + this many oracle codes
AUDIO_CODE_BASE = 151669


def fnv1a64(ids: np.ndarray) -> str:
    h = 0xCBF29CE484222325
    for v in np.sort(ids.astype(np.int64)):
        for byte in int(v).to_bytes(4, "little", signed=False):
            h ^= byte; h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path); ap.add_argument("--github", required=True, type=Path)
    ap.add_argument("--cues", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()
    sys.path.insert(0, str(args.github)); sys.path.insert(0, str(args.cues.parent.parent / "harness"))
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(args.weights)
    import common as C  # phys_footprint sampler (WIP/acestep-eval/harness/common.py)
    import mlx.core as mx
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    from acestep.constrained_logits_processor import FSMState
    from acestep.models.mlx.dit_generate import mlx_generate_diffusion, get_timestep_schedule

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    prompts = {p["id"]: p for p in json.loads(args.cues.read_text(encoding="utf-8"))["prompts"]}
    sampler = C.FootprintSampler(0.1)

    t0 = time.perf_counter()
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=str(args.github), config_path="acestep-v15-turbo", device="cpu", offload_to_cpu=False)
    assert ok, msg
    assert dit.use_mlx_dit and dit.use_mlx_vae, (dit.use_mlx_dit, dit.use_mlx_vae)
    assert dit.dtype == torch.float32 and next(dit.model.parameters()).dtype == torch.float32
    dit.mlx_vae_chunk_size = 512
    llm = LLMHandler()
    msg, ok = llm.initialize(checkpoint_dir=str(args.weights), lm_model_path="acestep-5Hz-lm-1.7B", backend="mlx", device="auto",
                             offload_to_cpu=False, dtype=None)
    assert ok, msg
    print(f"loaded in {time.perf_counter()-t0:.1f}s: torch fp32 CPU pieces, MLX DiT fp32, MLX VAE {dit._mlx_vae_dtype}, "
          f"chunk {dit.mlx_vae_chunk_size}; phys {sampler.mark()/1e9:.2f} GB", flush=True)
    tok = llm.llm_tokenizer
    eos = tok.eos_token_id

    # ---- hooks: the planner's prompts + generated ids, the DiT's text/lyric prompts + ids ----
    lm_calls = []
    orig_run = llm._run_mlx_single_native
    def run_hook(formatted_prompt, *a, **kw):
        out = orig_run(formatted_prompt, *a, **kw)
        lm_calls.append(dict(prompt=formatted_prompt, phase=kw.get("generation_phase"), cot_text=kw.get("cot_text"),
                             caption=kw.get("caption"), lyrics=kw.get("lyrics"), output=out,
                             negative_prompt=kw.get("negative_prompt"), cfg_scale=kw.get("cfg_scale"),
                             temperature=kw.get("temperature"), top_p=kw.get("top_p"), top_k=kw.get("top_k"),
                             target_duration=kw.get("target_duration"), user_metadata=kw.get("user_metadata")))
        return out
    llm._run_mlx_single_native = run_hook
    decoded_lists = []
    orig_decode = tok.decode
    def decode_hook(ids, *a, **kw):
        if hasattr(ids, "__len__") and len(ids) > 1:
            decoded_lists.append([int(i) for i in ids])
        return orig_decode(ids, *a, **kw)
    tok.decode = decode_hook
    text_calls = []
    orig_text = dit._prepare_text_conditioning_inputs
    def text_hook(*a, **kw):
        r = orig_text(*a, **kw)
        text_calls.append(dict(text_inputs=r[0], text_ids=r[1].clone(), text_mask=r[2].clone(), lyric_ids=r[3].clone(),
                               lyric_mask=r[4].clone(), captions=kw.get("captions") or a[2], instructions=kw.get("instructions") or a[1],
                               lyrics=kw.get("lyrics") or a[3], metas=kw.get("parsed_metas") or a[4], languages=kw.get("vocal_languages") or a[5]))
        return r
    dit._prepare_text_conditioning_inputs = text_hook

    for name in args.cases.split(","):
        spec = CASES[name]; p = prompts[spec["prompt_id"]]
        out = args.out / name; out.mkdir(parents=True, exist_ok=True)
        lm_calls.clear(); decoded_lists.clear(); text_calls.clear()
        timing = {}
        params = GenerationParams(task_type="text2music", thinking=True, caption=p["caption"], lyrics=p["lyrics"],
                                  vocal_language=p["language"], duration=spec["duration"], inference_steps=8, seed=spec["seed"])
        config = GenerationConfig(batch_size=1, audio_format="wav", seeds=[spec["seed"]], use_random_seed=False)
        mx.reset_peak_memory(); phys_before = sampler.mark(); sampler.peak = phys_before
        t1 = time.perf_counter()
        res = generate_music(dit, llm, params=params, config=config, save_dir=str(out / "_raw"))
        timing["pipeline_seconds"] = time.perf_counter() - t1
        assert res.success, (res.status_message, res.error)
        tc = res.extra_outputs.get("time_costs", {})
        print(f"[{name}] pipeline {timing['pipeline_seconds']:.1f}s; time_costs {json.dumps(tc)[:300]}; phys peak {sampler.peak/1e9:.2f} GB; "
              f"mx peak {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
        audio_params = res.audios[0].get("params", {})
        codes_str = audio_params.get("audio_codes", "")
        codes = dit._parse_audio_code_string(codes_str)
        assert len(codes) == spec["duration"] * 5, (len(codes), spec["duration"])
        assert len(lm_calls) == 2 and lm_calls[0]["phase"] == "cot" and lm_calls[1]["phase"] == "codes", [c["phase"] for c in lm_calls]
        assert len(text_calls) == 1

        # ---- planner goldens ----
        cot_call, codes_call = lm_calls
        cot_out_ids, codes_out_ids = decoded_lists[0], decoded_lists[-1]
        assert tok.decode(cot_out_ids, skip_special_tokens=False) == cot_call["output"]
        assert tok.decode(codes_out_ids, skip_special_tokens=False) == codes_call["output"]
        assert all(AUDIO_CODE_BASE <= i < AUDIO_CODE_BASE + 64000 for i in codes_out_ids[:-1]) and codes_out_ids[-1] == eos, codes_out_ids[-5:]
        assert [i - AUDIO_CODE_BASE for i in codes_out_ids[:-1]] == codes
        metadata, _ = llm.parse_lm_output(cot_call["output"])
        cot_block = llm._format_metadata_as_cot(metadata)
        assert cot_block == codes_call["cot_text"], (cot_block, codes_call["cot_text"])
        uncond = llm._build_unconditional_prompt(codes_call["caption"], codes_call["lyrics"], cot_block, codes_call["negative_prompt"], "codes", False)
        cot_prompt_ids = tok(cot_call["prompt"]).input_ids
        codes_prompt_ids = tok(codes_call["prompt"]).input_ids
        uncond_ids = tok(uncond).input_ids
        planner = dict(cot_prompt=cot_call["prompt"], cot_prompt_ids=cot_prompt_ids, cot_output=cot_call["output"], cot_output_ids=cot_out_ids,
                       metadata=metadata, cot_block=cot_block, codes_prompt=codes_call["prompt"], codes_prompt_ids=codes_prompt_ids,
                       uncond_prompt=uncond, uncond_prompt_ids=uncond_ids, codes=codes, codes_output_ids=codes_out_ids,
                       eos_token_id=eos, pad_token_id=tok.pad_token_id, audio_code_base=AUDIO_CODE_BASE,
                       settings=dict(temperature=codes_call["temperature"], cfg_scale=codes_call["cfg_scale"], top_p=codes_call["top_p"],
                                     top_k=codes_call["top_k"], negative_prompt=codes_call["negative_prompt"],
                                     target_duration=codes_call["target_duration"], user_metadata=cot_call["user_metadata"],
                                     cot_cfg_scale=cot_call["cfg_scale"]))
        json.dump(planner, (out / "planner.json").open("w"), ensure_ascii=False, indent=1)

        # ---- DiT text goldens (fp32 CPU) ----
        tcall = text_calls[0]
        text_ids = tcall["text_ids"][0]; lyric_ids = tcall["lyric_ids"][0]
        assert bool(tcall["text_mask"].all()) and bool(tcall["lyric_mask"].all())
        text_prompt = tcall["text_inputs"][0].split("\n\n# Languages\n")[0]
        lyric_prompt = "# Languages\n" + tcall["text_inputs"][0].split("\n\n# Languages\n", 1)[1]
        assert tok_ids_equal(dit.text_tokenizer, text_prompt, text_ids, 256) and tok_ids_equal(dit.text_tokenizer, lyric_prompt, lyric_ids, 2048)
        with torch.inference_mode():
            text_hidden = dit.text_encoder(input_ids=text_ids[None]).last_hidden_state.float()
            lyric_embed = dit.text_encoder.embed_tokens(lyric_ids[None]).float()
        json.dump(dict(text_prompt=text_prompt, text_ids=text_ids.tolist(), lyric_prompt=lyric_prompt, lyric_ids=lyric_ids.tolist(),
                       caption=tcall["captions"][0], instruction=tcall["instructions"][0], metas=tcall["metas"][0],
                       language=tcall["languages"][0], lyrics=tcall["lyrics"][0], eos_appended=int(text_ids[-1])),
                  (out / "text.json").open("w"), ensure_ascii=False, indent=1)
        np.save(out / "text_hidden.npy", text_hidden.numpy()); np.save(out / "lyric_embed.npy", lyric_embed.numpy())

        # ---- hints (lane F), conditions (lane C), context ----
        T = max(128, len(codes) * 5)
        with torch.inference_mode():
            hints = dit._decode_audio_codes_to_latents(codes_str).float()           # [1, T, 64]
            assert hints.shape[1] == len(codes) * 5
            sil = dit.silence_latent.float()                                          # [1, 15000, 64]
            enc, enc_mask = dit.model.encoder(text_hidden_states=text_hidden, text_attention_mask=torch.ones(1, text_ids.numel(), dtype=torch.bool),
                                              lyric_hidden_states=lyric_embed, lyric_attention_mask=torch.ones(1, lyric_ids.numel(), dtype=torch.bool),
                                              refer_audio_acoustic_hidden_states_packed=sil[:, :750, :], refer_audio_order_mask=torch.tensor([0]))
        src = hints[:, :T]
        if src.shape[1] < T:
            src = torch.cat([src, sil[:, :T - src.shape[1]]], dim=1)
        ctx = torch.cat([src, torch.ones(1, T, 64)], dim=-1)
        xo = res.extra_outputs
        checks = {}
        # The pipeline's own packed conditions are the golden; the direct recomputation above is a cross-check (it differs at
        # the 1e-6 level — a different kernel path for the same fp32 math — and the 8-step solve amplifies that, see MEASUREMENTS).
        enc_direct = enc.float()
        if xo.get("encoder_hidden_states") is not None:
            enc = xo["encoder_hidden_states"].float().cpu()
            checks["encoder_direct_vs_pipeline_maxabs"] = float((enc - enc_direct).abs().max())
            checks["encoder_direct_vs_pipeline_relL2"] = float((enc - enc_direct).norm() / enc.norm())
        if xo.get("context_latents") is not None:
            checks["context_vs_pipeline_maxabs"] = float((xo["context_latents"].float() - ctx).abs().max())
        np.save(out / "hints.npy", hints.numpy()); np.save(out / "encoder_hidden.npy", enc.numpy()); np.save(out / "encoder_hidden_direct.npy", enc_direct.numpy())
        np.save(out / "encoder_mask.npy", enc_mask.numpy().astype(np.int32)); np.save(out / "context.npy", ctx.numpy())

        # ---- noise + the 8-step solve (lane E, official MLX path) ----
        shift = float(params.shift)   # the pipeline default is 1.0 → the linspace schedule [1, .875, …, .125], NOT the shift-3 turbo table
        schedule = get_timestep_schedule(shift=shift, timesteps=None, infer_steps=params.inference_steps)
        noise = mx.random.normal((1, T, 64), key=mx.random.key(spec["seed"]))
        mx.eval(noise); np.save(out / "noise.npy", np.array(noise, dtype=np.float32))
        t1 = time.perf_counter()
        gen = mlx_generate_diffusion(dit.mlx_decoder, enc.numpy(), ctx.numpy(), (1, T, 64), seed=[spec["seed"]], infer_method="ode",
                                     shift=shift, infer_steps=params.inference_steps, guidance_scale=1.0, disable_tqdm=True, dcw_enabled=True, dcw_mode="double",
                                     dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar")
        timing["mlx_solve_seconds"] = time.perf_counter() - t1
        latents = gen["target_latents"].astype(np.float32)                           # [1, T, 64]
        gen_d = mlx_generate_diffusion(dit.mlx_decoder, enc_direct.numpy(), ctx.numpy(), (1, T, 64), seed=[spec["seed"]], infer_method="ode",
                                       shift=shift, infer_steps=params.inference_steps, guidance_scale=1.0, disable_tqdm=True, dcw_enabled=True,
                                       dcw_mode="double", dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar")["target_latents"]
        checks["solve_sensitivity_direct_enc_relL2"] = float(np.linalg.norm(gen_d - latents) / np.linalg.norm(latents))
        np.save(out / "latents_from_direct_enc.npy", gen_d.astype(np.float32))
        if xo.get("pred_latents") is not None:
            checks["latents_vs_pipeline_maxabs"] = float(np.abs(xo["pred_latents"].float().numpy() - latents).max())
            assert checks["latents_vs_pipeline_maxabs"] == 0.0, "the direct mlx_generate_diffusion call must reproduce the pipeline's latents"
        np.save(out / "latents.npy", latents)
        json.dump(dict(schedule=schedule, shift=shift, steps=params.inference_steps, dcw=dict(mode="double", scaler=0.05, high_scaler=0.02, wavelet="haar"),
                       params=params.to_dict()), (out / "sampler.json").open("w"), indent=1, default=str)

        # ---- lane N: one velocity at t = 0.75 on x = 0.75·noise + 0.25·latents ----
        x075 = 0.75 * np.array(noise, dtype=np.float32) + 0.25 * latents
        tt = 0.75
        t1 = time.perf_counter()
        v_mlx, _ = dit.mlx_decoder(hidden_states=mx.array(x075), timestep=mx.full((1,), tt), timestep_r=mx.full((1,), tt),
                                   encoder_hidden_states=mx.array(enc.numpy()), context_latents=mx.array(ctx.numpy()), cache=None, use_cache=False)
        mx.eval(v_mlx); timing["mlx_velocity_seconds"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        with torch.inference_mode():
            v_torch = dit.model.decoder(hidden_states=torch.from_numpy(x075), timestep=torch.full((1,), tt), timestep_r=torch.full((1,), tt),
                                        attention_mask=torch.ones(1, T), encoder_hidden_states=enc, encoder_attention_mask=enc_mask,
                                        context_latents=ctx, use_cache=False)[0].float().numpy()
        timing["torch_velocity_seconds"] = time.perf_counter() - t1
        np.save(out / "dit_x_t075.npy", x075); np.save(out / "dit_v_t075_torch.npy", np.ascontiguousarray(v_torch)); np.save(out / "dit_v_t075_mlx.npy", np.array(v_mlx, dtype=np.float32))
        d = np.abs(v_torch - np.array(v_mlx)); checks["velocity_torch_vs_mlx_relL2"] = float(np.linalg.norm(d) / np.linalg.norm(v_torch))

        # ---- lane D: VAE decodes ----
        t1 = time.perf_counter()
        with torch.inference_mode():
            audio_torch = dit.vae.decode(torch.from_numpy(latents).transpose(1, 2)).sample[0].T.contiguous().float().numpy()   # [S, 2]
        timing["torch_vae_seconds"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        audio_mlx = dit._mlx_decode_single(mx.array(latents), decode_fn=None); mx.eval(audio_mlx)
        timing["mlx_vae_chunked_seconds"] = time.perf_counter() - t1
        audio_mlx = np.array(audio_mlx, dtype=np.float32)[0]
        audio_mlx_full = np.array(dit.mlx_vae.decode(mx.array(latents)), dtype=np.float32)[0] if T <= 1024 else None
        np.save(out / "audio_torch_full.npy", audio_torch); np.save(out / "audio_mlx_chunked.npy", audio_mlx)
        if audio_mlx_full is not None:
            np.save(out / "audio_mlx_full.npy", audio_mlx_full)
            checks["mlx_chunked_vs_full_relL2"] = float(np.linalg.norm(audio_mlx - audio_mlx_full) / np.linalg.norm(audio_mlx_full))
        checks["vae_mlx_vs_torch_relL2"] = float(np.linalg.norm(audio_mlx - audio_torch) / np.linalg.norm(audio_torch))
        final = res.audios[0]["tensor"].float().numpy().T                                  # [S, 2], peak-normalised to −1 dBFS
        np.save(out / "audio_final.npy", final)
        import soundfile as sf
        sf.write(out / "audio_final.wav", final, 48000, subtype="PCM_24")
        checks["final_peak"] = float(np.abs(final).max())

        # ---- lane P: teacher-forced logits (fp32) ----
        model = llm._mlx_model
        model.set_dtype(mx.float32); mx.eval(model.parameters())
        def teacher(ids, tag):
            x = mx.array(np.asarray(ids, dtype=np.int32))[None]
            logits = model(x)[0].astype(mx.float32); mx.eval(logits)
            np.save(out / f"p_{tag}_ids.npy", np.asarray(ids, dtype=np.int32))
            np.save(out / f"p_{tag}_argmax.npy", np.array(mx.argmax(logits, axis=-1), dtype=np.int32))
            np.save(out / f"p_{tag}_logits_tail.npy", np.array(logits[-TAIL:], dtype=np.float32))
            return np.array(logits, dtype=np.float32)
        t1 = time.perf_counter()
        logits_cot = teacher(cot_prompt_ids + cot_out_ids, "cot")
        teacher(codes_prompt_ids + codes_out_ids[:P_CODES], "codes")
        teacher(uncond_ids + codes_out_ids[:P_CODES], "uncond")
        timing["planner_teacher_seconds"] = time.perf_counter() - t1

        # ---- lane K: the Python FSM's allowed sets along the oracle's CoT ----
        cp = llm._setup_constrained_processor(use_constrained_decoding=True, constrained_decoding_debug=False, target_duration=None,
                                              user_metadata=cot_call["user_metadata"], stop_at_reasoning=True, skip_genres=True,
                                              skip_caption=False, skip_language=False, generation_phase="cot", is_batch=False)
        steps = []
        n0 = len(cot_prompt_ids)
        for i, tid in enumerate(cot_out_ids):
            scores = torch.from_numpy(logits_cot[n0 - 1 + i].copy())[None]
            state = cp.state.name
            constrained = cp(torch.tensor([cot_prompt_ids + cot_out_ids[:i]]), scores)
            allowed = np.nonzero(np.isfinite(constrained[0].numpy()))[0]
            steps.append(dict(i=i, token=tid, state=state, n_allowed=int(len(allowed)), hash=fnv1a64(allowed),
                              allowed=allowed.tolist() if len(allowed) <= 64 else None, argmax=int(np.argmax(constrained[0].numpy()))))
            assert tid in set(allowed.tolist()), (i, tid, state, len(allowed))
            cp.update_state(tid)
        final_state = cp.state.name   # before the shared processor is reconfigured for the codes phase
        cp2 = llm._setup_constrained_processor(use_constrained_decoding=True, constrained_decoding_debug=False, target_duration=spec["duration"],
                                               user_metadata=None, stop_at_reasoning=False, skip_genres=True, skip_caption=True, skip_language=True,
                                               generation_phase="codes", is_batch=False)
        cp2.state = FSMState.CODES_GENERATION; cp2.codes_count = 0
        codes_mask = cp2.non_audio_code_mask[0].numpy()
        codes_allowed = np.nonzero(np.isfinite(codes_mask))[0]
        json.dump(dict(cot_steps=steps, final_state=final_state, codes_phase=dict(target_codes=cp2.target_codes, n_allowed_with_eos=int(len(codes_allowed)),
                       hash_with_eos=fnv1a64(codes_allowed), hash_without_eos=fnv1a64(codes_allowed[codes_allowed != eos]), eos=eos)),
                  (out / "fsm.json").open("w"), indent=1)

        timing.update(dict(time_costs=tc, phys_peak_bytes=sampler.peak, mx_peak_bytes=mx.get_peak_memory(), checks=checks))
        json.dump(timing, (out / "timing.json").open("w"), indent=1, default=float)
        json.dump(dict(case=name, prompt_id=spec["prompt_id"], caption=p["caption"], lyrics=p["lyrics"], language=p["language"],
                       duration=spec["duration"], seed=spec["seed"], frames=T, n_codes=len(codes), upstream_code="ace-step/ACE-Step-1.5@ca1e85f",
                       weights="ACE-Step/Ace-Step1.5@19671f4"), (out / "request.json").open("w"), ensure_ascii=False, indent=1)
        print(f"[{name}] DONE: T={T}, codes={len(codes)}, enc {tuple(enc.shape)}, checks {json.dumps(checks)}", flush=True)
        model.set_dtype(mx.bfloat16); mx.eval(model.parameters())
    print("ALL DONE", flush=True)


def tok_ids_equal(tokenizer, text, ids, max_length):
    got = tokenizer(text, padding="longest", truncation=True, max_length=max_length, return_tensors="pt").input_ids[0]
    return got.shape == ids.shape and bool((got == ids).all())


if __name__ == "__main__":
    main()
