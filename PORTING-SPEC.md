# ACE-Step 1.5: MLX-Swift port spec

Task **AB-T-0352** (E25), decision **AB-D-0153** (full port from upstream), survey receipt AB-R-0604, V0 receipt AB-R-0606.
A full port of `ace-step/ACE-Step-1.5` @ `ca1e85f` (MIT) — the turbo DiT checkpoint with its lyric / timbre encoders and FSQ
detokenizer, the 5 Hz planner (Qwen3-1.7B fine-tune), Qwen3-Embedding-0.6B as the text encoder, and the Oobleck VAE — reading
the weights of `ACE-Step/Ace-Step1.5` @ `19671f4`. Third-party MLX ports (`arraypress/swift-music-generation` 0.3.0, mlx-audio
`pc/add-ace`) were read for structure and serve as parity references only. The scratch side (upstream clone, venv, goldens,
bundles, receipts) lives in `mlxengine-audio/WIP/acestep-port/` + `WIP/acestep-eval/`; `Tools/` rebuilds it.

## 0. Status

| rung | outcome |
|---|---|
| V0 oracle | PASSED 2026-10-24 (AB-R-0606): PER 0.129 vs YuE2 0.384, AES PQ 8.36, 14.5 s per 120 s song on the official MLX path |
| V1 converter | PASSED 2026-10-24: 677 + 310 + 310 + 291 tensors bit-equal to upstream's own conversions; mlx-lm loads the Qwen3 bundles |
| V2 Swift core | PASSED 2026-10-24 (fp32, TF32 off, two goldens): tokenizer 30/30 exact, text 109 dB, conditions 121–140 dB, planner argmax 100 % / ≤ 6e-6, FSM exact at 327 steps, detokenizer 122 dB, DiT velocity 112–115 dB, solve 106–110 dB, VAE 118–121 dB vs the MLX oracle (107–110 dB vs diffusers — disclosed), end-to-end 88–92 dB |
| V2b tiers | DONE 2026-10-24 (`MEASUREMENTS.md` §V2b): four tiers on the 12 V0 songs — bf16 PER 0.197 (**rule B missed by 0.018 — disclosed**; a declared second draw reads 0.120 and the cached generation path is parity-exact: sampling noise, AB-L-0312), planner 8-bit 0.108 (Q8 PASS, offered), planner 4-bit 0.212 (Q4 PASS on the letter only — the operator's listening is the tiebreaker, not offered), 0.6B planner 0.179 (L PASS — the lite tier); AES PQ 8.32–8.35 on all; RTF 4.9–6.2 |
| V3 engine | DONE 2026-10-25 — `MLXACEStep` (`ACEStepPackage`, `ACEStepConfiguration`) on `Capability.music` (mlx-engine-swift 0.71.0, contract 1.56.0): `MusicControls(features: [.anchors, .vocalLanguage], 50 languages, ≤ 600 s)`; offline C1 / C7 / C8 / C-memory / C11 + MAT + CAN tests green (18 tests); `acestep-gates validate` through `MLXServeEngine` (register → prepare → two 120 s songs → six refusals → evict) PASS on bf16 / int8 / lite planner, footprints from `phys_footprint` (§6) |
| V3b | DONE 2026-10-25 (§7; `MEASUREMENTS.md` §V3b): reference-audio timbre, cover / repaint / extend (`sourceAudio`), LRC `timestamps` — each on its own oracle lane: resample 131 / 149 dB; VAE encoder 123.6 dB on `mean ‖ scale` (**the sampled mean 106.8–108.2 dB against a 110 dB letter — disclosed; upstream's own CPU-vs-GPU encoder sits at 109 dB, diffusers at 106–108 dB**); timbre conditions 109.5 dB, latents 105 dB, waveform 93 dB; FSQ tokenize indices 300 / 300 exact; cover 92 / 88 dB, cover 0.5 91 / 88 dB; repaint [10, 20] s **43.7 / 54.6 dB — disclosed (a sensitive sample: the declared [15, 25] s and seed-1005 fixtures read 87 / 80 and 98 / 96 dB)**; extend 106 / 100 and 108 / 120 dB; LRC `lrc_text` string-equal on en and zh. Declared on `MusicControls` (+ `editModes` cover / repaint / extend) and admitted through `MLXServeEngine` on bf16 / int8 / lite with one request of each kind (peaks 17.98 / 16.55 / 15.47 GB — within the V3 declarations); 31 offline tests green |
| V4 publish | DONE 2026-10-25 (`MEASUREMENTS.md` §V4): code at [xocialize/mlx-ace-step-swift](https://github.com/xocialize/mlx-ace-step-swift) tag `v0.1.0`; weights `mlx-community/ACE-Step-1.5-{turbo-bf16, planner-1.7B-bf16, planner-0.6B-bf16, text-encoder-bf16, vae-fp32}` (collection `ace-step-15-swift-mlx-6ade2270e273bf4e0278e14b`), every safetensors sha256-equal to the gated bundle; a fresh download renders (14.1 s for 30 s) and passes `validate` through `MLXServeEngine`. Not published: an `-8bit` planner bundle (the package quantises at load; a pre-quantised bundle needs a loader change) |

| V4b 8-bit default | DONE 2026-10-28 (`MEASUREMENTS.md` §V4b, `v0.2.0`): the planner shipped pre-quantised as `mlx-community/ACE-Step-1.5-planner-1.7B-8bit` (affine g64, the layers' Linears, 2.39 GB) — every tensor bit-equal to the quantise-at-load `.int8` tier (702 / 702), teacher-forced and cached logits identical, the seeded p04 song sample-exact against the V2b p8 take; `Qwen3LM.load` builds `QuantizedLinear`s from the bundle's `quantization` config; `.int8` is the package default and its planner source; validate through `MLXServeEngine` 9.22 GB resident / 16.1–16.9 GB peak (declared 10.0 + 8.0 GB stands); fresh download PASS |

Package: `mlxengine-audio/PROD/ACEStep/mlx-ace-step-swift` (`ACEStepCore`, `MLXACEStep`, `acestep-gates`, `Tools/`).

## 1. Licence posture

Code MIT (this package and upstream). Weights MIT: the GitHub `LICENSE` (Copyright (c) 2026 ACEStep) is the controlling
text — the HF repo ships no LICENSE file, only the card tag + README (footnote 1 in every bundle's NOTICE). Qwen3 bases
Apache-2.0. The Oobleck VAE is MIT as ACE-Step publishes it; its Stability-init provenance is stated nowhere (footnote 2).
Studio-eligible under AB-D-0060 (permissive-only); `weightLicense = "MIT"`, nothing to allowlist. The engine offer (V3) is the
music slot's **default** with `commercial_use: true`, no required notice, `notes` = the two footnotes + planner caption /
language drift + covers need the original's rights (mes-audio-compose-shape, AB-D-0159/0160).

## 2. Architecture as read (`acestep-v15-turbo/config.json`, `modeling_acestep_v15_turbo.py`, `acestep/models/mlx/*`, `llm_inference.py`, `constrained_logits_processor.py`)

**Planner** (`acestep-5Hz-lm-1.7B`, `Qwen3Model` 28 × 2048, GQA 16/8 × 128, SwiGLU 6144, RMSNorm 1e-6, RoPE θ 1e6, vocab
217 204 = Qwen3 151 669 + 65 535 `<|audio_code_N|>` (ids 151 669 + N; only N ≤ 63 999 are valid codes), tied embedding).
Two phases through the Qwen chat template (system `# Instruction\n{LM instruction}\n\n`, user `# Caption\n…\n\n# Lyric\n…\n`):
1. **CoT** (no CFG): the FSM forces `<think>\nbpm: …\ncaption: …\nduration: …\nkeyscale: …\nlanguage: …\ntimesignature: …\n</think>`
   — fixed strings by longest single-token prefix, numeric / keyscale / language values through token-prefix trees built from
   the tokenizer's own encodings of `"bpm: 73"` etc., the caption free but for audio codes and backticks (512-token cap,
   ending detected at a non-indented line), language by top-1 among the 51 codes, user-provided fields injected; with
   `stop_at_reasoning` EOS is forced in place of `</think>`. `parse_lm_output` reads the YAML-ish block.
2. **Codes** (CFG 2.0 vs `NO USER INPUT` + `<think>\n\n</think>\n\n`): the prompt re-emits the metadata as PyYAML
   (`yaml.dump(sort_keys=True, allow_unicode=True)`: plain or single-quoted scalars, 80-column folding with 2-space
   continuations) + `\n\n`; only codes + EOS pass, EOS blocked until `duration × 5` codes then forced. Sampling = mlx-lm
   `make_sampler(temp 0.85, top_p 0.9)`: log-softmax → nucleus → categorical. Codes are **sampled** — reference only.

**Text encoder** `Qwen3-Embedding-0.6B` (28 × 1024, GQA 16/8 × 128, SwiGLU 3072): causal, last hidden states after `norm`
on `SFT_GEN_PROMPT` = `# Instruction\n{instr}\n\n# Caption\n{caption}\n\n# Metas\n- bpm: …\n- timesignature: …\n- keyscale: …\n- duration: N seconds\n<|endoftext|>\n`
(≤ 256 tokens; the tokenizer's TemplateProcessing appends another `<|endoftext|>`); the lyric prompt
`# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>` (≤ 2048) goes through the embedding table only. With planner codes the
instruction is the **cover** one (`Generate audio semantic tokens based on the given conditions:`), the caption the planner's
rewrite (`use_cot_caption`), the language the request's.

**Condition encoder**: `lyric_encoder` (Linear 1024→2048, 8 layers alternating sliding-128 / full bidirectional, norm),
`timbre_encoder` (Linear 64→2048, 4 layers, first position; 750 frames of `silence_latent` when no reference), `text_projector`
(1024→2048, no bias); `pack_sequences` → `[lyrics ‖ timbre ‖ text]` (337–346 tokens on the goldens).

**FSQ / detokenizer**: `ResidualFSQ(dim 2048, levels [8,8,8,5,5,5], 1 quantizer)` — codes → level digits (`idx // basis %
L`) → `digit·2/(L−1) − 1` → `project_out`; `AudioTokenDetokenizer`: Linear 2048→2048, 5 learned patch tokens per code,
**2** encoder layers (`num_attention_pooler_hidden_layers`; the config's `num_audio_decoder_hidden_layers` 24 is unused by the
turbo class), norm, proj_out 2048→64 → 25 Hz hints. `AttentionPooler` + `AceStepAudioTokenizer` (covers) are built and
loaded, gated at V3.

**DiT** `AceStepDiTModel`: `context = [hints ‖ 1]` (128) ‖ noisy latents (64) → Conv1d 192→2048 k2 s2 (12.5 Hz tokens) →
24 layers: adaLN (`scale_shift_table` + the 6-way `time_proj` of `temb_t + temb_r`, `t − r = 0` at inference) around
self-attention (sliding-128 / full alternating, RoPE θ 1e6 rotate-half tables), cross-attention to `condition_embedder(enc)`
with K/V cached per condition, SwiGLU; output adaLN (2-way) → RMSNorm → ConvTranspose1d 2048→64 k2 s2 → crop.

**Sampler** (`mlx_generate_diffusion`): noise `mx.random.normal(key=key(seed))`; schedule `get_timestep_schedule(shift,
None, infer_steps=8)` — the pipeline's `GenerationParams.shift` default is **1.0** → `[1, .875, .75, .625, .5, .375, .25, .125]`
(E25's "shift 3.0 turbo table" was wrong; shift 3 is an option); Euler `x − v·(t − t_next)`, last step `x − v·t`; DCW
(native Haar, mode double: low band `t·0.05`, high band `(1 − t)·0.02`) after every step. No CFG on turbo.

**VAE** `AutoencoderOobleck` (channels 128 × [1,2,4,8,16], strides [2,4,4,6,10] → hop 1920, 48 kHz stereo; weights ship
**bf16**, run fp32): decoder conv1 k7 → 5 blocks (Snake → ConvTranspose k=2s, s, pad ⌈s/2⌉ → 3 residual units k7 d1/3/9)
→ Snake → conv2 k7 (no bias). Chunked decode = `_mlx_decode_single`: 512-frame windows, 64-frame overlap discarded, cache
cleared every 4 windows (the memory knob: 18 GB working set at 512 vs 34 GB at 2048). Output: peak > 1 scaled down, then
`normalize_audio(−1 dBFS)`.

## 3. Rungs and gates

Goldens (`Tools/oracle-capture/dump_goldens.py`, two cases `folk-30s` / `citypop-20s`, fp32 CPU torch pieces + the fp32 MLX
path with `MLX_ENABLE_TF32=0`) and the lane table with its rules are pre-registered in `WIP/acestep-port/MEASUREMENTS.md`
(T0 T C P K F N E D W S). Lanes = `acestep-gates` subcommands; the exact fp32 lane runs on the GPU with `MLX_ENABLE_TF32=0`.
V2b tiers: bf16 everything; 8-bit planner; 4-bit planner — the planner is the quant-sensitive half (acestep.cpp: breaks below
Q5; MiniMax's LM doubled PER at mxfp8), the VAE stays fp32. V3: `MLXACEStep` on `Capability.music` (contract 1.56.0:
`MusicRequest` → `MusicResponse`), footprints from `phys_footprint` (AB-L-0113), C0–C13. V4: xocialize/mlx-ace-step-swift,
mlx-community bundles, the Python rung as `mlx_audio/music/models/ace_step`.

## 4. Not ported

LoRA, Heun / SDE / velocity clamps (options of the sampler, off by default), base / sft 50-step CFG + APG, `cover-nofsq`,
`extract` / `lego` / `complete` (the stems tasks — the SFT-stems checkpoint), `flow_edit_morph`, retake variance, batch > 1,
the `understand` / `inspiration` / `format` planner tasks, the cached-repaint sidecars. Reference-audio timbre, cover / repaint
/ extend and LRC timestamps were "not in V2" and are V3b (§7).

## 5. The engine package (V3 — `MLXACEStep`, mlx-engine-swift 0.71.0 / contract 1.56.0)

`ACEStepPackage` (`@InferenceActor`, `ModelPackage`) wraps `ACEStepPipeline`; the Core has no engine import. One surface,
`ace-step-1.5` on `Capability.music`, born declared: `MusicControls(features: [.anchors, .vocalLanguage], vocalLanguages: the
planner's 50 BCP-47 tags, maxDurationSeconds: 600)` — no `.planning` / `.score` (anchors, not a score), no `.guidanceScale`
(turbo), and **no `.referenceAudio` / `.sourceAudio` / `.timestamps` until their lanes are gated (V3b)**: the engine's
pre-flight refuses those requests before admission, so a supplied clip is never silently dropped (C11).

| canonical field | mapped to | rule |
|---|---|---|
| `caption` | planner + DiT caption (the planner may rewrite it, `rewriteCaption` metaData) | non-empty |
| `lyrics` | the lyric prompt verbatim | empty → `[Instrumental]` |
| `durationSeconds` | planner target → `duration × 5` codes | 10–600 (the engine refuses > 600; the package refuses < 10); nil = the planner's choice |
| `seed` | `key(seed)` noise + the planner's sampler | nil = drawn, reproducible within Swift-MLX only |
| `anchors.bpm / key / timeSignature` | `UserMetadata` injected into the CoT | whole bpm 30–300; one of the 70 keyscales (spelling normalised, bare note = major); 2 / 3 / 4 / 6 over 4 |
| `vocalLanguage` | the DiT lyric prompt's `# Languages` | one of the 50 tags; nil = the planner's `language` line |
| → `audio` | 48 kHz stereo `.wav`, −1 dBFS | `sampleRate` / `channels` set (C2) |
| → `anchors` | the planner's bpm / keyscale / timesignature (as `N/4`), overridden by what was pinned | nil when the plan named none |
| → `truncated` | the codes phase hit its cap before the planner's EOS | honest (C3) |

Configuration (C9): `quant` `.int8` (the default since V4b — the pre-quantised `ACE-Step-1.5-planner-1.7B-8bit` bundle, built as
`QuantizedLinear`s from its `quantization` config before the weights land; a local bf16 planner is still quantised at load) / `.bf16`
(the planner as shipped) / `.fp32` (parity, not published); `planner` `.standard1_7B` / `.lite0_6B`; `vaeChunkFrames` (512);
`bundleDirectory` (+ `plannerDirectory`) as the no-network escape hatch; four `WeightSource` roles (`turbo`, `planner`,
`text-encoder`, `vae` → `mlx-community/ACE-Step-1.5-{turbo-bf16, planner-1.7B-8bit (.int8) | planner-1.7B-bf16 (.bf16 / .fp32) |
planner-0.6B-bf16 (the lite planner, every tier), text-encoder-bf16, vae-fp32}`, each with its file globs) resolved from the engine's store (snapshot or flat layout). Licence (C7 / C8): `.mit` +
`additionalWeightLicenses: [.apache2]`, port `.mit`, three informational notices (no LICENSE file in the HF repo; the VAE's
provenance footnote; the planner's caption / language drift). Progress: `plan` (stage 1 CoT, stage 2 codes, per token) →
`denoise` (8 steps) → `decode` (VAE windows) → `postprocess`; cancellation (CAN-1..3) at every one of those seams, the
`CancellationError` rethrown unchanged. Offline gate: `Tests/MLXACEStepTests` (manifest, MAT per tier, CAN, request plane).
Live gate: `acestep-gates validate` — `MLXServeEngine` register → prepare → run → refusals → evict with `phys_footprint` at
each step; §6 carries the numbers.

## 6. Footprints (V3, measured through `MLXServeEngine`, M5 Max, 2026-10-25, VAE chunk 512 — `MEASUREMENTS.md` §V3)

| tier | phys after prepare | after a 120 s song | peak phys over two songs | declared (`QuantFootprint` / hint) | fits (0.7 × RAM) |
|---|---|---|---|---|---|
| `.bf16` (default) | 10.54 GB | 10.91 GB | 17.87 GB | resident 11.0 GB, activation 8.0 GB | 32 GB Mac |
| `.int8` (**the default since v0.2.0**: the pre-quantised `-8bit` planner; 9.34 GB when a local bf16 planner is quantised at load) | 9.22 GB | 9.63–9.65 GB | 16.12–16.90 GB (V4b) | 10.0 + 8.0 GB | 32 GB Mac (18.0 vs 16.8 on 24 GB) |
| `.bf16` + `planner: .lite0_6B` | 8.15 GB | 8.53 GB | 14.88 GB | `residentBytesHint` 8.7 GB + 8.0 GB | **24 GB Mac** |
| `.fp32` (parity, not offered) | 22.4 GB (V2 lane run) | — | — | 23.0 + 8.0 GB | 48 GB Mac |

The activation is the Oobleck decoder's final 128-channel stage (elementwise temporaries on 1.23 M samples per 640-frame window);
the DiT and the planner add under 1 GB; the V3b encoder (a 30 s window, evals between blocks) peaks 2.2 GB on its own and runs
before the decode, so the V3b requests peak within 0.1 GB of the 120 s songs (`MEASUREMENTS.md` §V3b: 17.98 / 16.55 / 15.47 GB). A smaller window buys little (128 frames: −1.4 GB at 2.2× the decode time); the lever is
a compiled Snake / fused residual unit — a V3 follow-up. No tier fits a 16 GB Mac as measured.

## 7. V3b — reference audio, source audio (cover / repaint / extend), LRC timestamps (2026-10-25; `WIP/acestep-port/MEASUREMENTS.md` §V3b)

Upstream's three input features, read at ca1e85f and ported op for op; each has its own oracle lane (`Tools/oracle-capture/
dump_goldens_v3b.py` → `oracle/goldens-v3b/`, lanes `resample | encoder | timbre | tokenize | cover | repaint | extend | lrc`
of `acestep-gates`, pre-registered before any result — the rules and the readings are in §V3b of the measurements).

| piece | upstream | here |
|---|---|---|
| loader | `_normalize_audio_to_stereo_48k`: mono duplicated, > 2 channels cut, torchaudio `Resample` (sinc_interp_hann, width 6, rolloff 0.99), clamp ±1; `is_silence` (all \|x\| < 1e-6) refused | `AudioLoader.stereo48k` / `isSilence`; `Resampler` builds torchaudio's kernel in Double and runs the strided conv in fp32 (lane R: 131 / 149 dB); `WAV.decode` reads any PCM / float WAV |
| reference clip | `process_reference_audio`: repeat to ≥ 30 s, three 10 s segments at `random.randint` offsets | `AudioLoader.referenceClip` — the offsets from a SplitMix64 seeded by the request seed (declared divergence 1) |
| VAE encode | `encode_and_sample` (`mean + (softplus(scale) + 1e-4)·noise`, unseeded) in 30 s windows with 2 s of overlap discarded (`_mlx_encode_single`) | `AutoencoderOobleck.encode` / `sample` / `encodeChunked`; the noise from `key(seed + 1_000_000 + i)`, `i` counting encode calls in upstream's order — source first, then the reference (declared divergence 2; the oracle hooks the same keyed draw) |
| timbre | the 750-frame reference latents replace the silence slice in the timbre encoder's packed input | `conditions(…, references:)` — the V2 encoder, a different input |
| cover | `DIRECT_CONDITIONING_TASKS`: no planner; `tokenize(target_latents)` (pad to ×5 with silence → `audio_acoustic_proj` → the 2-layer `AttentionPooler` → `ResidualFSQ` indices) → detokenize → `[hints ‖ 1]`; the cover instruction; `audio_cover_strength` s < 1 switches at step `int(8·s)` to the default-instruction conditions and the silence context with a fresh cache | `ACEStepModel.tokenize` / `coverHints` (the V2 FSQ forward, now gated: indices exact on 150 + 150 positions), `CoverSwitch` in `DiffusionSampler.solve` |
| repaint | `chunk_mask_mode = "explicit"` (the Gradio path): 0 / 1 over the range, `src_latents` with the range as silence, `repaint_mask`; `_resolve_repaint_config("balanced", 0.5)` → the first `round(0.5·8) = 4` steps re-inject `t_next·noise + (1 − t_next)·source` outside the range, a 12-frame latent crossfade after the loop, the original source spliced back outside the range after the VAE with a 25 ms ramp (after the handler's peak clamp, before normalisation) | `EditGeometry` / `RepaintMasks` / `RepaintPlan` / `Repaint.inject` + `blend` / `WaveformSplice` — `Edits.swift` and the sampler |
| extend | a repaint whose range lies before the source (`start < 0` pads it left) or past it (`end > D` pads it right) | `EditGeometry.make(.extend)`: `[−L, 0]` → the leading L s, `[D, D + R]` → the trailing R s; both at once = two ranges, per-range ramps (declared divergence 4; the lane gates the single-sided forms upstream can express) |
| LRC | `get_lyric_timestamp`: `x = t·noise + (1 − t)·latents` at `t = 1/8`, `torch.randn` seed 42; the decoder with `output_attentions` returns the cross-attention probabilities (eager softmax, fp32); heads `{2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}` over the lyric rows (after the `# Languages … # Lyric` header, up to `<\|endoftext\|>`); `MusicStampsAligner`: row × column softmax, −2 × row / column medians, ReLU, square, global z-score, median filter 1; `dtw_cpu`; token stamps (a token with no path step inherits the previous end); sentences on a token containing `\n`, decoded jointly; `[mm:ss.xx]text` | `AceStepAttention.crossProbabilities` + `AceStepDiTModel.crossAttentions` (layers 0–6 only, the wanted heads), `LyricAligner` (consensus in fp32 MLX, DTW in Double, Python's `round(x, 3)` and `%05.2f`); the noise from `key(42)` (declared divergence 3; the lane feeds the oracle's) — `lrc_text` string-equal on both goldens |

Engine side (§5 amended): `MusicControls(features: [.anchors, .vocalLanguage, .referenceAudio, .sourceAudio, .timestamps],
editModes: [.cover, .repaint, .extend])`. `referenceAudio` / `edit.source` arrive as `Audio(.wav)` and are decoded by
`WAV.decode`; `edit.operation` maps onto `ACEStepEditKind` (cover strength 0…1; a repaint range inside the source; an extension
with at least one side > 0; the result ≤ 600 s); with an edit `durationSeconds` must be nil or within 1 s of the edit's own
length (refused otherwise — declared divergence 5); `anchors` and `vocalLanguage` still apply (they are the metas and the lyric
header of the direct-conditioned prompt). `MusicResponse.timestamps` carries one entry per lyric line (LRC's own granularity)
with `end` = the line's last token's end; produced on every run. Progress adds `encode` (VAE encode windows) before `denoise`.
