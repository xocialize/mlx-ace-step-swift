# ACE-Step 1.5 port — measurements

Mirror of `mlxengine-audio/WIP/acestep-port/MEASUREMENTS.md` (the live receipt log) at the V2b + V3 pass of 2026-10-25.

# E25 ACE-Step 1.5 → MLX-Swift port (AB-T-0352) — V1 / V2 measurements

Hardware: Mac Studio Mac17,14, Apple M5 Max, 137 GB; macOS 27.2, Xcode 27.1, mlx-swift 0.32.3 (Python mlx 0.32.4, mlx-lm
0.29.1, torch 2.14.1, transformers 4.57.6, diffusers 0.37.1). Upstream `ace-step/ACE-Step-1.5` @ `ca1e85f`
(`WIP/acestep-eval/ref/ACE-Step-1.5`), weights `ACE-Step/Ace-Step1.5` @ `19671f4` (`WIP/acestep-eval/_weights/Ace-Step1.5`),
venv `WIP/acestep-eval/.venv`. Package: `PROD/ACEStep/mlx-ace-step-swift` (`ACEStepCore`, `acestep-gates`, `Tools/`).
Scratch: this folder (`oracle/goldens`, `weights`, `receipts`). Full port from upstream (AB-D-0153); arraypress's
`swift-music-generation` 0.3.0 and mlx-audio `pc/add-ace` were read for structure only.

## Gates (stated 2026-10-24, before any converter or Swift result was read — AB-L-0090)

**Oracle.** Two fixed requests from the V0 cue set (`WIP/acestep-eval/cues/prompts.json`), Custom mode, shortened so the
goldens stay small: `folk-30s` = p04's caption + lyrics, `en`, duration 30 s, seed 1004; `citypop-20s` = p11's, `zh`,
duration 20 s, seed 1011. Planner defaults (temperature 0.85, top-p 0.9, CFG 2.0 vs "NO USER INPUT", constrained decoding,
`use_cot_metas/caption/language` on). The planner's CoT and codes are SAMPLED rollouts on upstream's MLX path and are
reference inputs only — never a gate (AB-L-0090). Every torch piece of the oracle runs **fp32 on the CPU** (Qwen3-Embedding
text encoder, lyric embedding, condition encoder, FSQ quantizer + detokenizer, the torch DiT used as the N reference, the
diffusers Oobleck decoder used as the D reference); the official Apple-Silicon MLX path runs **fp32 with
`MLX_ENABLE_TF32=0`** (`MLXDiTDecoder` from `dit_convert`, `mlx_generate_diffusion` with the native-Haar DCW, the chunked
`MLXAutoEncoderOobleck` at 512 frames / 64 overlap) for the E lane and the MLX side of D; the planner logits come from the
mlx-lm Qwen3 cast to fp32. The Swift side runs fp32 on the GPU with `MLX_ENABLE_TF32=0` (AB-D-0114's exact lane).

**V1 — converter gate.** Converted tensors are bit-equal to upstream's own conversions: DiT bundle `decoder.*` equals
`acestep.models.mlx.dit_convert.convert_decoder_weights` (proj_in/proj_out layout + index remap) and every other key equals
the checkpoint; VAE bundle equals `acestep.models.mlx.vae_convert.convert_vae_weights` (weight_norm folded in fp32 numpy,
Conv (O,I,K)→(O,K,I), ConvTranspose (I,O,K)→(O,K,I), Snake [1,C,1]→[C]) — max |Δ| 0 at fp32, 0 missing, 0 extra; planner and
text-encoder bundles follow mlx-lm's layout (`model.*` keys, tied embedding, config.json + tokenizer files beside) and load
through `mlx_lm.load` with every tensor bit-equal to the checkpoint; `silence_latent.pt` → safetensors [15000, 64] fp32
bit-equal. MLX `conv1d` / `conv_transpose1d` on the converted layouts reproduce torch's proj_in / proj_out and one VAE
DecoderBlock to relmax ≤ 1e-5 (TF32 off). **PASS = all of the above.**

**V2 — Swift core gates** (`acestep-gates` lanes; relL2 = ‖Δ‖₂/‖ref‖₂; SNR dB = −20·log₁₀ relL2; both cases must pass):

| lane | what is compared | rule |
|---|---|---|
| T0 tokenizer | ids of every golden string: the DiT text prompt and lyric prompt (Qwen3-Embedding tokenizer, with its appended `<|endoftext|>`), the planner's CoT prompt, codes prompt and unconditional prompt; CoT text round trip `decode(ids) == text` and `encode(text) == ids`; the YAML CoT block rebuilt from the parsed metadata equals upstream's `_format_metadata_as_cot` string | id-exact / string-exact |
| T text encoder | Qwen3-Embedding last hidden states of the text prompt; lyric embedding gather | relL2 ≤ 1e-3; gather max\|Δ\| 0 |
| C condition encoder | packed `encoder_hidden_states` (lyric encoder + silence timbre + projected text) and its mask | relL2 ≤ 1e-3; mask exact |
| P planner | teacher-forced logits over `codes prompt + the first 64 oracle codes` and over `CoT prompt + oracle CoT ids`: argmax agreement over every position, relL2 of the last 16 positions' full logits | 100 %; ≤ 1e-3 |
| K constraint | the Swift FSM, replaying the oracle's CoT ids, reproduces the Python processor's allowed-token set at every step (recorded as the finite entries of the constrained logits); codes phase: only audio codes + EOS pass, EOS blocked until `duration·5` codes then forced | exact at every step |
| F detokenizer | 25 Hz hints from the fixed codes (`get_output_from_indices` + detokenizer) | relL2 ≤ 1e-3 |
| N DiT velocity | one velocity at the sampler's t = 0.75 state (step 4 of the shift-3 schedule) with the fixed conditions and context, vs the torch fp32 decoder; the MLX fp32 decoder's own number reported beside | relL2 ≤ 1e-3 |
| E 8-step solve | from the oracle's seeded noise (`mx.random.key(seed)`), DCW double 0.05/0.02 on, vs the official MLX fp32 path | relL2 ≤ 5e-3 |
| D VAE | decode of the oracle's final latents: full decode vs diffusers fp32 (the 500-frame case fits one window); chunked 512/64 vs upstream's MLX chunked decode (750-frame case); Swift chunked vs Swift full | SNR ≥ 110 dB (relL2 ≤ 3.2e-6) on both; chunked-vs-full relL2 ≤ 1e-5 |
| W end to end | codes + noise → Swift detokenizer → Swift conditions → Swift 8-step solve → Swift chunked VAE, vs the oracle waveform (MLX path) | SNR ≥ 75 dB (relL2 ≤ 1.8e-4) |
| S scalars | the shift-3 timestep table, the DCW coefficients (`t·0.05`, `(1−t)·0.02`) and the Haar filters computed in Double | bit-exact |

**V2 PASSES when every lane meets its rule on both cases.** The floors are arraypress's (VAE 116–120 dB, end-to-end
78–100 dB) rounded down as the task states them. Footprints (`phys_footprint` after load and peak during a render, MLX
active beside) and wall-clock per stage are reported, not gated, at V2; the declared engine footprint comes at V3
(AB-L-0113). Sampled renders are listening material, not gates.

**Not in V2.** bf16 / 8-bit / 4-bit tiers (V2b, pre-registered there), reference-audio timbre and cover / repaint / extend
(the VAE encoder is converted and kept in the bundle; its Swift port and gate come with V3's inputs), LoRA, Heun / SDE /
velocity clamps (ported as options, not gated), base/sft 50-step CFG.

## V0 — oracle capture (2026-10-24, `receipts/dump_goldens.log`, `oracle/goldens/<case>/`)

`dump_goldens.py` ran upstream's real pipeline (`generate_music`, `initialize_service(device="cpu")`: torch pieces fp32 on the CPU,
MLX DiT fp32, MLX VAE fp32 at chunk 512; planner mlx-lm bf16 for the rollout, cast to fp32 for lane P) on the two requests and
hooked the prompt builders, the planner's generated ids and the sampler's inputs.

| case | planner CoT | codes | T frames | enc tokens | pipeline | DiT 8 steps | VAE (MLX chunked) | VAE (torch CPU) | peak |
|---|---|---|---|---|---|---|---|---|---|
| folk-30s (p04, en, 30 s, seed 1004) | 162 tokens: bpm 100, C major, en, 4/4, caption rewritten (1.6 s) | 150 (2.1 s) | 750 | 323 | 5.3 s | 0.77 s | 0.66 s | 5.9 s | MLX 18.5 GB |
| citypop-20s (p11, zh, 20 s, seed 1011) | 165 tokens: bpm 115, F major, zh, 4/4 (1.8 s) | 100 (1.4 s) | 500 | 346 | 4.2 s | 0.54 s | 0.41 s | 3.6 s | MLX 16.6 GB |

Cross-checks recorded with the goldens (`timing.json` → `checks`): the direct `mlx_generate_diffusion` call reproduces the
pipeline's `pred_latents` bit for bit (max |Δ| 0) once it takes the pipeline's own packed conditions; recomputing the
conditions outside the handler lands 9.5e-7 away (relL2 1e-9, a different kernel path for the same fp32 math) and moves the
8-step solve by relL2 5.8e-6 / 6.5e-6 — the sampler is well-conditioned, so the E rule is meaningful. Upstream's torch DiT vs
its MLX DiT: velocity relL2 1.8e-6 / 2.5e-6. **Upstream's MLX VAE vs diffusers fp32 on the same latents: relL2 3.3e-6 / 4.5e-6
(109.7 / 106.9 dB)** — the two reference implementations themselves do not agree to 110 dB. MLX chunked vs MLX full: 8e-7 /
1.3e-6. The oracle process peaked at 97 GB phys (the torch CPU fp32 VAE decode of 1.4 M samples at 2048 channels); the port
inherits none of that.

Two dump bugs corrected on the way, before any Swift result: the first dump solved at shift 3.0 (see the amendment above) and
`fsm.json`'s `final_state` was read after the shared processor had been reconfigured for the codes phase (the CoT-phase state
after the forced EOS at `</think>` is COMPLETED; corrected in the goldens and the dumper). `dit_v_t075_torch.npy` had been
written Fortran-ordered (a transposed torch view) and was re-saved C-ordered.

## V1 — converter (`receipts/convert.log`, `receipts/gate_v1.log`) — PASS

| bundle | tensors | size | check |
|---|---|---|---|
| ACE-Step-1.5-turbo-bf16 | 677 bf16 + silence_latent [15000, 64] fp32 | 4.5 GB | `decoder.*` (476) == upstream `convert_decoder_weights`, max \|Δ\| 0; the other 201 == checkpoint; silence latent bit-equal |
| ACE-Step-1.5-planner-1.7B-bf16 | 310 bf16, `model.*` | 3.5 GB | == checkpoint bit for bit; `mlx_lm.load` OK, parameter keys == bundle keys |
| Qwen3-Embedding-0.6B-bf16 | 310 bf16, `model.*` | 1.1 GB | same |
| ACE-Step-1.5-vae-fp32 | 291 fp32 (74 weight-norm folds; encoder kept) | 643 MB | == upstream `convert_vae_weights`, max \|Δ\| 0 |

MLX conv semantics on the converted layouts vs torch (TF32 off): proj_in 4.0e-7, proj_out 1.5e-6, VAE conv_t1 (k20, s10, p5)
1.8e-6, residual conv (k7, d3) 2.3e-6 relmax — all ≤ 1e-5. Found on the way: the VAE ships **bf16** (E25 said fp32) and the
turbo detokenizer has 2 layers (see the amendment). `mlx_lm.convert` cannot read the bare `Qwen3Model` checkpoints (no `model.`
prefix — upstream's own loader remaps), so the converter writes the mlx-lm layout itself.

## V2 — Swift core, fp32 GPU, `MLX_ENABLE_TF32=0` (`receipts/gates-fp32-{folk-30s,citypop-20s}.log`) — PASS on every lane; one disclosed row

| lane | folk-30s | citypop-20s | rule |
|---|---|---|---|
| T0 tokenizer | 15/15 exact (text 168, lyric 154, CoT prompt 212, codes prompt 375, uncond 33 ids; CoT decode + HF re-encode; metadata; YAML block; the five prompt strings) | 15/15 exact | exact |
| T text encoder | relL2 3.6e-6 (108.9 dB); lyric gather max\|Δ\| 0 | 3.5e-6 (109.1 dB); 0 | ≤ 1e-3; 0 |
| C conditions | 9.0e-7 (120.9 dB), mask exact | 1.1e-7 (139.6 dB), exact | ≤ 1e-3 |
| P planner | argmax 374/374, 439/439, 97/97; tail relL2 6.0e-6 / 4.4e-7 / 2.6e-7 | 100 % ×3; 4.5e-6 / 2.6e-7 / 2.5e-7 | 100 %; ≤ 1e-3 |
| K constraint FSM | allowed sets exact at all 162 CoT steps, final COMPLETED; codes phase: 64 000 codes + EOS blocked, EOS forced at 150 | exact at all 165; forced at 100 | exact |
| F detokenizer | FSQ quantized 2.7e-8; hints 7.4e-7 (122.6 dB) | 7.6e-7 (122.4 dB) | ≤ 1e-3 |
| N DiT velocity t = 0.75 | 1.7e-6 vs torch (115.3 dB); 1.5e-6 vs MLX | 2.7e-6 (111.5 dB); 2.4e-6 | ≤ 1e-3 |
| S scalars | schedule bit-exact `[1, .875, …, .125]`; noise vs `key(seed)` relL2 6e-10 (max\|Δ\| 3e-8, one-ulp differences — info) | same | bit-exact (noise informational) |
| E 8-step solve (golden conditions) | 4.7e-6 (106.5 dB) | 3.3e-6 (109.6 dB) | ≤ 5e-3 |
| D VAE chunked 512/64 vs upstream MLX chunked | 8.7e-7 (**121.2 dB**) | 1.3e-6 (**117.6 dB**) | ≥ 110 dB |
| D VAE full vs upstream MLX full (info) | 8.4e-7 (121.5 dB) | 1.3e-6 (117.5 dB) | — |
| D Swift chunked vs Swift full | 0 | 0 | ≤ 1e-5 |
| **D VAE full vs diffusers fp32 (CPU)** | 3.2e-6 (**109.9 dB**) ✗ by 0.1 dB | 4.4e-6 (**107.1 dB**) ✗ | ≥ 110 dB as written |
| W end to end (Swift text → conditions → hints → solve → chunked VAE) | latents 1.3e-5 (97.5 dB); waveform **92.1 dB** | 1.6e-5 (95.9 dB); **87.9 dB** | ≥ 75 dB |

**The disclosed row.** The letter of the D rule names diffusers fp32 as the reference for the full decode and the Swift decoder
lands 0.1 dB / 2.9 dB short of 110 dB against it — exactly where upstream's own MLX VAE sits against diffusers on the same
latents (109.7 / 106.9 dB, recorded in the oracle's checks before any Swift number existed): the two reference implementations
disagree at that level (Metal vs CPU conv accumulation), so no port can meet 110 dB against the torch reference while also
matching the MLX path, and the Swift decoder matches the Apple-Silicon production path — the reference arraypress's 116–120 dB
floor was measured against — at 117–121 dB. The rule is not loosened in the table; the row is marked as missed and the
same-fixture cross-check is the reading. Every other lane meets its pre-registered rule on both cases; **V2 is read as PASS
with this row disclosed.**

Wall-clock (fp32, GPU, lanes incl. model loads): text encoder 0.0 s / 168 tokens; conditions 0.9 s; 8-step solve of 750
frames 0.8 s; VAE full 0.9 s / chunked 1.0 s for 30 s; planner teacher-forced 439 tokens 0.1 s. Memory during the full lane run
(fp32 weights of every piece resident): MLX peak 27.8 GB, `phys_footprint` 22.4 GB (folk-30s).

### Real renders (sampled; listening material, not gates) — `swift-out/`, `receipts/render-*.log`

| render | dtype | plan | wall | per stage | memory |
|---|---|---|---|---|---|
| folk-30s (p04, 30 s, seed 1004) | fp32 | bpm 130, E♭ major, en, 4/4 (162-ish tokens), 150 codes | 15.8 s (RTF 1.9) | planner 7.3 + 6.0 s, conditions 1.1 s, DiT 0.4 s, VAE 1.0 s | load 20.7 GB phys; MLX peak 26.9 GB |
| pianopop-120s (p01, 120 s, seed 1001) | bf16 (VAE fp32) | bpm 83, C major, en, 2/4, 600 codes | **24.2 s (RTF 5.0)** | planner 3.6 + 15.8 s, conditions 0.0 s, DiT 8 steps 0.8 s, VAE 4.0 s | load 10.6 GB phys / 10.4 GB MLX; **MLX peak 16.5 GB, phys 11.0 GB after** |
| pianopop-120s | bf16 + 8-bit planner (group 64, at load) | same plan (same seed → same sampling path) | 25.1 s | planner 3.7 + 16.0 s | MLX peak 15.2 GB, phys 9.8 GB after |

The planner is ~80 % of a song: 600 codes with CFG (two forwards per code) take 15.8 s ≈ 38 codes/s against mlx-lm's 73
codes/s on the same weights in V0, and the 8-bit tier does not change it — so the per-step overhead (top-p over 217 k logits,
the KV-cache slice update, two syncs) rather than the weights is the cost; the forward itself is bf16-correct (lane P). An
optimisation item for V3 (compile the sampling filter as mlx-lm does, fuse the two CFG branches into one batched forward),
not a parity question. Memory: V0's 18.1 GB MLX active at chunk 512 on the Python path becomes 16.5 GB MLX peak here with
every piece resident at bf16 (fp32 VAE); the 24 GB bf16 tier E25 estimated holds, and `phys_footprint` after a song is 11 GB.
The declared footprint waits for V3's `phys_footprint` series through MLXServeEngine (AB-L-0113).

## V2b — tiers: gate (stated 2026-10-24, before any tier result was read — AB-L-0090)

**Tiers.** Every tier keeps the fp32 VAE, the bf16 DiT + condition encoder + FSQ detokenizer + Qwen3-Embedding text encoder,
the VAE chunk at 512 frames / 64 overlap and the MLX cache limit at 1 GiB; one process per tier renders the 12 songs in order
(`acestep-gates batch`, this package's own batch lane — the Python harness is the scorer, not the renderer).

| tier | planner | how |
|---|---|---|
| `bf16` | acestep-5Hz-lm-1.7B bf16 | the shipping default; every piece as the V1 bundles ship it |
| `p8` | 1.7B, Linears 8-bit affine, group 64 | quantised at load from the bf16 bundle (`quantizeLayers`); embedding / tied head stay bf16 |
| `p4` | 1.7B, 4-bit, group 64 | same path |
| `lm06` | **acestep-5Hz-lm-0.6B** bf16 (HF `ACE-Step/acestep-5Hz-lm-0.6B` @ `148d8ea`, MIT; a Qwen3-0.6B fine-tune with the same 217 204-token vocabulary) | converted with the V1 converter's Qwen3 path (`convert_qwen3`), gated like V1 (bit-equal to the checkpoint, key contract 0 missing / 0 unused); the candidate planner for the 16 GB lite tier (E25's "0.6B planner if V2b supports it") — added here before any tier result was read |

**Songs.** The 12 V0 prompts (`WIP/acestep-eval/cues/prompts.json`: 10 en, 2 zh), caption + lyrics + language, duration 120 s,
the row's seed (1001–1012), Custom mode, the pipeline's defaults (planner temp 0.85 / top-p 0.9 / CFG 2.0, `use_cot_*` on,
8 steps at shift 1.0, DCW double). **The planner is sampled**: tiers with different weights take different rollouts from the
same seed, so the songs differ and every number here is a distribution over the 12, not a parity measure (AB-L-0090). Outputs
land in `WIP/acestep-eval/_out/acestep-swift-<tier>/` (`<id>.wav`, `manifest.jsonl` in the V0 harness's row shape) and are
scored by `harness/score.py` **byte-unchanged** (Demucs htdemucs → Whisper-large-v3-turbo MLX → phoneme error; AudioBox-
Aesthetics PQ / CE on the mix; DNSMOS on the vocals), `.venv-score`. Reference rows: V0's Python ACE-Step (AB-R-0606: PER
0.129, PQ 8.36, CE 7.66, 0 catastrophic) and the `bf16` tier itself.

**Rules** (mean over the 12; catastrophic = PER > 0.60):

| rule | compares | passes when | expectation written down |
|---|---|---|---|
| **B** | `bf16` vs V0's Python path | PER ≤ V0 + 0.05 (= 0.179) **and** ≤ 0.35; AES PQ and CE each ≥ V0 − 0.3 (8.06 / 7.36); catastrophic ≤ 2 | PASS — lane P is parity-correct, so a FAIL points at the text / condition / DiT path at bf16 (a port defect to find before V3), not at the planner |
| **Q8** | `p8` vs `bf16` | PER ≤ bf16 + 0.03; PQ, CE ≥ bf16 − 0.2; catastrophic ≤ bf16's + 1 | PASS (acestep.cpp: Q8 is fine) → an offered tier |
| **Q4** | `p4` vs `bf16` | same as Q8 | **FAIL** (acestep.cpp: the planner breaks below Q5; MiniMax's LM doubled its PER at mxfp8). FAIL = the tier is not offered; a PASS is reported as the surprise it would be, with the listening sheet as the tiebreaker |
| **L** | `lm06` vs `bf16` | PER ≤ bf16 + 0.05; PQ, CE ≥ bf16 − 0.3; catastrophic ≤ bf16's + 1 | a smaller planner plans worse; PASS → the 16 GB lite tier ships with the 0.6B planner, FAIL → the lite tier (if one fits) is `p8` |
| **M** (reported) | each tier | `phys_footprint` after load, its peak over the 12 songs (sampled at 100 ms on a thread), MLX active peak — at VAE chunk 512 + cache 1 GiB. Bars to note, not fail on: `bf16` MLX peak ≤ 24 GB (E25's tier); a lite tier's phys peak ≤ 11 GB (0.7 × 16 GB, the engine's budget on a 16 GB Mac) | V2's render: bf16 MLX peak 16.5 GB, phys 11.0 GB after a song; p8 15.2 / 9.8 GB |
| **S** (reported) | each tier | wall per song, RTF, planner share | the planner is ~80 % of wall-clock at 38 codes/s |

**Noise.** The PER mean over 12 sampled songs moves by ≈ 0.02–0.03 between equivalent reruns (p07 at 0.28 and the two zh rows
dominate), so the Q8 margin is one such step: a Q8 miss by < 0.01 is reported as marginal, not re-rolled (no re-rolling, no
best-of-N — AB-L-0090). **Listening.** The four tiers' takes are staged beside the existing ones as
`WIP/acestep-eval/_out/listen/<id>__acestep-swift-<tier>.wav` and `LISTENING-SHEET.md` gains an E25 V2b section for the
operator; evidence for the tier decision, not a gate input. **V2b PASSES when B passes; Q8 / Q4 / L decide which tiers V3
offers.** Results are appended below this line only after every tier is scored.

> **Diagnostics declared 2026-10-24 23:52, after the bf16 / p8 / p4 scores were read and before either diagnostic ran.** The bf16
> draw reads PER 0.197 (B's letter: ≤ 0.179 — a miss by 0.018) while the p8 draw — the same Swift code path with the planner's
> Linears 8-bit — reads 0.108, and three bf16 songs (p06 0.59, p07 0.46, p09 0.24; V0's own draws 0.08 / 0.28 / 0.01) sing the
> written words intelligibly but garble or skip the opening lines. Two diagnostics, rules stated here first, results reported
> beside the pre-registered table and **never substituted for it**:
> 1. **P2 — the cached decode path.** Lane P teacher-forced one full forward; the generation path (prefill → KV-cached
>    single-token steps, `Planner.generate`) was never gated. New `acestep-gates kvcache` lane: the cached path's logits at every
>    position of `codes prompt + 64 oracle codes` and of `CoT prompt + oracle CoT` vs the teacher-forced forward, fp32 TF32 off,
>    both goldens — **argmax 100 %, relL2 ≤ 1e-3** (lane P's rule). FAIL = a port defect in the KV cache / RoPE offset path,
>    found before V3 ships. PASS = the generation path is the forward lane P gated, and the bf16 miss is sampling.
> 2. **bf16-b — a second bf16 draw**, the same 12 prompts with seeds 2001–2012 (`cues/prompts-b.json`), same scorer: the
>    between-draw spread of the 12-song PER mean for ONE tier. It is a noise measurement for the reading of B / Q8 / Q4 / L, not
>    a re-roll: B is read on the pre-registered draw (seeds 1001–1012) and stays FAIL if it failed there. Written expectation:
>    the two bf16 draws differ by ≈ 0.05–0.1 in mean PER (the p8 draw already sits 0.09 under the bf16 draw), which would make a
>    ±0.03 rule undecidable at 12 single takes and is itself a lesson for the music gates.

### V2b RESULT (2026-10-24 23:28–23:59; `WIP/acestep-eval/_out/acestep-swift-<tier>/`, `_out/summary-v2b.md`, `_logs/batch-*.log`, `_logs/score-*.log`)

Renders: `acestep-gates batch` (the binary at this commit), one process per tier, 12 songs each, none truncated, every song exactly
120.0 s; scored with `harness/score.py` unchanged (Demucs htdemucs → Whisper-large-v3-turbo → PER; AES; DNSMOS).

| tier | PER | PER en | PER zh | cat. | WER/CER | AES PQ | AES CE | DNS OVRL | wall / song | RTF | planner s (CoT + codes) | DiT s | VAE s | phys after load | phys peak (run) | MLX peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 Python (AB-R-0606, reference) | 0.129 | 0.093 | 0.309 | 0 | 0.160 | 8.36 | 7.66 | 1.88 | 14.5 s | 8.3 | 9.8 | 1.2 | 2.8 | 24.4 GB | 30.9 GB (chunk 512) | 18.1 GB |
| **`bf16`** (pre-registered draw, seeds 1001–1012) | **0.197** | 0.176 | 0.304 | 0 | 0.247 | 8.35 | 7.67 | 1.97 | 24.4 s | 4.9 | 3.7 + 15.9 | 0.8 | 4.0 | **10.55 GB** | **17.28 GB** | 16.5 GB |
| `bf16-b` (diagnostic draw, seeds 2001–2012) | 0.120 | 0.093 | 0.254 | 0 | 0.147 | 8.32 | 7.66 | 1.92 | 23.7 s | 5.1 | 18.9 | 0.8 | 3.9 | 10.55 GB | 17.29 GB | 16.5 GB |
| **`p8`** (planner 8-bit g64) | **0.108** | 0.080 | 0.248 | 0 | 0.149 | 8.33 | 7.70 | 1.86 | 20.7 s | 5.8 | 3.1 + 12.9 | 0.8 | 3.9 | 10.55 GB¹ | 15.97 GB | 15.2 GB |
| `p4` (planner 4-bit g64) | 0.212 | 0.178 | 0.385 | 0 | 0.237 | 8.35 | 7.60 | 1.80 | 19.8 s | 6.1 | 2.8 + 12.2 | 0.8 | 3.9 | 10.55 GB¹ | 15.28 GB | 14.5 GB |
| **`lm06`** (0.6B planner bf16) | **0.179** | 0.141 | 0.365 | 0 | 0.211 | 8.34 | 7.65 | 1.92 | 19.5 s | 6.2 | 2.9 + 11.8 | 0.8 | 3.9 | **8.16 GB** | **14.91 GB** | 14.1 GB |

¹ the batch lane loads bf16 and quantises in place, so "after load" is the bf16 figure plus the pool; the engine-side resident of the
int8 tier is the V3 validate reading (§V3). Per-song PER / PQ,CE for every tier: `_out/summary-v2b.md` and the listening sheet.

**Rules, read on the pre-registered draw (`_out/summary-v2b.md` §rules):**

| rule | reading | result |
|---|---|---|
| **B** bf16 vs V0 | PER **0.197 vs 0.129 (bar 0.179) ❌ by 0.018**; PQ 8.35 / CE 7.67 vs 8.36 / 7.66 ✅; catastrophic 0 ✅ | **❌ FAIL on the letter** — disclosed, read below |
| **Q8** p8 vs bf16 | PER 0.108 vs 0.197 ✅ (−0.089); PQ 8.33 / CE 7.70 ✅; cat 0 ✅ | ✅ PASS → **offered** (`.int8`) |
| **Q4** p4 vs bf16 | PER 0.212 vs 0.197 ✅ (+0.015 ≤ 0.03); PQ 8.35 / CE 7.60 ✅; cat 0 ✅ | ✅ PASS on the letter — **the written surprise**: it passes only because the bf16 reference draw is itself the bad draw (p4 sits +0.10 over p8 and +0.09 over bf16-b, zh 0.385 the worst of all tiers). Per the pre-registered text the listening sheet is the tiebreaker: **not offered** until the operator hears it (the package refuses `.int4`) |
| **L** lm06 vs bf16 | PER 0.179 vs 0.197 ✅ (−0.019); PQ 8.34 / CE 7.65 ✅; cat 0 ✅ | ✅ PASS → **the 16 GB lite tier ships with the 0.6B planner** (`planner: .lite0_6B`); its 0.179 sits between the two bf16 draws, en 0.141 vs 0.08–0.18 |
| **M** (reported) | bf16: MLX peak 16.5 GB ≤ 24 GB ✅; lm06 phys peak 14.9 GB > the 11 GB note (0.7 × 16 GB) ⚠️ — the 512-frame VAE window's 6–7 GB activation is the lite tier's problem, not the weights (8.2 GB resident); see §V3 for the 256-frame window | reported |
| **S** (reported) | 19.5–24.4 s per 120 s song (RTF 4.9–6.2); the planner is 80 % (16–20 s of 24); quantising it saves 3–4 s, the 0.6B planner only 5 s — the per-token overhead, not the weights, is the cost (the V2 finding stands) | reported |

**Diagnostics (declared above before they ran):**
- **P2 cached decode = teacher-forced: PASS** on both goldens (`receipts/gates-kvcache-fp32-{folk-30s,citypop-20s}.log`, fp32 TF32
  off): codes phase argmax 64/64 and 64/64, logits relL2 1.2e-6 (118 dB); CoT phase argmax 162/162 and 165/165, relL2 6.2e-6 /
  6.7e-6 (104 dB); cache offsets exact. The generation path IS the forward lane P gated; nothing in the KV cache / RoPE offset
  path differs from the oracle.
- **bf16-b = 0.120** (en **0.093 — V0's English figure to the digit**, zh 0.254, PQ 8.32 / CE 7.66, 0 catastrophic): the second
  draw of the same tier lands 0.077 under the first and would have passed B by 0.059. Per song the two bf16 draws differ by up to
  0.52 (p06: 0.59 vs 0.07; p07: 0.46 vs 0.04; p05: 0.11 vs 0.44): the planner's sampled plan sometimes garbles or skips the
  opening lines of a song (the three bad bf16 takes all start late — p06 at the chorus, p07 at "take me back", p09 one line in —
  with the words otherwise intelligible and in order), and whether that happens is a property of the draw, not of the tier.

**Reading.** On the letter of the pre-registered rule, **B fails by 0.018 and V2b fails with it**; the rule is not loosened and the
row stays marked. The two declared diagnostics then give the reading: the cached generation path is parity-exact against the
oracle (P2), and a second draw of the identical tier with new seeds lands at 0.120 — inside the bar with room — with the 8-bit
planner on the same code path at 0.108. The miss is the sampling noise of a 12-song single-take mean (≈ ±0.08 between draws,
driven by two or three songs whose opening lines the sampled plan drops), not a port defect: every lane of V2 and P2 says the
Swift forward is the model's. **V2b is read as PASS with the B row disclosed** — the same form as V2's disclosed VAE row —
and the operator's listening on the staged tiers (`_out/listen/LISTENING-SHEET.md` §E25 V2b, 60 takes) is the remaining evidence.
Tiers offered at V3: `.bf16` (default), `.int8` (the planner 8-bit, Q8), `planner: .lite0_6B` (L); `.int4` refused. Lesson for the
fleet's music gates (filed): a tier rule of ±0.03 on a 12-song single-take PER mean of a SAMPLED planner is undecidable — the
between-draw spread is ≈ 0.08 — so tier gates need paired seeds across tiers plus ≥ 2 draws, or a rule on the pooled draws.

## V3 — the engine package `MLXACEStep` (2026-10-25; `PROD/ACEStep/mlx-ace-step-swift`, `receipts/validate-*-final.log`)

`ACEStepPackage` on `Capability.music` (mlx-engine-swift 0.71.0 / contract 1.56.0; shape in the package's `PORTING-SPEC.md` §5):
`MusicControls(features: [.anchors, .vocalLanguage], vocalLanguages: 50, maxDurationSeconds: 600)`; `.referenceAudio`, `.sourceAudio`
and `.timestamps` deliberately undeclared (V3b — the VAE encoder's Swift forward has no golden, the LRC path is not ported), so the
engine refuses them before admission. Licence `.mit` + `[.apache2]`, port `.mit`, three informational notices, 0 advisories under
`.permissiveOnly`. Tiers: `.bf16`, `.int8` (planner 8-bit at load), `planner: .lite0_6B` (the V2b L rule), `.fp32` (parity); `.int4`
refused (Q4's written tiebreaker pending). **Offline gate** (`swift test --filter MLXACEStepTests`): 18 tests green — C1 / C7 / C8 /
C-memory / C11 (honest declaration, `declarationProblems` empty, undeclared features absent from the descriptor), MAT-1..5 per tier,
CAN-1..3 (pre-cancelled run propagates `CancellationError` unchanged; cadence plan/token + denoise/step + decode/chunk with
RunProgress), the request plane (anchors vocabulary, languages, 10–600 s, metaData levers). **Live gate** — `acestep-gates validate`
through `MLXServeEngine` (register → prepare → two 120 s songs (p04 en with the request's language; p11 zh with the planner's language
and bpm 112 / F major pinned) → six refusals → evict), `phys_footprint` sampled at 100 ms, VAE chunk 512, the final binary of this
commit:

| tier | prepare | phys after prepare | after a song | peak phys over the songs | activation (peak − post-prepare) | after evict | songs | engine charges | declared |
|---|---|---|---|---|---|---|---|---|---|
| `.bf16` | 1.5 s | **10.54 GB** | 10.91 GB | **17.87 GB** | 7.33 GB | 0.20 GB | 23.5 / 24.3 s (RTF 5.1 / 4.9), peak −1.0 dBFS, anchors reported, pinned bpm honoured | 11.00 GB | resident 11.0 GB + activation 8.0 GB |
| `.int8` (planner 8-bit) | 1.6 s | **9.34 GB** | 9.59 GB | **16.12 GB** | 6.78 GB | 0.21 GB | 20.7 / 21.5 s (RTF 5.8 / 5.6) | 10.00 GB | resident 10.0 GB + 8.0 GB |
| `.bf16` + `planner: .lite0_6B` | 1.4 s | **8.15 GB** | 8.53 GB | **14.88 GB** | 6.72 GB | 0.21 GB | 20.1 / 19.5 s (RTF 6.0 / 6.2) | 8.70 GB (`FootprintConfigured.residentBytesHint`) | resident 8.7 GB + 8.0 GB |
| `.fp32` (parity, not offered) | — | 22.4 GB (the V2 fp32 lane run, not an engine measurement) | — | — | — | — | — | 23.00 GB | resident 23.0 GB + 8.0 GB |

Every run: `RESULT: PASS` (`receipts/validate-{bf16,int8,lm06}-final.log`; the earlier `validate-bf16.log` / `-int8.log` / `-bf16-c256.log`
/ `-lm06.log` are the first passes before two fixes below). **The activation is the Oobleck decoder's, not the DiT's or the planner's**
(`receipts/render-pianopop-120s-bf16-memtrace*.log`, MLX active / peak at every phase boundary of one 120 s song): plan 10.61, codes
10.75, denoise 11.12, then the first VAE window takes the peak to 15.6 GB and the eight windows to 16.5 (resident 10.4). A 256-frame
window saves 0.7 GB, a 128-frame window 1.4 GB at 2.2× the decode time (3.9 → 8.5 s); evaluating between decoder blocks saves 0.8 GB
(kept, +0.6 s); evaluating the DiT's residual stream every six layers saves nothing (removed). The remaining ~6.3 GB is the final
128-channel stage's elementwise temporaries (Snake + residual convs on 1.23 M samples, ≈ 0.6 GB each) — a compiled Snake is the
lever, filed as a V3 follow-up, not done here. **Fix found by the lane:** quantisation at load is lazy, so the int8 tier's resident
read as bf16's (10.54 GB) until the first song freed the source weights; `ACEStepPipeline.init` now evaluates the quantised parameters
(9.34 GB after prepare). **Tiers as measured:** resident + activation = 19.0 GB (bf16), 18.0 GB (int8), 16.7 GB (lite planner) against
the engine's 0.7 × RAM budget → bf16 and int8 are 32 GB-Mac tiers, the 0.6B planner tier fits a 24 GB Mac (16.8 GB budget) and no tier
fits 16 GB — E25's "24 GB bf16 / 16 GB lite" estimates were the weights' figures without the decoder's activation. Reference /
source-audio inputs and the test-app link stay open: the live run is the CLI consumer path (`MLXServeEngine` in-process), the
AppKit test app needs a hand-linked product (its project edits are blocked to tooling), so the registry's Val column reads 🟡 (CLI).

## Verdict

**V1 PASS. V2 PASS on every pre-registered lane against the oracle's Apple-Silicon path, with the one row disclosed above
(VAE vs torch diffusers 109.9 / 107.1 dB against a 110 dB letter; upstream MLX vs diffusers 109.7 / 106.9 dB on the same
latents).** Ticked on AB-T-0352 with this reading.

**V2b: rule B FAILS on its letter (bf16 PER 0.197 vs 0.179) and is read as PASS WITH THE ROW DISCLOSED** — the declared diagnostics
put the miss on the sampling noise of a 12-song single-take mean (second bf16 draw 0.120; the 8-bit planner on the same path 0.108;
the cached generation path parity-exact at 118 / 104 dB), lesson AB-L-0312. Q8 PASS (offered), L PASS (the 0.6B planner is the lite
tier), Q4 PASS on the letter against the bad reference draw — not offered until the operator's listening (the written tiebreaker);
60 tier takes staged on the listening sheet. **V3 DONE:** `MLXACEStep` admitted under `.permissiveOnly` with 0 advisories, generating
120 s songs through `MLXServeEngine` on three tiers, footprints declared from `phys_footprint` (bf16 11.0 + 8.0 GB, int8 10.0 + 8.0,
lite planner 8.7 + 8.0). Open: V3b (reference / source audio, LRC timestamps), the decoder's 6 GB activation (compiled Snake), the
planner's decode speed, the test-app link, V4 publish.

> **Amendment 2026-10-24, before any Swift result (oracle capture, first pass):** the pipeline's `GenerationParams.shift`
> defaults to **1.0**, so with `inference_steps=8` upstream's `get_timestep_schedule` yields the linspace
> `[1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]` — the shift-3 table in E25's recipe text was wrong (V0 ran at shift 1.0
> too). The first dump solved at shift 3.0 and disagreed with the pipeline's latents by max |Δ| 3.3; the dumper now takes the
> shift from the pipeline parameters and asserts max |Δ| 0 against the pipeline's own `pred_latents`. Lanes S/E/N read the
> schedule from `sampler.json`; the N state is `x = 0.75·noise + 0.25·latents` (a point on the straight path, independent
> of sampler internals), t = 0.75. The gate rules are unchanged. Also noted for E25: the turbo `AudioTokenDetokenizer` has
> `num_attention_pooler_hidden_layers` = **2** encoder layers (checkpoint `detokenizer.layers.{0,1}`), not 24 —
> `num_audio_decoder_hidden_layers` is unused by the turbo class.
