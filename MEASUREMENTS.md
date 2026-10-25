# ACE-Step 1.5 port — measurements

Mirror of `mlxengine-audio/WIP/acestep-port/MEASUREMENTS.md` (the live receipt log) at the V2 pass of 2026-10-24.

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

## Verdict

**V1 PASS. V2 PASS on every pre-registered lane against the oracle's Apple-Silicon path, with the one row disclosed above
(VAE vs torch diffusers 109.9 / 107.1 dB against a 110 dB letter; upstream MLX vs diffusers 109.7 / 106.9 dB on the same
latents).** Ticked on AB-T-0352 with this reading. Open for V2b/V3: tiers (bf16 render exists, no gate yet), the planner's
decode speed, `phys_footprint` declaration, reference-audio / cover / repaint inputs (encoder converted, Swift forward unverified).

> **Amendment 2026-10-24, before any Swift result (oracle capture, first pass):** the pipeline's `GenerationParams.shift`
> defaults to **1.0**, so with `inference_steps=8` upstream's `get_timestep_schedule` yields the linspace
> `[1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]` — the shift-3 table in E25's recipe text was wrong (V0 ran at shift 1.0
> too). The first dump solved at shift 3.0 and disagreed with the pipeline's latents by max |Δ| 3.3; the dumper now takes the
> shift from the pipeline parameters and asserts max |Δ| 0 against the pipeline's own `pred_latents`. Lanes S/E/N read the
> schedule from `sampler.json`; the N state is `x = 0.75·noise + 0.25·latents` (a point on the straight path, independent
> of sampler internals), t = 0.75. The gate rules are unchanged. Also noted for E25: the turbo `AudioTokenDetokenizer` has
> `num_attention_pooler_hidden_layers` = **2** encoder layers (checkpoint `detokenizer.layers.{0,1}`), not 24 —
> `num_audio_decoder_hidden_layers` is unused by the turbo class.
