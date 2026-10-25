# mlx-ace-step-swift

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (ACE Studio + StepFun) — caption + lyrics → a 48 kHz stereo song —
ported to Swift-MLX for Apple Silicon: the 5 Hz planner LM with its constrained YAML reasoning, Qwen3-Embedding text encoder,
the lyric / timbre condition encoder, the FSQ detokenizer, the 2 B flow-matching DiT (8 Euler steps + DCW) and the Oobleck VAE
decoded in 512-frame chunks. A full port from the released code (`ca1e85f`), gated lane by lane against an fp32 oracle; the
third-party MLX ports were read for structure only.

**Licence.** This code is MIT. The weights (`ACE-Step/Ace-Step1.5`) are MIT per the GitHub LICENSE and the HF card (the HF
repo has no LICENSE file); the Qwen3 bases are Apache-2.0; the Oobleck VAE's Stability-init provenance is unstated. Both
footnotes ride in every bundle's `NOTICE`. Commercial use is allowed.

## What it does

```swift
import ACEStepCore
let pipe = try ACEStepPipeline(bundles: bundlesDir, dtype: .bfloat16)         // fp32 for the exact lane; plannerBits: 8 for the lite tier
var req = ACEStepRequest(caption: "Indie folk, acoustic guitar, warm female vocal", lyrics: "[Verse]\n…\n[Chorus]\n…", language: "en", duration: 60, seed: 7)
let song = try pipe.generate(req) { phase, fraction in /* plan / codes / denoise / decode */ }
try WAV.write16(song.audio, channels: 2, sampleRate: 48000, to: out)           // song.metadata = bpm / key / caption the planner chose
```

Custom mode as upstream's `generate_music(thinking=True)`: the planner fills bpm / key / time signature / language and rewrites
the caption, writes `duration × 5` codes; the DiT renders them with the cover instruction; the VAE decodes; peak clamp and
−1 dBFS. Defaults are the pipeline's (`shift` 1.0, 8 steps, DCW double 0.05/0.02, LM 0.85 / top-p 0.9 / CFG 2.0).

## Bundles

`Tools/convert/convert.py` writes `ACE-Step-1.5-turbo-bf16/`, `ACE-Step-1.5-planner-1.7B-bf16/`, `Qwen3-Embedding-0.6B-bf16/`
(both in mlx-lm's layout, loadable with `mlx_lm.load`) and `ACE-Step-1.5-vae-fp32/` (weight norm folded). Weights are cast
to the requested dtype at load; the planner / DiT can be quantised at load (`plannerBits`, `ditBits`).

## Parity

`acestep-gates` runs the lanes of `PORTING-SPEC.md` §3 against the oracle goldens (`Tools/oracle-capture`): tokenizer
(id-exact), text encoder, condition encoder, planner (teacher-forced logits), the constraint FSM (allowed sets at every step),
detokenizer, one DiT velocity, the 8-step solve, VAE full + chunked, end to end. Numbers: `MEASUREMENTS.md`.

Build with the Xcode build system so MLX's metallib ships: `swift build -c release --build-system swiftbuild`. The exact fp32
lane needs `MLX_ENABLE_TF32=0`.

## Status

`PORTING-SPEC.md` §0 and `MEASUREMENTS.md`. The engine package (`MLXACEStep`, `Capability.music`) is V3.
