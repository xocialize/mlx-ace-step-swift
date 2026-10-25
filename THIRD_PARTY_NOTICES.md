# Third-party notices

This package is a Swift-MLX port of **ACE-Step 1.5** (`ace-step/ACE-Step-1.5` @ `ca1e85f`, MIT, Copyright (c) 2026 ACEStep;
model code under `acestep/models/turbo/modeling_acestep_v15_turbo.py`, `acestep/models/mlx/`, `acestep/llm_inference.py`,
`acestep/constrained_logits_processor.py`). Module structure, constants and the sampler follow that code 1:1.

- **ACE-Step/Ace-Step1.5 weights** @ `19671f4` — MIT per the HF card tag + README and the GitHub LICENSE (the HF repo has no
  LICENSE file). Qwen3-Embedding-0.6B and the planner's Qwen3 base: Apache-2.0 (Alibaba Cloud / Qwen).
- **Oobleck VAE** (`diffusers.AutoencoderOobleck`, Apache-2.0 diffusers; weights `ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format`,
  MIT) — Stable Audio Open's architecture retargeted to 48 kHz; the Snake activation is from NVIDIA BigVGAN (MIT). Whether the
  weights were initialised from Stability's community-licensed VAE is stated nowhere upstream.
- **vector-quantize-pytorch** 1.32.0 (MIT, Phil Wang) — the `ResidualFSQ` / `FSQ` maths ported in `AudioTokens.swift`.
- **PyYAML** (MIT) — the emitter's scalar analysis and 80-column folding re-implemented in `PlannerText.swift` so the codes-phase
  prompt is byte-identical to `yaml.dump`.
- **mlx-lm** (MIT, Apple) — the Qwen3 model layout and `make_sampler` (top-p / top-k / categorical) ported in `Qwen3.swift` / `Planner.swift`.
- **HF tokenizers** (Apache-2.0) — the byte-level BPE semantics re-implemented in `BPETokenizer.swift`; GPT-2's `bytes_to_unicode`.
- Read for structure and used as parity references only (AB-D-0153): `arraypress/swift-music-generation` 0.3.0 (MIT) and
  `Blaizzy/mlx-audio` `pc/add-ace`. No code was copied from either.
- `Support/NPY.swift` is lifted from xocialize/mlx-step-audio-editx-swift; `Support/WAV.swift`, `WeightIO.swift` and the KV cache
  from xocialize/mlx-yue2-swift (both MIT, same author).
