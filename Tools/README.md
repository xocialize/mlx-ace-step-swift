# Tools

The Python side of the port. Nothing here assumes a directory layout: every location is an argument. Run with the acestep venv
(`WIP/acestep-eval/.venv`: torch 2.14.1, mlx 0.32.4, mlx-lm 0.29.1, transformers 4.57.6, diffusers 0.37.1, vector-quantize-pytorch
1.32.0) and `python -I`; the oracle imports upstream from the `ace-step/ACE-Step-1.5` clone at `ca1e85f`.

| path | what |
|---|---|
| `convert/convert.py` | **V1 converter.** `ACE-Step/Ace-Step1.5` → four MLX-layout bundles: `ACE-Step-1.5-turbo-bf16` (keys verbatim but `decoder.proj_in.1`/`proj_out.1` → `proj_in`/`proj_out` with MLX conv layouts; `silence_latent.safetensors`), `ACE-Step-1.5-planner-1.7B-bf16` and `Qwen3-Embedding-0.6B-bf16` (mlx-lm layout, tokenizer files beside), `ACE-Step-1.5-vae-fp32` (weight_norm folded, convs (O, K, I), Snake params [C], encoder kept). LICENSE + NOTICE (the two manifest footnotes) + README in every bundle. `--weights <snapshot> --github <clone> --out <dir> [--parts turbo,planner,text,vae]` |
| `convert/convert.py --parts planner-8bit --planner-8bit-from <ACE-Step-1.5-planner-1.7B-bf16>` | **V4b: the pre-quantised planner.** `ACE-Step-1.5-planner-1.7B-8bit`: the layers' 196 Linears through `mx.quantize` (affine, group 64, 8 bits), `weight` / `scales` / `biases` in mlx-lm's layout, the embedding / norms bf16, `"quantization"` in config.json — bit-equal to the package's quantise-at-load tier (`acestep-gates planner8`). |
| `convert/gate_v1.py` | **V1 gate.** Converted tensors bit-equal to upstream's own `dit_convert` / `vae_convert`, the mlx-lm bundles load through `mlx_lm.load` with every tensor bit-equal, MLX conv semantics on the converted layouts reproduce torch (relmax ≤ 1e-5). Run with `MLX_ENABLE_TF32=0`. |
| `oracle-capture/dump_goldens_v3b.py` | **V3b oracle.** The handler loaded as the V2 oracle (no planner): reference-audio timbre (the V2 folk codes + a 30 s reference), cover (strength 1.0 and 0.5), repaint `[10, 20]` s, extend tail / head (repaint past the source), LRC on the V2 goldens, two torchaudio resample cases — with `_mlx_compiled_encode_sample` hooked to sample from `key(seed + 1_000_000 + i)` so the VAE encode is reproducible. `--weights … --github … --goldens <V2 goldens> --cues <prompts.json> --harness <acestep-eval/harness> --out <goldens-v3b> [--cases …]` |
| `oracle-capture/dump_goldens.py` | **V0/V2 oracle.** Runs upstream's pipeline (torch pieces fp32 on the CPU, the official MLX DiT / VAE fp32, mlx-lm planner) on two fixed requests and dumps per case: prompts + ids, the sampled CoT / codes (reference only), text-encoder states, packed conditions, hints, noise, the 8-step latents, DiT velocity at t = 0.75 (torch + MLX), VAE decodes (diffusers full, MLX chunked), teacher-forced planner logits, the FSM's allowed sets. `--weights … --github … --cues <prompts.json> --out <goldens>` |

Swift-side lanes are `acestep-gates` subcommands (header of `Sources/acestep-gates/main.swift`). Build with the Xcode build
system so MLX's metallib ships (AB-B-0007): `swift build -c release --build-system swiftbuild --product acestep-gates`; the
exact fp32 lane needs `MLX_ENABLE_TF32=0` (AB-D-0114).
