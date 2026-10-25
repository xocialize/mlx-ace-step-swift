// swift-tools-version: 6.2
// mlx-ace-step-swift — ACE-Step 1.5 (ACE Studio + StepFun; caption + lyrics → a 48 kHz stereo song through a 5 Hz planner LM,
// an FSQ detokenizer, a 2 B flow-matching DiT and the Oobleck VAE) ported to Swift-MLX. Port plan, gates and numbers:
// PORTING-SPEC.md, MEASUREMENTS.md; task AB-T-0352 (E25), decision AB-D-0153 (full port from upstream).
//
// Licence: this package's code is MIT. The weights (ACE-Step/Ace-Step1.5 @ 19671f4) are MIT per the GitHub LICENSE + the HF
// card; the Qwen3 bases are Apache-2.0. Two manifest footnotes travel with every bundle (Tools/convert/convert.py NOTICE).
//
// Pipeline (ACEStepCore, 1:1 with ace-step/ACE-Step-1.5 @ ca1e85f):
//   • Tokenizer  — the Qwen2 byte-level BPE (tokenizer.json), added-token split, the embedding tokenizer's <|endoftext|> template.
//   • Planner    — Qwen3-1.7B (mlx-lm layout), two phases: constrained YAML CoT (bpm/caption/duration/keyscale/language/
//                  timesignature), then duration × 5 audio codes with CFG 2.0 against "NO USER INPUT".
//   • Text       — Qwen3-Embedding-0.6B last hidden states (caption prompt) and its embedding table (lyrics).
//   • Conditions — lyric encoder (8 layers) + timbre encoder (4 layers, silence when no reference) + projected text, packed.
//   • Hints      — ResidualFSQ codes → 25 Hz latents through the 2-layer detokenizer.
//   • DiT        — 24 layers alternating sliding-128 / full attention, adaLN, cross-attention K/V cached per condition;
//                  8 Euler steps (shift 1.0 default, the pipeline's) with the DCW Haar correction.
//   • VAE        — Oobleck decoder, chunked at 512 frames / 64 overlap; peak clamp, −1 dBFS normalisation.
//   • acestep-gates — the parity lanes against the fp32 oracle (Tools/oracle-capture) and a render lane.
import PackageDescription

let package = Package(
    name: "mlx-ace-step-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "ACEStepCore", targets: ["ACEStepCore"]),
        .executable(name: "acestep-gates", targets: ["acestep-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
    ],
    targets: [
        .target(
            name: "ACEStepCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/ACEStepCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "acestep-gates",
            dependencies: [
                "ACEStepCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/acestep-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ACEStepCoreTests",
            dependencies: ["ACEStepCore"],
            path: "Tests/ACEStepCoreTests"
        ),
    ]
)
