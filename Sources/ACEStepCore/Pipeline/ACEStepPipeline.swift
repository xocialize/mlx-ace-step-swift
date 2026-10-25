// ACEStepPipeline.swift — one song the way `acestep.inference.generate_music` + `AceStepHandler.generate_music` make it in
// Custom mode (text2music, thinking): the planner writes the reasoning block and `duration × 5` codes; text2music with codes
// runs as a cover of them (cover instruction, `is_covers`); the caption prompt (`SFT_GEN_PROMPT`) goes through Qwen3-Embedding,
// the lyrics (`# Languages … # Lyric …`) through its embedding table, the timbre reference is 750 frames of the silence latent;
// hints → context `[hints ‖ 1]`; seeded noise → the 8-step solve → the chunked VAE; peak > 1 scaled down, then −1 dBFS.
//
// Cancellation (the engine's CAN gate): `generate(_:progress:checkpoint:)` calls `checkpoint` before every planner token, every
// DiT step and every VAE window and rethrows whatever it throws unchanged; the engine package passes `Task.checkCancellation`.
import Foundation
import MLX
import MLXRandom

public struct ACEStepRequest: Sendable {
    public var caption: String
    public var lyrics: String
    /// The vocal language for the DiT's lyric prompt (`vocal_language`). `nil` = take the planner's `language` line (its CoT
    /// picks one among the 51 codes), falling back to "en" when the planner names none — upstream's default is a literal "en".
    public var language: String? = "en"
    public var duration: Double? = nil
    public var bpm: Int? = nil
    public var keyscale: String? = nil
    public var timesignature: Int? = nil
    public var seed: UInt64 = 0
    public var thinking = true
    public var planner = PlannerSettings()
    public var sampler = SamplerSettings()
    public var vaeChunk = 512
    public var normalize = true
    public var normalizationDB: Double = -1
    public init(caption: String, lyrics: String, language: String? = "en", duration: Double? = nil, seed: UInt64 = 0) {
        self.caption = caption; self.lyrics = lyrics; self.language = language; self.duration = duration; self.seed = seed
    }
}

public struct ACEStepSong: Sendable {
    public var audio: [Float]                 // interleaved stereo (frames × 2)
    public var channels = 2
    public var sampleRate = 48000
    public var codes: [Int]
    public var metadata: PlannerMetadata
    public var cotText: String
    public var frames: Int
    /// The language the lyric prompt was rendered with (the request's, or the planner's when the request left it open).
    public var language: String
    /// The codes phase hit its token cap before the planner's EOS — the song is shorter than the plan asked for.
    public var truncated: Bool
    public var timings: [String: Double]
}

/// The four V1 bundles (`Tools/convert/convert.py`), each a directory with `model.safetensors` + `config.json` (+ tokenizer files,
/// + `silence_latent.safetensors` for the turbo bundle). `under(root)` is the converter's layout; an engine resolves each from
/// its own store instead.
public struct ACEStepBundles: Sendable, Equatable {
    public var turbo: URL
    public var planner: URL
    public var textEncoder: URL
    public var vae: URL
    public init(turbo: URL, planner: URL, textEncoder: URL, vae: URL) {
        self.turbo = turbo; self.planner = planner; self.textEncoder = textEncoder; self.vae = vae
    }
    public static let turboName = "ACE-Step-1.5-turbo-bf16"
    public static let plannerName = "ACE-Step-1.5-planner-1.7B-bf16"
    public static let plannerLiteName = "ACE-Step-1.5-planner-0.6B-bf16"
    public static let textEncoderName = "Qwen3-Embedding-0.6B-bf16"
    public static let vaeName = "ACE-Step-1.5-vae-fp32"
    /// The converter's layout: the four bundle folders side by side under `root`; `planner` overrides the planner folder.
    public static func under(_ root: URL, planner: URL? = nil) -> ACEStepBundles {
        ACEStepBundles(turbo: root.appendingPathComponent(turboName),
                       planner: planner ?? root.appendingPathComponent(plannerName),
                       textEncoder: root.appendingPathComponent(textEncoderName),
                       vae: root.appendingPathComponent(vaeName))
    }
    /// Every file `ACEStepPipeline.init` opens, per bundle — the contract an engine's materialization probes.
    public static let turboFiles = ["model.safetensors", "config.json", "silence_latent.safetensors"]
    public static let qwen3Files = ["model.safetensors", "config.json", "tokenizer.json"]
    public static let vaeFiles = ["model.safetensors", "config.json"]
}

public final class ACEStepPipeline {
    public let model: ACEStepModel
    public let silence: MLXArray               // [15000, 64] fp32
    public let textEncoder: Qwen3LM
    public let textTokenizer: BPETokenizer
    public let planner: Planner
    public let vae: AutoencoderOobleck
    public static let sampleRate = 48000
    public static let hop = 1920

    /// Loads the four bundles; `dtype` nil keeps the bundles' bf16, `.float32` is the exact lane; `plannerBits` / `ditBits`
    /// quantise the Linears at load (group `groupSize`).
    public init(bundles: ACEStepBundles, dtype: DType?, vaeDType: DType = .float32, plannerBits: Int? = nil, ditBits: Int? = nil, groupSize: Int = 64) throws {
        (model, silence) = try ACEStepModel.load(bundles.turbo, dtype: dtype)
        if let ditBits { model.quantizeDiT(groupSize: groupSize, bits: ditBits); eval(model.parameters()) }
        textEncoder = try Qwen3LM.load(bundles.textEncoder, dtype: dtype)
        textTokenizer = try BPETokenizer(directory: bundles.textEncoder)
        planner = try Planner(directory: bundles.planner, dtype: dtype)
        // Quantisation at load is lazy: evaluating the quantised parameters here lets the bf16 source weights go now, not at
        // the first song — otherwise the int8 tier's resident footprint reads as bf16's until the first run (V3 validate).
        if let plannerBits { planner.model.quantizeLayers(groupSize: groupSize, bits: plannerBits); eval(planner.model.parameters()) }
        vae = try AutoencoderOobleck.load(bundles.vae, dtype: vaeDType)
        Memory.clearCache()
    }

    /// The converter's layout under one root (`ACEStepBundles.under`).
    public convenience init(bundles root: URL, dtype: DType?, vaeDType: DType = .float32, plannerBits: Int? = nil, ditBits: Int? = nil, groupSize: Int = 64, planner: URL? = nil) throws {
        try self.init(bundles: .under(root, planner: planner), dtype: dtype, vaeDType: vaeDType, plannerBits: plannerBits, ditBits: ditBits, groupSize: groupSize)
    }

    // MARK: prompts (`_build_metadata_dict`, `_dict_to_meta_string`, SFT_GEN_PROMPT, `_format_lyrics`)

    public static func metas(bpm: Int?, timesignature: String?, keyscale: String?, duration: Double?) -> String {
        let b = bpm.map(String.init) ?? "N/A"
        let ks = (keyscale?.trimmingCharacters(in: .whitespaces).isEmpty == false) ? keyscale! : "N/A"
        let ts = (timesignature.map { !$0.trimmingCharacters(in: .whitespaces).isEmpty && $0 != "N/A" } ?? false) ? timesignature! : "N/A"
        let d = duration.map { "\(Int($0)) seconds" } ?? "30 seconds"
        return "- bpm: \(b)\n- timesignature: \(ts)\n- keyscale: \(ks)\n- duration: \(d)\n"
    }
    public static func textPrompt(instruction: String, caption: String, metas: String) -> String {
        let line = instruction.hasSuffix(":") ? instruction : instruction + ":"
        return "# Instruction\n\(line)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }
    public static func lyricPrompt(lyrics: String, language: String) -> String { "# Languages\n\(language)\n\n# Lyric\n\(lyrics)<|endoftext|>" }

    // MARK: pieces

    /// `infer_text_embeddings`: Qwen3-Embedding last hidden states `[1, L, 1024]`.
    public func textHidden(_ ids: [Int]) -> MLXArray { textEncoder.model(MLXArray(ids.map { Int32($0) }).reshaped(1, -1), caches: nil) }
    /// `infer_lyric_embeddings`: the embedding table `[1, L, 1024]`.
    public func lyricEmbed(_ ids: [Int]) -> MLXArray { textEncoder.model.embedTokens(MLXArray(ids.map { Int32($0) }).reshaped(1, -1)) }
    /// `silence_latent[:, :n]` (tiled when longer than the stored 15000 frames), `[1, n, 64]` in the model dtype.
    public func silenceLatent(_ n: Int) -> MLXArray {
        var s = silence
        if n > s.dim(0) { s = tiled(s, repetitions: [(n + s.dim(0) - 1) / s.dim(0), 1]) }
        return s[..<n].asType(model.dtype).expandedDimensions(axis: 0)
    }
    /// `prepare_condition`'s encoder half with the silence timbre reference.
    public func conditions(textHidden: MLXArray, lyricEmbed: MLXArray, references: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let refs = references ?? silenceLatent(model.config.timbre_fix_frame)
        let d = model.dtype
        return model.encoder(text: textHidden.asType(d), textKeep: MLXArray.ones([1, textHidden.dim(1)], dtype: .int32),
                             lyrics: lyricEmbed.asType(d), lyricKeep: MLXArray.ones([1, lyricEmbed.dim(1)], dtype: .int32),
                             references: refs.asType(d), referenceOrder: [0])
    }
    /// `context_latents` for `codes`: hints padded with silence to `frames`, ‖ the all-ones chunk mask.
    public func context(codes: [Int], frames: Int) -> MLXArray {
        var src = model.hints(codes: codes)
        if src.dim(1) < frames { src = concatenated([src, silenceLatent(frames - src.dim(1))], axis: 1) }
        src = src[0..., ..<frames, 0...]
        return concatenated([src, MLXArray.ones([1, frames, src.dim(2)], dtype: src.dtype)], axis: -1)
    }
    /// `normalize_audio` after the peak clamp: `[S, 2]` fp32 host samples.
    public static func finish(_ audio: MLXArray, normalize: Bool, db: Double) -> MLXArray {
        var a = audio.asType(.float32)
        let peak = abs(a).max().item(Float.self)
        if peak > 1 { a = a / peak }
        if normalize, db <= 0 {
            let p = abs(a).max().item(Float.self)
            if p >= 1e-6 { a = a * Float(pow(10.0, db / 20.0) / Double(p)) }
        }
        return a
    }

    // MARK: generate

    /// `progress(phase, done, total)` with phases `plan` (CoT tokens), `codes`, `denoise` (steps), `decode` (VAE windows);
    /// `checkpoint` runs before every token / step / window and its error is rethrown unchanged.
    public func generate(_ r: ACEStepRequest, progress: ((String, Int, Int) -> Void)? = nil, checkpoint: (() throws -> Void)? = nil) throws -> ACEStepSong {
        var timings: [String: Double] = [:]
        let started = Date()
        var user = UserMetadata()
        if let b = r.bpm, b > 0 { user.bpm = String(b) }
        if let k = r.keyscale?.trimmingCharacters(in: .whitespaces), !k.isEmpty, k.lowercased() != "n/a" { user.keyscale = k }
        if let t = r.timesignature { user.timesignature = String(t) }
        if let d = r.duration, d > 0 { user.duration = String(Int(d)) }
        try checkpoint?()
        planner.random = MLXRandom.RandomState(seed: r.seed)
        let plan = try planner.plan(caption: r.caption, lyrics: r.lyrics, targetDuration: r.duration, userMetadata: user, settings: r.planner,
                                    progress: { phase, done, total in progress?(phase, done, total) }, checkpoint: checkpoint)
        timings["lm_phase1"] = plan.phase1Seconds; timings["lm_phase2"] = plan.phase2Seconds
        guard !plan.codes.isEmpty else { throw ACEStepError.generation("the planner wrote no audio codes") }

        // `_update_metadata_from_lm` + `use_cot_caption`; the DiT language is the request's (upstream reads `vocal_language`),
        // or the planner's when the request left it open.
        var bpm = r.bpm, keyscale = r.keyscale ?? "", timesig = r.timesignature.map(String.init) ?? "", duration = r.duration
        if bpm == nil, let b = plan.metadata.bpm { bpm = b }
        if keyscale.isEmpty, let k = plan.metadata.keyscale, k != "N/A" { keyscale = k }
        if timesig.isEmpty, let t = plan.metadata.timesignature, t != "N/A" { timesig = t }
        if (duration ?? 0) <= 0, let d = plan.metadata.duration { duration = Double(d) }
        let caption = r.planner.useCotCaption ? (plan.metadata.caption ?? r.caption) : r.caption
        let language: String = {
            if let l = r.language, !l.isEmpty { return l }
            if let l = plan.metadata.language, ACEStepConstants.validLanguages.contains(l), l != "unknown" { return l }
            return "en"
        }()

        let t0 = Date()
        try checkpoint?()
        let metas = Self.metas(bpm: bpm, timesignature: timesig.isEmpty ? nil : timesig, keyscale: keyscale, duration: duration)
        let textIDs = textTokenizer.encode(Self.textPrompt(instruction: ACEStepConstants.coverInstruction, caption: caption, metas: metas), maxLength: 256)
        let lyricIDs = textTokenizer.encode(Self.lyricPrompt(lyrics: r.lyrics, language: language), maxLength: 2048)
        let (enc, _) = conditions(textHidden: textHidden(textIDs), lyricEmbed: lyricEmbed(lyricIDs))
        let frames = max(128, plan.codes.count * model.config.pool_window_size)
        let ctx = context(codes: plan.codes, frames: frames)
        let cache = model.decoder.crossCache(enc)
        eval(enc, ctx); eval(cache.layers.flatMap { [$0.0, $0.1] })
        timings["conditions"] = Date().timeIntervalSince(t0)

        let t1 = Date()
        let noise = DiffusionSampler.noise(seed: r.seed, frames: frames, channels: model.config.audio_acoustic_hidden_dim)
        let latents = try DiffusionSampler.solve(decoder: model.decoder, cache: cache, context: ctx, noise: noise, settings: r.sampler) { done, total in
            try checkpoint?()
            progress?("denoise", done, total)
        }
        timings["diffusion"] = Date().timeIntervalSince(t1)
        guard !isNaN(latents).any().item(Bool.self) else { throw ACEStepError.generation("the diffusion produced NaN latents") }

        let t2 = Date()
        try checkpoint?()
        var audio = try vae.decodeChunked(latents, chunk: r.vaeChunk) { done, total in
            try checkpoint?()
            progress?("decode", done, total)
        }[0]
        audio = Self.finish(audio, normalize: r.normalize, db: r.normalizationDB)
        eval(audio)
        timings["vae"] = Date().timeIntervalSince(t2)
        timings["total"] = Date().timeIntervalSince(started)
        Memory.clearCache()
        return ACEStepSong(audio: audio.asArray(Float.self), codes: plan.codes, metadata: plan.metadata, cotText: plan.cotText, frames: frames,
                           language: language, truncated: plan.codesTruncated, timings: timings)
    }
}
