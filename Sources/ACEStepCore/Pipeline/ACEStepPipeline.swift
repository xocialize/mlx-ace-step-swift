// ACEStepPipeline.swift — one song the way `acestep.inference.generate_music` + `AceStepHandler.generate_music` make it.
// Custom mode (text2music, thinking): the planner writes the reasoning block and `duration × 5` codes; text2music with codes
// runs as a cover of them (cover instruction, `is_covers`); the caption prompt (`SFT_GEN_PROMPT`) goes through Qwen3-Embedding,
// the lyrics (`# Languages … # Lyric …`) through its embedding table, the timbre reference is a 30 s clip through the VAE
// encoder (or 750 frames of the silence latent); hints → context `[hints ‖ 1]`; seeded noise → the 8-step solve → the chunked
// VAE; peak > 1 scaled down, then −1 dBFS. The source-audio tasks (`DIRECT_CONDITIONING_TASKS`: cover / repaint, and extend as
// a repaint past the source's ends) skip the planner: caption, lyrics and anchors come from the request, the length is the
// source's, the source is VAE-encoded (cover: tokenized to its own hints; repaint: the range replaced by silence, re-injected
// for half the steps, blended, and spliced back after the VAE). LRC timestamps (`get_lyric_timestamp`) come from the DiT's
// cross-attention at t = 1/8 on every run that asks for them.
//
// Cancellation (the engine's CAN gate): `generate(_:progress:checkpoint:)` calls `checkpoint` before every planner token, every
// DiT step, every VAE window (encode and decode) and rethrows whatever it throws unchanged; the engine package passes
// `Task.checkCancellation`.
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
    /// A clip whose timbre the vocals take (`reference_audio`); nil = the silence reference.
    public var referenceAudio: AudioClip? = nil
    /// A source clip and what to do to it (`src_audio` + `task_type`); nil = text2music.
    public var edit: ACEStepEdit? = nil
    /// Produce LRC timestamps from the DiT's cross-attention (`get_lyric_timestamp`).
    public var timestamps = true
    /// The LRC noise seed (upstream's UI hard-codes 42).
    public var lrcSeed: UInt64 = 42
    public var repaint = RepaintConfig.balanced
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
    /// LRC timestamps (lines + tokens + the LRC text) when the request asked for them.
    public var timestamps: LyricTimestamps?
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
    /// The VAE encode's sampling keys: `key(seed + 1_000_000 + i)`, `i` counting encode calls in upstream's order (the source's
    /// windows first, then the reference's) — the V3b declared divergence from upstream's unseeded draw.
    public static func encodeKey(seed: UInt64, call: Int) -> MLXArray { MLXRandom.key(seed &+ 1_000_000 &+ UInt64(call)) }

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
    public static func lyricHeader(language: String) -> String { "# Languages\n\(language)\n\n# Lyric\n" }

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
    /// `prepare_condition`'s encoder half; `references` `[1, 750, 64]` from a reference clip, else the silence timbre reference.
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
    /// `[latents ‖ chunk mask]` — `chunkMask` `[1, T]` (nil = all ones).
    public func context(latents: MLXArray, chunkMask: MLXArray? = nil) -> MLXArray {
        let m = (chunkMask ?? MLXArray.ones([1, latents.dim(1)], dtype: latents.dtype)).asType(latents.dtype)
        return concatenated([latents, broadcast(m.expandedDimensions(axis: -1), to: latents.shape)], axis: -1)
    }
    /// `infer_refer_latent`'s non-silence branch: a loaded stereo 48 kHz clip → the 30 s reference → `[1, 750, 64]`.
    public func referenceLatents(_ audio: MLXArray, seed: UInt64, key: (Int) -> MLXArray, onWindow: ((Int, Int) throws -> Void)? = nil) throws -> MLXArray {
        let clip = AudioLoader.referenceClip(audio, seed: seed)
        return try vae.encodeChunked(clip.expandedDimensions(axis: 0), key: key, onWindow: onWindow)
    }
    /// The handler's peak clamp before the splice: a decode whose |peak| > 1 is scaled to 1.
    public static func clampPeak(_ audio: MLXArray) -> MLXArray {
        let a = audio.asType(.float32)
        let peak = abs(a).max().item(Float.self)
        return peak > 1 ? a / peak : a
    }
    /// `normalize_audio` after the peak clamp: `[S, 2]` fp32 host samples.
    public static func finish(_ audio: MLXArray, normalize: Bool, db: Double) -> MLXArray {
        var a = clampPeak(audio)
        if normalize, db <= 0 {
            let p = abs(a).max().item(Float.self)
            if p >= 1e-6 { a = a * Float(pow(10.0, db / 20.0) / Double(p)) }
        }
        return a
    }

    /// `get_lyric_timestamp`: the DiT's cross-attention at `t = 1 / steps` on `t·noise + (1 − t)·latents` → LRC.
    public func lyricTimestamps(latents: MLXArray, cache: CrossAttentionCache, context: MLXArray, lyricIDs: [Int], language: String,
                                duration: Double, steps: Int, noise: MLXArray) -> LyricTimestamps {
        let t = 1.0 / Double(steps)
        let x = (Float(t) * noise.asType(latents.dtype) + Float(1 - t) * latents).asType(model.dtype)
        let tArr = MLXArray([Float(t)]).asType(model.dtype)
        let wanted = Dictionary(uniqueKeysWithValues: LyricAligner.defaultHeads.map { ($0.layer, $0.heads) })
        let captured = model.decoder.crossAttentions(x, timestep: tArr, timestepR: tArr, contextLatents: context.asType(model.dtype), cache: cache, layers: wanted)
        let header = textTokenizer.encode(Self.lyricHeader(language: language), addSpecialTokens: false)
        let rows = LyricAligner.lyricRows(lyricIDs: lyricIDs, headerLength: header.count)
        let stack = LyricAligner.stack(captured, heads: LyricAligner.defaultHeads, rows: rows)
        return LyricAligner.align(stack: stack, tokens: Array(lyricIDs[rows]), duration: duration, tokenizer: textTokenizer)
    }

    // MARK: generate

    /// `progress(phase, done, total)` with phases `plan` (CoT tokens), `codes`, `encode` (VAE encode windows), `denoise`
    /// (steps), `decode` (VAE windows); `checkpoint` runs before every token / step / window and its error is rethrown unchanged.
    public func generate(_ r: ACEStepRequest, progress: ((String, Int, Int) -> Void)? = nil, checkpoint: (() throws -> Void)? = nil) throws -> ACEStepSong {
        var timings: [String: Double] = [:]
        let started = Date()
        var encodeCalls = 0
        let encodeKey: (Int) -> MLXArray = { _ in defer { encodeCalls += 1 }; return Self.encodeKey(seed: r.seed, call: encodeCalls) }

        // ---- the plan: the planner's (text2music) or the request's (the direct-conditioned source tasks) ----
        var codes: [Int] = []
        var plan: PlannerResult? = nil
        var bpm = r.bpm, keyscale = r.keyscale ?? "", timesig = r.timesignature.map(String.init) ?? "", duration = r.duration
        var caption = r.caption
        var instruction = ACEStepConstants.coverInstruction
        var geometry: EditGeometry? = nil
        var source: MLXArray? = nil               // the padded source `[S, 2]`
        var target: MLXArray? = nil               // its latents `[1, T, 64]`
        var coverStrength = 1.0
        if let edit = r.edit {
            try checkpoint?()
            let t0 = Date()
            let loaded = try AudioLoader.stereo48k(edit.source)
            guard !AudioLoader.isSilence(loaded) else { throw ACEStepError.invalidInput("the source audio is silent") }
            let g = EditGeometry.make(sourceSeconds: Double(loaded.dim(0)) / Double(Self.sampleRate), kind: edit.kind)
            let padded = g.pad(loaded, sampleRate: Self.sampleRate)
            eval(padded)
            target = try vae.encodeChunked(padded.expandedDimensions(axis: 0), key: encodeKey) { done, total in
                try checkpoint?(); progress?("encode", done, total)
            }
            source = padded; geometry = g
            duration = g.sourceSeconds                           // `calculated_duration`: the unpadded source's length
            if case .cover(let s) = edit.kind { coverStrength = max(0, min(1, s)); instruction = ACEStepConstants.coverInstruction }
            else { instruction = ACEStepConstants.repaintInstruction }
            timings["encode"] = Date().timeIntervalSince(t0)
        } else {
            var user = UserMetadata()
            if let b = r.bpm, b > 0 { user.bpm = String(b) }
            if let k = r.keyscale?.trimmingCharacters(in: .whitespaces), !k.isEmpty, k.lowercased() != "n/a" { user.keyscale = k }
            if let t = r.timesignature { user.timesignature = String(t) }
            if let d = r.duration, d > 0 { user.duration = String(Int(d)) }
            try checkpoint?()
            planner.random = MLXRandom.RandomState(seed: r.seed)
            let p = try planner.plan(caption: r.caption, lyrics: r.lyrics, targetDuration: r.duration, userMetadata: user, settings: r.planner,
                                     progress: { phase, done, total in progress?(phase, done, total) }, checkpoint: checkpoint)
            timings["lm_phase1"] = p.phase1Seconds; timings["lm_phase2"] = p.phase2Seconds
            guard !p.codes.isEmpty else { throw ACEStepError.generation("the planner wrote no audio codes") }
            // `_update_metadata_from_lm` + `use_cot_caption`
            if bpm == nil, let b = p.metadata.bpm { bpm = b }
            if keyscale.isEmpty, let k = p.metadata.keyscale, k != "N/A" { keyscale = k }
            if timesig.isEmpty, let t = p.metadata.timesignature, t != "N/A" { timesig = t }
            if (duration ?? 0) <= 0, let d = p.metadata.duration { duration = Double(d) }
            if r.planner.useCotCaption, let c = p.metadata.caption { caption = c }
            codes = p.codes; plan = p
        }
        // the DiT language is the request's (upstream reads `vocal_language`), or the planner's when the request left it open
        let language: String = {
            if let l = r.language, !l.isEmpty { return l }
            if let l = plan?.metadata.language, ACEStepConstants.validLanguages.contains(l), l != "unknown" { return l }
            return "en"
        }()

        // ---- the reference (after the source, in upstream's encode order) ----
        var references: MLXArray? = nil
        if let ref = r.referenceAudio {
            try checkpoint?()
            let t0 = Date()
            let loaded = try AudioLoader.stereo48k(ref)
            guard !AudioLoader.isSilence(loaded) else { throw ACEStepError.invalidInput("the reference audio is silent") }
            references = try referenceLatents(loaded, seed: r.seed, key: encodeKey) { done, total in try checkpoint?(); progress?("encode", done, total) }
            timings["reference"] = Date().timeIntervalSince(t0)
        }

        // ---- conditions + context ----
        let t0 = Date()
        try checkpoint?()
        let metas = Self.metas(bpm: bpm, timesignature: timesig.isEmpty ? nil : timesig, keyscale: keyscale, duration: duration)
        let textIDs = textTokenizer.encode(Self.textPrompt(instruction: instruction, caption: caption, metas: metas), maxLength: 256)
        let lyricIDs = textTokenizer.encode(Self.lyricPrompt(lyrics: r.lyrics, language: language), maxLength: 2048)
        let lyric = lyricEmbed(lyricIDs)
        let (enc, _) = conditions(textHidden: textHidden(textIDs), lyricEmbed: lyric, references: references)
        var frames: Int
        var ctx: MLXArray
        var repaintPlan: RepaintPlan? = nil
        var coverSwitch: CoverSwitch? = nil
        if let geometry, let target {
            // `_prepare_target_latents_and_wavs`: at least 128 frames, padded with the silence latent
            frames = max(128, target.dim(1))
            var tgt = target.asType(model.dtype)
            if tgt.dim(1) < frames { tgt = concatenated([tgt, silenceLatent(frames - tgt.dim(1))], axis: 1) }
            if geometry.isCover {
                ctx = context(latents: model.coverHints(tgt, silence: silence))
                if coverStrength < 1 {
                    // the non-cover pair: the default instruction's text, the silence context, a fresh cache
                    let nonCoverIDs = textTokenizer.encode(Self.textPrompt(instruction: ACEStepConstants.defaultDiTInstruction, caption: caption, metas: metas), maxLength: 256)
                    let (encNC, _) = conditions(textHidden: textHidden(nonCoverIDs), lyricEmbed: lyric, references: references)
                    let cacheNC = model.decoder.crossCache(encNC)
                    eval(encNC); eval(cacheNC.layers.flatMap { [$0.0, $0.1] })
                    coverSwitch = CoverSwitch(atStep: Int(Double(r.sampler.steps) * coverStrength), cache: cacheNC, context: context(latents: silenceLatent(frames)))
                }
            } else {
                let ranges = geometry.latentRanges(frames: frames, sampleRate: Self.sampleRate, hop: Self.hop)
                let masks = RepaintMasks.build(target: tgt, silence: silence, ranges: ranges)
                ctx = context(latents: masks.sourceLatents, chunkMask: masks.chunkMask)
                repaintPlan = RepaintPlan(mask: masks.repaintMask, cleanSource: tgt, crossfadeFrames: r.repaint.crossfadeFrames, injectionRatio: r.repaint.injectionRatio)
            }
        } else {
            frames = max(128, codes.count * model.config.pool_window_size)
            ctx = context(codes: codes, frames: frames)
        }
        let cache = model.decoder.crossCache(enc)
        eval(enc, ctx); eval(cache.layers.flatMap { [$0.0, $0.1] })
        timings["conditions"] = Date().timeIntervalSince(t0)

        // ---- the solve ----
        let t1 = Date()
        let noise = DiffusionSampler.noise(seed: r.seed, frames: frames, channels: model.config.audio_acoustic_hidden_dim)
        let latents = try DiffusionSampler.solve(decoder: model.decoder, cache: cache, context: ctx, noise: noise, settings: r.sampler,
                                                 repaint: repaintPlan, cover: coverSwitch) { done, total in
            try checkpoint?()
            progress?("denoise", done, total)
        }
        timings["diffusion"] = Date().timeIntervalSince(t1)
        guard !isNaN(latents).any().item(Bool.self) else { throw ACEStepError.generation("the diffusion produced NaN latents") }

        // ---- LRC (the first conditions' cache and context, as the handler hands them to get_lyric_timestamp) ----
        var stamps: LyricTimestamps? = nil
        if r.timestamps {
            let t = Date()
            try checkpoint?()
            let total = geometry == nil ? (duration ?? Double(frames) / 25) : Double(frames) / 25
            stamps = lyricTimestamps(latents: latents, cache: cache, context: ctx, lyricIDs: lyricIDs, language: language, duration: total,
                                     steps: r.sampler.steps, noise: MLXRandom.normal(latents.shape, key: MLXRandom.key(r.lrcSeed)))
            timings["lrc"] = Date().timeIntervalSince(t)
        }

        // ---- the VAE, the splice, the finish ----
        let t2 = Date()
        try checkpoint?()
        var audio = try vae.decodeChunked(latents, chunk: r.vaeChunk) { done, total in
            try checkpoint?()
            progress?("decode", done, total)
        }[0]
        audio = Self.clampPeak(audio)
        if let geometry, let source, !geometry.isCover, r.repaint.injectionRatio > 0 || r.repaint.crossfadeFrames > 0 || r.repaint.waveformCrossfadeSeconds > 0 {
            // `do_wav_splice` = repaint_mode != "aggressive"
            audio = WaveformSplice.apply(pred: audio, source: source, ranges: geometry.ranges, sampleRate: Self.sampleRate, crossfadeSeconds: r.repaint.waveformCrossfadeSeconds)
        }
        audio = Self.finish(audio, normalize: r.normalize, db: r.normalizationDB)
        eval(audio)
        timings["vae"] = Date().timeIntervalSince(t2)
        timings["total"] = Date().timeIntervalSince(started)
        Memory.clearCache()
        return ACEStepSong(audio: audio.asArray(Float.self), codes: codes, metadata: plan?.metadata ?? PlannerMetadata(bpm: bpm, caption: caption, duration: duration.map { Int($0) }, keyscale: keyscale.isEmpty ? nil : keyscale, language: language, timesignature: timesig.isEmpty ? nil : timesig),
                           cotText: plan?.cotText ?? "", frames: frames, language: language, truncated: plan?.codesTruncated ?? false, timestamps: stamps, timings: timings)
    }
}
