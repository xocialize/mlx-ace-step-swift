// Planner.swift — the 5 Hz planner as `LLMHandler.generate_with_stop_condition` + `_run_mlx_single_native` run it: phase 1
// writes the reasoning block under the metadata constraint (no CFG), phase 2 writes `duration × 5` audio codes with CFG
// `uncond + s·(cond − uncond)` against the "NO USER INPUT" prompt. Prefill is chunked (2048) up to the last prompt token, the
// last token alone; each step is logits → repetition penalty → constraint → log-softmax → top-p → top-k → categorical
// (mlx-lm `make_sampler`). A sampled rollout is never a gate (AB-L-0090); lane P teacher-forces the oracle's ids.
import Foundation
import MLX
import MLXRandom

public struct PlannerSettings: Sendable {
    public var temperature: Float = 0.85
    public var cfgScale: Float = 2.0
    public var topP: Float? = 0.9
    public var topK: Int? = nil
    public var repetitionPenalty: Float = 1.0
    public var negativePrompt = ACEStepConstants.noUserInput
    public var useCotMetas = true, useCotCaption = true, useCotLanguage = true
    public var constrainedDecoding = true
    public init() {}
}

public struct PlannerResult: Sendable {
    public var metadata: PlannerMetadata
    public var cotText: String          // the model's phase-1 output (decoded)
    public var cotBlock: String         // the YAML block fed to phase 2
    public var codes: [Int]
    public var cotTokens: Int
    public var phase1Seconds: Double, phase2Seconds: Double
}

public final class Planner {
    public let model: Qwen3LM
    public let tokenizer: BPETokenizer
    public let constraint: MetadataConstraint
    public let audioCodeBase: Int
    public let maxModelLength = 4096
    public var random = MLXRandom.RandomState(seed: 0)

    public init(directory: URL, dtype: DType?) throws {
        model = try Qwen3LM.load(directory, dtype: dtype)
        tokenizer = try BPETokenizer(directory: directory)
        constraint = try MetadataConstraint(tokenizer: tokenizer)
        guard let base = tokenizer.addedTokens["<|audio_code_0|>"] else { throw ACEStepError.badConfig("planner tokenizer has no <|audio_code_0|>") }
        audioCodeBase = base
    }

    public func encode(_ text: String) -> [Int] { tokenizer.encode(text, addSpecialTokens: false) }

    /// `_compute_max_new_tokens`.
    func maxNewTokens(targetDuration: Double?, phase: String) -> Int {
        var n: Int
        if let d = targetDuration, d > 0 {
            let effective = max(Double(ACEStepConstants.durationRange.lowerBound), min(Double(ACEStepConstants.durationRange.upperBound), d))
            let codes = Int(effective * 5)
            n = phase == "codes" ? codes + 10 : codes + 500
        } else {
            n = min(maxModelLength - 64, ACEStepConstants.durationRange.upperBound * 5 + 500)
        }
        return min(n, maxModelLength - 64)
    }

    /// `has_all_metas`: bpm, keyscale, timesignature and duration all provided.
    public static func hasAllMetas(_ u: UserMetadata) -> Bool { u.bpm != nil && u.keyscale != nil && u.timesignature != nil && u.duration != nil }

    // MARK: the two phases

    /// `generate_with_stop_condition(infer_type="llm_dit")`.
    public func plan(caption: String, lyrics: String, targetDuration: Double?, userMetadata: UserMetadata, settings: PlannerSettings,
                     progress: ((String, Int, Int) -> Void)? = nil) throws -> PlannerResult {
        var metadata = PlannerMetadata()
        var cotText = "", cotTokens = 0, phase1 = 0.0
        if !Self.hasAllMetas(userMetadata) && settings.useCotMetas {
            let t0 = Date()
            let prompt = PlannerText.cotPrompt(caption: caption, lyrics: lyrics)
            constraint.configure(phase: "cot", stopAtReasoning: true, targetDuration: nil, userMetadata: userMetadata,
                                 skipCaption: !settings.useCotCaption, skipLanguage: !settings.useCotLanguage)
            constraint.enabled = settings.constrainedDecoding
            let ids = generate(prompt: encode(prompt), unconditional: nil, cfgScale: 1.0, settings: settings,
                               maxNewTokens: maxNewTokens(targetDuration: targetDuration, phase: "cot")) { progress?("plan", $0, $1) }
            cotText = tokenizer.decode(ids)
            cotTokens = ids.count
            (metadata, _) = PlannerText.parse(cotText)
            phase1 = Date().timeIntervalSince(t0)
        } else {
            metadata = PlannerMetadata(bpm: userMetadata.bpm.flatMap(Int.init), caption: userMetadata.caption, duration: userMetadata.duration.flatMap(Int.init),
                                       keyscale: userMetadata.keyscale, language: userMetadata.language, timesignature: userMetadata.timesignature)
        }
        var target = targetDuration
        if (target ?? 0) <= 0, let d = metadata.duration, d > 0 { target = Double(d) }
        let t1 = Date()
        let cotBlock = PlannerText.cotBlock(metadata)
        let prompt = PlannerText.codesPrompt(caption: caption, lyrics: lyrics, cotBlock: cotBlock)
        let uncond = PlannerText.codesUnconditionalPrompt(negative: settings.negativePrompt)
        constraint.configure(phase: "codes", stopAtReasoning: false, targetDuration: target, userMetadata: UserMetadata(),
                             skipCaption: true, skipLanguage: true, promptContainsThinkEnd: prompt.contains("</think>"))
        constraint.enabled = settings.constrainedDecoding
        let ids = generate(prompt: encode(prompt), unconditional: settings.cfgScale > 1 ? encode(uncond) : nil, cfgScale: settings.cfgScale,
                           settings: settings, maxNewTokens: maxNewTokens(targetDuration: target, phase: "codes")) { progress?("codes", $0, $1) }
        let codes = ids.compactMap { id -> Int? in
            let c = id - audioCodeBase
            return (0 ... ACEStepConstants.maxAudioCode).contains(c) ? c : nil
        }
        return PlannerResult(metadata: metadata, cotText: cotText, cotBlock: cotBlock, codes: codes, cotTokens: cotTokens,
                             phase1Seconds: phase1, phase2Seconds: Date().timeIntervalSince(t1))
    }

    // MARK: decoding loop

    /// Chunked prefill (all but the last token, 2048 at a time), then the last token → its logits `[V]` fp32.
    func prefill(_ ids: [Int], caches: [KVCache]) -> MLXArray {
        let all = MLXArray(ids.map { Int32($0) })
        var start = 0
        var first = true
        while ids.count - start > 1 {
            let chunk = min(2048, ids.count - start - 1)
            _ = model.model(all[start ..< (start + chunk)].expandedDimensions(axis: 0), caches: caches, chunked: !first)
            eval(caches.compactMap(\.keys)); eval(caches.compactMap(\.values))
            start += chunk; first = false
        }
        return model.lastLogits(all[start...].expandedDimensions(axis: 0), caches: caches)
    }

    /// Tokens sampled where `forced` was fed instead (lane P teacher forcing): (sampled, forced) pairs.
    public var teacherLog: [(Int, Int)] = []

    public func generate(prompt: [Int], unconditional: [Int]?, cfgScale: Float, settings: PlannerSettings, maxNewTokens: Int,
                         forced: [Int]? = nil, progress: ((Int, Int) -> Void)? = nil) -> [Int] {
        let caches = model.newCaches()
        let uncaches = unconditional != nil ? model.newCaches() : nil
        var last = prefill(prompt, caches: caches)
        var lastUncond = unconditional.map { prefill($0, caches: uncaches!) }
        var history = prompt
        var produced: [Int] = []
        let eos = constraint.eosTokenID
        let pad = tokenizer.padTokenID
        for step in 0 ..< maxNewTokens {
            var logits = (lastUncond.map { $0 + cfgScale * (last - $0) } ?? last).asType(.float32)
            if settings.repetitionPenalty != 1 {
                let idx = MLXArray(history.map { Int32($0) })
                let selected = logits[idx]
                logits[idx] = MLX.where(selected .> 0, selected / settings.repetitionPenalty, selected * settings.repetitionPenalty)
            }
            if settings.constrainedDecoding { logits = constraint.apply(constraint.constrain(logits), to: logits) }
            var id = TokenSampling.sample(logits, temperature: settings.temperature, topP: settings.topP, topK: settings.topK, random: &random)
            if let forced {
                guard step < forced.count else { break }
                teacherLog.append((id, forced[step])); id = forced[step]
            }
            produced.append(id); history.append(id)
            constraint.update(with: id)
            progress?(produced.count, maxNewTokens)
            if id == eos { break }
            if let pad, pad != eos, id == pad { break }
            let input = MLXArray([Int32(id)]).reshaped(1, 1)
            last = model.lastLogits(input, caches: caches)
            if let uncaches { lastUncond = model.lastLogits(input, caches: uncaches) }
            if step % 256 == 0 && step > 0 { Memory.clearCache() }
        }
        return produced
    }

    /// Teacher-forced logits `[L, V]` fp32 over `ids` (lane P): one forward, no cache.
    public func teacherForcedLogits(_ ids: [Int]) -> MLXArray {
        let x = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
        return model.logits(x, caches: nil)[0].asType(.float32)
    }
}

/// mlx-lm `make_sampler(temp, top_p, top_k)`: log-softmax → top-p (nucleus, ascending cumsum) → top-k → categorical(logits/temp).
public enum TokenSampling {
    public static func sample(_ logits: MLXArray, temperature: Float, topP: Float?, topK: Int?, random: inout MLXRandom.RandomState) -> Int {
        if temperature == 0 { return Int(argMax(logits).item(Int32.self)) }
        var logprobs = logits - logSumExp(logits)
        if let p = topP, p > 0, p < 1 {
            let probs = exp(logprobs)
            let sortedIdx = argSort(logprobs)
            let sortedProbs = takeAlong(probs, sortedIdx, axis: -1)
            let cumulative = cumsum(sortedProbs)
            var inverse = MLXArray.zeros([logprobs.dim(0)], dtype: .int32)
            inverse[sortedIdx] = MLXArray(Int32(0) ..< Int32(logprobs.dim(0)))
            let cumulativeProbs = takeAlong(cumulative, inverse, axis: -1)
            logprobs = MLX.where(cumulativeProbs .> (1 - p), logprobs, MLXArray(-Float.infinity))
        }
        if let k = topK, k > 0, k < logprobs.dim(0) {
            let order = argSort(-logprobs)
            let drop = order[k...]
            logprobs[drop] = MLXArray(-Float.infinity)
        }
        let key = random.next()
        let token = MLXRandom.categorical(logprobs * (1 / temperature), key: key)
        eval(token)
        return Int(token.item(Int32.self))
    }
}
