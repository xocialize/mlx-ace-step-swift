// Constraint.swift — acestep/constrained_logits_processor.py `MetadataConstrainedLogitsProcessor`: the FSM that forces the
// planner's reasoning block into `<think>\nbpm: …\ncaption: …\nduration: …\nkeyscale: …\nlanguage: …\ntimesignature: …\n</think>`
// token by token (fixed strings by longest single-token prefix, numeric / keyscale / language values through prefix trees built
// from the tokenizer's own encodings, the caption free but for audio codes and backticks, user-provided fields injected), then
// audio codes only (EOS blocked until `duration × 5` codes, forced after). Ported state for state; gated by replaying the
// oracle's CoT ids (lane K). `genres` is skipped as upstream skips it; the `understand` phase is not ported.
import Foundation
import MLX

public enum FSMState: String, Sendable {
    case thinkTag = "THINK_TAG", newlineAfterThink = "NEWLINE_AFTER_THINK"
    case bpmName = "BPM_NAME", bpmValue = "BPM_VALUE"
    case captionName = "CAPTION_NAME", captionValue = "CAPTION_VALUE"
    case durationName = "DURATION_NAME", durationValue = "DURATION_VALUE"
    case genresName = "GENRES_NAME", genresValue = "GENRES_VALUE"
    case keyscaleName = "KEYSCALE_NAME", keyscaleValue = "KEYSCALE_VALUE"
    case languageName = "LANGUAGE_NAME", languageValue = "LANGUAGE_VALUE"
    case timesigName = "TIMESIG_NAME", timesigValue = "TIMESIG_VALUE"
    case thinkEndTag = "THINK_END_TAG", codesGeneration = "CODES_GENERATION", completed = "COMPLETED"
}

/// What one step may sample: everything, a whitelist (`uniform` = upstream zeroed the scores before whitelisting — a
/// recursion quirk after an exhausted fixed string), or everything but a blocklist.
public enum TokenConstraint: Equatable {
    case none
    case whitelist([Int], uniform: Bool)
    case blocklist([Int])
}

/// User-provided fields (`set_user_metadata`): a value present is injected verbatim instead of generated.
public struct UserMetadata: Sendable {
    public var bpm: String? = nil, caption: String? = nil, duration: String? = nil, keyscale: String? = nil, language: String? = nil, timesignature: String? = nil
    public init() {}
    func value(_ field: String) -> String? {
        switch field {
        case "bpm": return bpm; case "caption": return caption; case "duration": return duration; case "keyscale": return keyscale
        case "language": return language; case "timesignature": return timesignature; default: return nil
        }
    }
}

public final class MetadataConstraint {
    let tokenizer: BPETokenizer
    public let vocabularySize: Int
    public let eosTokenID: Int
    let maxDuration: Int

    // precomputed tokens
    var digitTokens: [Int: Int] = [:]
    var newlineToken: Int = -1, periodToken: Int = -1, backtickToken: Int = -1
    public private(set) var audioCodeTokenIDs: [Int] = []          // sorted, codes 0 … 63999
    var audioCodeSet = Set<Int>()
    public private(set) var audioCodeMask: MLXArray!               // [V] fp32: −inf at audio codes
    public private(set) var nonAudioCodeMask: MLXArray!            // [V] fp32: −inf except audio codes + EOS
    var keyscaleTree: [[Int]: Set<Int>] = [:], bpmTree: [[Int]: Set<Int>] = [:], durationTree: [[Int]: Set<Int>] = [:]
    var timesigTree: [[Int]: Set<Int>] = [:], languageTree: [[Int]: Set<Int>] = [:]
    let fixedStrings: [FSMState: String] = [
        .thinkTag: "<think>", .newlineAfterThink: "\n", .bpmName: "bpm:", .captionName: "caption:", .durationName: "duration:",
        .genresName: "genres:", .keyscaleName: "keyscale:", .languageName: "language:", .timesigName: "timesignature:", .thinkEndTag: "</think>",
    ]

    // settings
    public var enabled = true
    public var skipGenres = true, skipCaption = false, skipLanguage = false
    public var userMetadata = UserMetadata() { didSet { buildStateTransitions() } }
    public var targetDuration: Double? = nil { didSet { targetCodes = (targetDuration ?? 0) > 0 ? Int(targetDuration! * 5) : nil } }
    public private(set) var targetCodes: Int? = nil
    public var stopAtReasoning = false
    public var generationPhase = "cot"
    var nextState: [FSMState: FSMState] = [:]

    // state
    public private(set) var state: FSMState = .thinkTag
    var positionInState = 0
    var accumulatedValue = ""
    var accumulatedTokenIDs: [Int] = []
    var captionAfterNewline = false, captionTokenCount = 0, captionEnding = false, pendingFieldName = ""
    var userFieldTokenQueue: [Int] = [], currentUserField: String? = nil
    public private(set) var codesCount = 0

    public init(tokenizer: BPETokenizer, maxDuration: Int = ACEStepConstants.durationRange.upperBound) throws {
        self.tokenizer = tokenizer
        self.maxDuration = maxDuration
        vocabularySize = tokenizer.vocabularySize
        guard let eos = tokenizer.eosTokenID else { throw ACEStepError.badConfig("planner tokenizer has no eos_token") }
        eosTokenID = eos
        precomputeTokens()
        keyscaleTree = buildTree(values: ACEStepConstants.validKeyscales, matching: "keyscale:", tokenization: "keyscale: ", requireNote: true)
        bpmTree = buildTree(values: ACEStepConstants.bpmRange.map(String.init), matching: "bpm:", tokenization: "bpm: ")
        durationTree = buildTree(values: (ACEStepConstants.durationRange.lowerBound ... maxDuration).map(String.init), matching: "duration:", tokenization: "duration: ")
        timesigTree = buildTree(values: ACEStepConstants.validTimeSignatures.map(String.init), matching: "timesignature:", tokenization: "timesignature: ")
        languageTree = buildTree(values: ACEStepConstants.validLanguages, matching: "language:", tokenization: "language: ")
        buildStateTransitions()
    }

    func encode(_ s: String) -> [Int] { tokenizer.encode(s, addSpecialTokens: false) }
    func decode(_ id: Int) -> String { tokenizer.decode([id]) }

    private func precomputeTokens() {
        for d in 0 ... 9 { if let last = encode(String(d)).last { digitTokens[d] = last } }
        newlineToken = encode("\n").last ?? -1
        periodToken = encode(".").last ?? -1
        backtickToken = encode("`").last ?? -1
        var ids: [Int] = []
        for (content, id) in tokenizer.addedTokens where content.hasPrefix("<|audio_code_") && content.hasSuffix("|>") {
            let body = content.dropFirst("<|audio_code_".count).dropLast(2)
            if let n = Int(body), n >= 0, n <= ACEStepConstants.maxAudioCode { ids.append(id) }
        }
        ids.sort()
        audioCodeTokenIDs = ids; audioCodeSet = Set(ids)
        var mask = [Float](repeating: 0, count: vocabularySize), inverse = [Float](repeating: -.infinity, count: vocabularySize)
        for id in ids { mask[id] = -.infinity; inverse[id] = 0 }
        inverse[eosTokenID] = 0
        audioCodeMask = MLXArray(mask); nonAudioCodeMask = MLXArray(inverse)
        eval(audioCodeMask, nonAudioCodeMask)
    }

    /// `_build_*_prefix_tree`: token-id-sequence prefixes → allowed next ids (newline once the value is complete).
    private func buildTree(values: [String], matching: String, tokenization: String, requireNote: Bool = false) -> [[Int]: Set<Int>] {
        var tree: [[Int]: Set<Int>] = [:]
        let context = encode(matching)
        for v in values {
            let full = encode(tokenization + v)
            guard full.count >= context.count, Array(full[..<context.count]) == context else { continue }
            let value = Array(full[context.count...])
            guard !value.isEmpty else { continue }
            if requireNote {
                let first = decode(value[0]).drop { $0 == " " }
                guard let ch = first.first, "ABCDEFG".contains(ch.uppercased()) else { continue }
            }
            for i in 0 ... value.count {
                let prefix = Array(value[..<i])
                if i < value.count { tree[prefix, default: []].insert(value[i]) } else if newlineToken >= 0 { tree[prefix, default: []].insert(newlineToken) }
            }
        }
        return tree
    }

    private func nextFieldState(after field: String) -> FSMState {
        let order = ["bpm", "caption", "duration", "genres", "keyscale", "language", "timesignature"]
        let toState: [String: FSMState] = ["bpm": .bpmName, "caption": .captionName, "duration": .durationName, "genres": .genresName,
                                           "keyscale": .keyscaleName, "language": .languageName, "timesignature": .timesigName]
        guard let i = order.firstIndex(of: field) else { return .thinkEndTag }
        for f in order[(i + 1)...] {
            if f == "genres" && skipGenres { continue }
            if f == "caption" && skipCaption { continue }
            if f == "language" && skipLanguage { continue }
            return toState[f]!
        }
        return .thinkEndTag
    }

    private func buildStateTransitions() {
        nextState = [.thinkTag: .newlineAfterThink, .newlineAfterThink: .bpmName, .thinkEndTag: .codesGeneration, .codesGeneration: .completed]
        nextState[.bpmName] = .bpmValue; nextState[.bpmValue] = nextFieldState(after: "bpm")
        if !skipCaption { nextState[.captionName] = .captionValue; nextState[.captionValue] = nextFieldState(after: "caption") }
        nextState[.durationName] = .durationValue; nextState[.durationValue] = nextFieldState(after: "duration")
        if !skipGenres { nextState[.genresName] = .genresValue; nextState[.genresValue] = nextFieldState(after: "genres") }
        nextState[.keyscaleName] = .keyscaleValue; nextState[.keyscaleValue] = nextFieldState(after: "keyscale")
        if !skipLanguage { nextState[.languageName] = .languageValue; nextState[.languageValue] = nextFieldState(after: "language") }
        nextState[.timesigName] = .timesigValue; nextState[.timesigValue] = .thinkEndTag
    }

    /// `_setup_constrained_processor` for one generation.
    public func configure(phase: String, stopAtReasoning: Bool, targetDuration: Double?, userMetadata: UserMetadata,
                          skipCaption: Bool, skipLanguage: Bool, skipGenres: Bool = true, promptContainsThinkEnd: Bool = false) {
        reset()
        self.targetDuration = targetDuration
        self.userMetadata = userMetadata
        self.stopAtReasoning = stopAtReasoning
        self.skipGenres = skipGenres; self.skipCaption = skipCaption; self.skipLanguage = skipLanguage
        buildStateTransitions()
        generationPhase = phase
        if phase == "codes" && state == .thinkTag && promptContainsThinkEnd { state = .codesGeneration; codesCount = 0 }
    }

    public func reset() {
        state = .thinkTag; positionInState = 0; accumulatedValue = ""; accumulatedTokenIDs = []; codesCount = 0
        userFieldTokenQueue = []; currentUserField = nil
        captionAfterNewline = false; captionTokenCount = 0; captionEnding = false; pendingFieldName = ""
    }

    // MARK: __call__

    /// The constraint for the next token given the (fp32, `[V]`) logits of this step.
    public func constrain(_ scores: MLXArray) -> TokenConstraint {
        guard enabled else { return .none }
        if state == .completed { return .none }
        if state == .codesGeneration {
            var allowed = audioCodeTokenIDs
            if let target = targetCodes {
                if codesCount < target { /* EOS blocked */ } else { return .whitelist([eosTokenID], uniform: false) }
            } else { allowed.append(eosTokenID) }
            return .whitelist(allowed, uniform: false)
        }
        return processSingle(scores, uniform: false)
    }

    private func allowedForFixedString(_ fixed: String) -> [Int] {
        let scalars = Array(fixed.unicodeScalars)
        guard positionInState < scalars.count else { return [] }
        let remaining = Array(scalars[positionInState...])
        for end in stride(from: remaining.count, to: 0, by: -1) {
            let prefix = String(String.UnicodeScalarView(remaining[..<end]))
            let toks = encode(prefix)
            if toks.count == 1 { return [toks[0]] }
        }
        var found: [Int: Int] = [:]
        for end in 1 ..< min(remaining.count + 1, 20) {
            let prefix = String(String.UnicodeScalarView(remaining[..<end]))
            let toks = encode(prefix)
            guard let first = toks.first else { continue }
            let np = prefix.drop { $0 == " " }.lowercased(), nd = decode(first).drop { $0 == " " }.lowercased()
            if nd.hasPrefix(np) || np.hasPrefix(nd) { if found[first] == nil || end > found[first]! { found[first] = end } }
        }
        return found.sorted { $0.value > $1.value }.map(\.key)
    }

    private func injectUserField(_ field: String, _ value: String) -> TokenConstraint? {
        let toks = encode(" \(value)\n")
        guard !toks.isEmpty else { return nil }
        userFieldTokenQueue = toks; currentUserField = field
        return .whitelist([toks[0]], uniform: false)
    }

    private func processSingle(_ scores: MLXArray, uniform: Bool) -> TokenConstraint {
        if let next = userFieldTokenQueue.first { return .whitelist([next], uniform: uniform) }
        if let fixed = fixedStrings[state] {
            let allowed = allowedForFixedString(fixed)
            if !allowed.isEmpty {
                if state == .thinkEndTag && stopAtReasoning {
                    let remaining = fixed.unicodeScalars.count - positionInState
                    if remaining <= 10 { return .whitelist([eosTokenID], uniform: uniform) }
                }
                return .whitelist(allowed, uniform: uniform)
            }
            if state == .thinkEndTag && stopAtReasoning { return .whitelist([eosTokenID], uniform: uniform) }
            transitionToNextState()
            if fixedStrings[state] != nil { return uniform ? .whitelist([], uniform: true) : .none }
            return processSingle(scores, uniform: true)
        }
        switch state {
        case .bpmValue:
            if let v = userMetadata.bpm, userFieldTokenQueue.isEmpty, accumulatedTokenIDs.isEmpty, let c = injectUserField("bpm", v) { return c }
            var allowed = Array(bpmTree[accumulatedTokenIDs] ?? [])
            if let set = bpmTree[accumulatedTokenIDs], set.contains(newlineToken) { allowed.append(newlineToken) }
            return .whitelist(allowed, uniform: uniform)
        case .captionValue:
            if let v = userMetadata.caption, userFieldTokenQueue.isEmpty, accumulatedValue.isEmpty, let c = injectUserField("caption", v) { return c }
            if captionAfterNewline {
                let top = argMax(scores).item(Int32.self)
                let text = decode(Int(top))
                if let first = text.first, first != " ", first != "\t" {
                    captionAfterNewline = false; captionEnding = true; pendingFieldName = ""
                    return .none
                }
                captionAfterNewline = false
            }
            if captionEnding { return .none }
            if captionTokenCount >= 512 { return .whitelist([newlineToken], uniform: uniform) }
            var blocked = audioCodeTokenIDs
            if backtickToken >= 0 { blocked.append(backtickToken) }
            return .blocklist(blocked)
        case .durationValue:
            if let v = userMetadata.duration, userFieldTokenQueue.isEmpty, accumulatedTokenIDs.isEmpty, let c = injectUserField("duration", v) { return c }
            if let target = targetDuration {
                let digits = Array(String(Int(target)))
                let pos = accumulatedValue.count
                if pos < digits.count, let d = digits[pos].wholeNumberValue, let tok = digitTokens[d] { return .whitelist([tok], uniform: uniform) }
                return .whitelist([newlineToken], uniform: uniform)
            }
            var allowed = Array(durationTree[accumulatedTokenIDs] ?? [])
            if let set = durationTree[accumulatedTokenIDs], set.contains(newlineToken) { allowed.append(newlineToken) }
            return .whitelist(allowed, uniform: uniform)
        case .genresValue:
            // No genres vocabulary is shipped with this port (upstream skips the field); without a vocab upstream blocks only
            // an empty newline — the field is unreachable while `skipGenres` is true.
            return accumulatedValue.trimmingCharacters(in: .whitespaces).isEmpty ? .blocklist([newlineToken]) : .none
        case .keyscaleValue:
            if let v = userMetadata.keyscale, userFieldTokenQueue.isEmpty, accumulatedTokenIDs.isEmpty, let c = injectUserField("keyscale", v) { return c }
            if let set = keyscaleTree[accumulatedTokenIDs], set.contains(newlineToken) { return .whitelist([newlineToken], uniform: uniform) }
            let allowed = Array(keyscaleTree[accumulatedTokenIDs] ?? [])
            return .whitelist(allowed.isEmpty ? [newlineToken] : allowed, uniform: uniform)
        case .languageValue:
            if let v = userMetadata.language, userFieldTokenQueue.isEmpty, accumulatedTokenIDs.isEmpty, let c = injectUserField("language", v) { return c }
            if accumulatedTokenIDs.isEmpty {
                guard let candidates = languageTree[[]], !candidates.isEmpty else { return .whitelist([newlineToken], uniform: uniform) }
                let list = Array(candidates)
                let values = scores[MLXArray(list.map { Int32($0) })].asArray(Float.self)
                var best = 0
                for i in 1 ..< values.count where values[i] > values[best] { best = i }
                return .whitelist([list[best]], uniform: uniform)
            }
            if let set = languageTree[accumulatedTokenIDs], set.contains(newlineToken) { return .whitelist([newlineToken], uniform: uniform) }
            let allowed = Array(languageTree[accumulatedTokenIDs] ?? [])
            return .whitelist(allowed.isEmpty ? [newlineToken] : allowed, uniform: uniform)
        case .timesigValue:
            if let v = userMetadata.timesignature, userFieldTokenQueue.isEmpty, accumulatedTokenIDs.isEmpty, let c = injectUserField("timesignature", v) { return c }
            if let set = timesigTree[accumulatedTokenIDs], set.contains(newlineToken) { return .whitelist([newlineToken], uniform: uniform) }
            return .whitelist(Array(timesigTree[accumulatedTokenIDs] ?? []), uniform: uniform)
        default:
            return uniform ? .whitelist([], uniform: true) : .none
        }
    }

    private func transitionToNextState() {
        guard let next = nextState[state] else { return }
        state = next
        positionInState = 0; accumulatedValue = ""; accumulatedTokenIDs = []
        captionAfterNewline = false; captionTokenCount = 0; captionEnding = false; pendingFieldName = ""
    }

    // MARK: update_state

    public func update(with token: Int) {
        guard enabled, state != .completed else { return }
        if state == .codesGeneration { codesCount += 1; return }
        if !userFieldTokenQueue.isEmpty {
            userFieldTokenQueue.removeFirst()
            if userFieldTokenQueue.isEmpty {
                let field = currentUserField ?? ""
                currentUserField = nil
                state = nextFieldState(after: field)
                positionInState = 0; accumulatedValue = ""; accumulatedTokenIDs = []
            }
            return
        }
        let text = decode(token)
        if let fixed = fixedStrings[state] {
            positionInState += text.unicodeScalars.count
            if positionInState >= fixed.unicodeScalars.count {
                if state == .thinkEndTag && stopAtReasoning {
                    state = .completed; positionInState = 0; accumulatedValue = ""; accumulatedTokenIDs = []
                } else { transitionToNextState() }
            }
            return
        }
        switch state {
        case .bpmValue, .durationValue, .timesigValue:
            if token == newlineToken { transitionToNextState(); return }
            accumulatedTokenIDs.append(token)
            let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !stripped.isEmpty, stripped.allSatisfy(\.isNumber) { accumulatedValue += stripped }
        case .genresValue:
            if token == newlineToken { transitionToNextState(); return }
            accumulatedValue += text
        case .captionValue:
            captionTokenCount += 1
            accumulatedValue += text
            captionAfterNewline = text.contains("\n")
            if captionEnding {
                pendingFieldName += text
                if text.contains(":") {
                    let name = pendingFieldName.trimmingCharacters(in: .whitespacesAndNewlines)
                    var field = name
                    while field.hasSuffix(":") { field.removeLast() }
                    field = field.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    let toValue: [String: FSMState] = ["duration": .durationValue, "genres": .genresValue, "keyscale": .keyscaleValue,
                                                       "language": .languageValue, "timesignature": .timesigValue]
                    if let s = toValue[field] {
                        state = s; positionInState = 0; accumulatedValue = ""; accumulatedTokenIDs = []; captionEnding = false; pendingFieldName = ""
                    } else {
                        captionEnding = false; pendingFieldName = ""
                        transitionToNextState()
                    }
                }
            }
        case .keyscaleValue, .languageValue:
            if token == newlineToken { transitionToNextState(); return }
            accumulatedTokenIDs.append(token)
            accumulatedValue += text
        default: break
        }
    }

    /// The codes-phase masks, built once: `non_audio_code_mask` with EOS blocked / allowed (the native fast path upstream).
    lazy var codesMaskNoEOS: MLXArray = {
        var m = nonAudioCodeMask.asArray(Float.self); m[eosTokenID] = -.infinity; let a = MLXArray(m); eval(a); return a
    }()
    lazy var eosOnlyMask: MLXArray = {
        var m = [Float](repeating: -.infinity, count: vocabularySize); m[eosTokenID] = 0; let a = MLXArray(m); eval(a); return a
    }()

    /// The constraint applied to fp32 logits `[V]`.
    public func apply(_ constraint: TokenConstraint, to logits: MLXArray) -> MLXArray {
        if state == .codesGeneration {           // precomputed masks: the per-step host array would cost more than the forward
            switch constraint {
            case .whitelist(let ids, _) where ids.count == audioCodeTokenIDs.count: return logits + codesMaskNoEOS
            case .whitelist(let ids, _) where ids.count == audioCodeTokenIDs.count + 1: return logits + nonAudioCodeMask
            case .whitelist(let ids, _) where ids == [eosTokenID]: return logits + eosOnlyMask
            default: break
            }
        }
        switch constraint {
        case .none: return logits
        case .blocklist(let ids):
            var mask = [Float](repeating: 0, count: vocabularySize)
            for id in ids { mask[id] = -.infinity }
            return logits + MLXArray(mask)
        case .whitelist(let ids, let uniform):
            var mask = [Float](repeating: -.infinity, count: vocabularySize)
            for id in ids { mask[id] = 0 }
            return uniform ? MLXArray(mask) : logits + MLXArray(mask)
        }
    }

    /// The set a constraint lets through (for the K lane and for tests).
    public func allowedSet(_ constraint: TokenConstraint) -> [Int] {
        switch constraint {
        case .none: return Array(0 ..< vocabularySize)
        case .whitelist(let ids, _): return Array(Set(ids)).sorted()
        case .blocklist(let ids): let b = Set(ids); return (0 ..< vocabularySize).filter { !b.contains($0) }
        }
    }
}
