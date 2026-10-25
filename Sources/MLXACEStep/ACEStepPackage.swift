import Foundation
import MLX
import MLXToolKit
import ACEStepCore

/// ACE-Step 1.5 (ACE Studio + StepFun, MIT) on the canonical `music` surface (contract 1.56.0, AB-A-0202): caption + lyrics
/// (+ a length, a seed, bpm / key / time-signature anchors, a vocal language) → one 48 kHz stereo `.wav` song, the anchors the
/// planner settled on, and whether the plan was cut short. The permissive-only Studio's DEFAULT music offer (E25, AB-T-0352).
///
/// Engine-owned lifecycle (C13): the engine constructs from an `ACEStepConfiguration`, materializes the four declared bundles
/// into its store, pages them in with `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// What the surface declares (`MusicControls`): the core (caption, lyrics, durationSeconds 10–600, seed) plus `.anchors`
/// (the planner's bpm / key / time-signature CoT, pinned when given, returned on `MusicResponse.anchors`) and `.vocalLanguage`
/// (the 50 BCP-47 tags the planner knows). NOT declared at V3 — and so refused by the engine before admission — `.referenceAudio`
/// (timbre from a clip), `.sourceAudio` (cover / repaint / extend) and `.timestamps` (LRC): the VAE encoder is converted and
/// loaded but its Swift forward is ungated, and the LRC path is not ported; they arrive as V3b with their own lanes (C11's
/// honest-descriptor rule: a declared feature is an honoured one). No `.planning` / `.score` (the planner emits anchors, not a
/// score) and no `.guidanceScale` (turbo: eight fixed Euler steps).
///
/// Sampling: the planner SAMPLES its plan and codes (temperature 0.85, top-p 0.9, CFG 2.0 against "NO USER INPUT"), so a seed
/// reproduces a song within Swift-MLX, not across to the Python reference; the plan may rewrite the caption (`rewriteCaption`,
/// upstream's `use_cot_caption`) and, when no language is given, picks one.
///
/// `metaData` keys (package-specific, C5):
/// - `temperature` (double, 0.85), `topP` (double, 0.9), `cfgScale` (double, 2.0): the planner's sampling.
/// - `rewriteCaption` (bool, true): let the planner's CoT rewrite the caption for the DiT (upstream's default).
/// - `shift` (double, 1.0): the sampler's timestep shift (1 = the pipeline default's linspace schedule, 3 = the turbo table).
@InferenceActor
public final class ACEStepPackage: ModelPackage {
    public typealias Configuration = ACEStepConfiguration

    /// Footprints MEASURED as phys_footprint THROUGH THE ENGINE (`acestep-gates validate`: MLXServeEngine register → prepare →
    /// two 120 s songs → evict, M5 Max, 2026-10-25, V3 — WIP/acestep-port/MEASUREMENTS.md §V3), pool-inclusive as the fleet
    /// declares (AB-L-0113 / AB-L-0155), VAE chunk 512: bf16 10.54 GB after prepare / 10.9 after a song, peak 17.2–18.0 GB over a
    /// 120 s song (activation 6.7–7.4 GB — the Oobleck decoder's final 128-channel stage, not the DiT or the planner; a 128-frame
    /// window saves only 1.4 GB at twice the decode time); int8 9.34 / 9.60 GB, peak 16.3; the 0.6B planner 8.15 / 8.53 GB, peak
    /// 15.4 (`residentBytesHint`). fp32 is the parity precision: 22.4 GB phys in the V2 lane run (not an offered tier; the figure
    /// is the lane's, not an engine measurement). Declared with headroom.
    nonisolated static let bf16ResidentBytes: UInt64 = 11_000_000_000
    nonisolated static let int8ResidentBytes: UInt64 = 10_000_000_000
    nonisolated static let litePlannerResidentBytes: UInt64 = 8_700_000_000
    nonisolated static let fp32ResidentBytes: UInt64 = 23_000_000_000
    nonisolated static let peakActivationBytes: UInt64 = 8_000_000_000

    /// The planner's language vocabulary (acestep/constants.py) minus its `unknown` sentinel — what `vocalLanguage` admits.
    public nonisolated static let vocalLanguages: [String] = ACEStepConstants.validLanguages.filter { $0 != "unknown" }
    public nonisolated static let maxDurationSeconds: Double = Double(ACEStepConstants.durationRange.upperBound)
    public nonisolated static let minDurationSeconds: Double = Double(ACEStepConstants.durationRange.lowerBound)

    public nonisolated static var controls: MusicControls {
        MusicControls(features: [.anchors, .vocalLanguage], vocalLanguages: vocalLanguages, maxDurationSeconds: maxDurationSeconds)
    }

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the ACE-Step checkpoints (turbo DiT, planner fine-tune, VAE, silence latent) are MIT — the GitHub LICENSE
            // (Copyright (c) 2026 ACEStep) is the controlling text, the HF repo has no LICENSE file (note 1); the Qwen3 bases
            // (Qwen3-Embedding-0.6B, the planner's Qwen3 base) are Apache-2.0. C8: port code MIT.
            license: LicenseDeclaration(
                weightLicense: .mit, additionalWeightLicenses: [.apache2], portCodeLicense: .mit,
                notices: [
                    LicenseNotice(text: "ACE-Step 1.5 weights are MIT per the ace-step/ACE-Step-1.5 GitHub LICENSE and the Hugging Face model card; "
                                  + "the Hugging Face repository (ACE-Step/Ace-Step1.5) ships no LICENSE file of its own.",
                                  display: .informational, license: .mit, termsURL: "https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE"),
                    LicenseNotice(text: "The Oobleck VAE (ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format, MIT) is Stable Audio Open's architecture "
                                  + "retargeted to 48 kHz; whether its weights were initialised from Stability's community-licensed VAE is stated nowhere upstream.",
                                  display: .informational, license: .mit),
                    LicenseNotice(text: "The planner may rewrite the caption and choose the vocal language, bpm, key and time signature it was not given; "
                                  + "the settled values come back on the response's anchors.",
                                  display: .informational),
                ]),
            provenance: Provenance(sourceRepo: "ACE-Step/Ace-Step1.5", revision: "19671f406d603126926c1b7e2adc169acbcade22", tier: 3),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes, peakActivationBytes: peakActivationBytes),
                    QuantFootprint(quant: .int8, residentBytes: int8ResidentBytes, peakActivationBytes: peakActivationBytes),
                    QuantFootprint(quant: .fp32, residentBytes: fp32ResidentBytes, peakActivationBytes: peakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                MusicContract.descriptor(
                    name: "ace-step-1.5",
                    summary: "ACE-Step 1.5 (ACE Studio + StepFun, MIT) song generator (.wav, 48 kHz stereo, 10–600 s): a caption (genre, mood, "
                        + "instrumentation, vocal) + lyrics with [Verse] / [Chorus] / [Bridge] / [Outro] tags (an [Instrumental] tag or empty lyrics "
                        + "for no vocals) → a full song with sung lyrics in order, in about a fifth of its length on an M5 Max. A 5 Hz planner LM "
                        + "writes bpm / key / time signature (pin any of them with anchors; the settled values come back) and the vocal language "
                        + "(50 BCP-47 tags: en, zh, ja, ko, es, fr, de, …) before a 2 B flow DiT renders and the Oobleck VAE decodes. The planner "
                        + "may rephrase the caption. metaData: temperature (0.85) / topP (0.9) / cfgScale (2.0) / rewriteCaption (true) / shift (1.0).",
                    controls: Self.controls
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: ACEStepPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard pipeline == nil else { return }
        guard Configuration.offeredQuants.contains(configuration.quant) else {
            throw ACEStepPackageError.unsupportedQuant("\(configuration.quant) — this package offers \(Configuration.offeredQuants.map { "\($0)" }.joined(separator: " / ")) (V2b: the planner breaks below 8 bits)")
        }
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw ACEStepPackageError.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let bundles = configuration.resolvedBundles(storeRoot: storeRoot) else {
            throw ACEStepPackageError.missingWeights("unresolved bundle directories (no store root)")
        }
        // Weight loads ride the CPU stream inside the Core (WeightIO.load) — the Metal-watchdog rule.
        pipeline = try ACEStepPipeline(bundles: bundles, dtype: configuration.dtype, plannerBits: configuration.plannerBits)
        MLX.Memory.clearCache()
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: every planner token, every DiT step, every VAE
        // window (the Core's `checkpoint` closure); the CancellationError is rethrown UNCHANGED.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .music, let req = request as? MusicRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let plan = try Self.plan(req, vaeChunk: configuration.vaeChunkFrames)
        try Task.checkCancellation()
        let song: ACEStepSong
        do {
            song = try pipeline.generate(plan.request, progress: { phase, done, total in
                switch phase {
                case "plan": RunProgress.report(.plan, step: done, totalSteps: total, stage: 1, totalStages: 2)
                case "codes": RunProgress.report(.plan, step: done, totalSteps: total, stage: 2, totalStages: 2)
                case "denoise": RunProgress.report(.denoise, step: done, totalSteps: total)
                case "decode": RunProgress.report(.decode, step: done, totalSteps: total)
                default: break
                }
            }, checkpoint: { try Task.checkCancellation() })
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ACEStepError {
            throw ACEStepPackageError.generationFailed(error.description)
        }
        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        let wav = ACEStepAudioIO.encodeWAV16(interleaved: song.audio, channels: song.channels, sampleRate: song.sampleRate)
        return MusicResponse(audio: Audio(format: .wav, data: wav, sampleRate: song.sampleRate, channels: song.channels),
                             truncated: song.truncated, anchors: Self.anchors(from: song.metadata, request: plan.request))
    }

    /// What the request asks for, judged against the declaration (the engine admits only declared features and listed
    /// languages; a direct caller meets the same rules here) and against the planner's own vocabularies (bpm 30–300, the 70
    /// keyscales, time signatures 2 / 3 / 4 / 6 over 4), mapped onto the Core's request.
    public struct Plan: Equatable {
        public let request: ACEStepRequest
        public static func == (a: Plan, b: Plan) -> Bool {
            a.request.caption == b.request.caption && a.request.lyrics == b.request.lyrics && a.request.language == b.request.language
                && a.request.duration == b.request.duration && a.request.bpm == b.request.bpm && a.request.keyscale == b.request.keyscale
                && a.request.timesignature == b.request.timesignature && a.request.seed == b.request.seed
        }
    }

    public nonisolated static func plan(_ req: MusicRequest, vaeChunk: Int = 512) throws -> Plan {
        if let refusal = controls.refusal(for: req) { throw PackageError.unsupportedRequestFeature(refusal) }
        let caption = req.caption.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !caption.isEmpty else { throw PackageError.unsupportedRequestFeature("empty caption — the planner and the DiT condition on it") }
        var lyrics = req.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        if lyrics.isEmpty { lyrics = "[Instrumental]" }           // upstream's instrumental convention
        if let d = req.durationSeconds {
            guard d >= minDurationSeconds else {
                throw PackageError.unsupportedRequestFeature(String(format: "durationSeconds(%g) — this surface renders at least %g s", d, minDurationSeconds))
            }
        }
        var language: String? = nil
        if let l = req.vocalLanguage?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            guard let canonical = vocalLanguages.first(where: { $0 == l }) else {
                throw PackageError.unsupportedRequestFeature("vocalLanguage(\"\(l)\") — not one of the \(vocalLanguages.count) tags the planner knows")
            }
            language = canonical
        }
        var r = ACEStepRequest(caption: caption, lyrics: lyrics, language: language, duration: req.durationSeconds,
                               seed: req.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max)))
        r.vaeChunk = vaeChunk
        if let a = req.anchors, !a.isEmpty {
            if let bpm = a.bpm {
                let b = Int(bpm.rounded())
                guard ACEStepConstants.bpmRange.contains(b), bpm == bpm.rounded() || abs(bpm - Double(b)) < 0.5 else {
                    throw PackageError.unsupportedRequestFeature(String(format: "anchors.bpm(%g) — the planner takes whole bpm in %d…%d", bpm, ACEStepConstants.bpmRange.lowerBound, ACEStepConstants.bpmRange.upperBound))
                }
                r.bpm = b
            }
            if let key = a.key {
                guard let canonical = canonicalKeyscale(key) else {
                    throw PackageError.unsupportedRequestFeature("anchors.key(\"\(key)\") — the planner spells keys as \"C major\" / \"F# minor\" / \"Bb minor\" (7 notes × ♮ # b × major / minor)")
                }
                r.keyscale = canonical
            }
            if let ts = a.timeSignature {
                guard let n = timeSignatureNumerator(ts) else {
                    throw PackageError.unsupportedRequestFeature("anchors.timeSignature(\"\(ts)\") — the planner takes 2/4, 3/4, 4/4 or 6/4 (or the bare numerator)")
                }
                r.timesignature = n
            }
        }
        let meta = req.metaData
        if let t = metaDouble(meta, "temperature") { guard t > 0 else { throw PackageError.unsupportedRequestFeature("temperature must be positive") }; r.planner.temperature = Float(t) }
        if let p = metaDouble(meta, "topP") { guard p > 0, p <= 1 else { throw PackageError.unsupportedRequestFeature("topP must be in (0, 1]") }; r.planner.topP = p < 1 ? Float(p) : nil }
        if let c = metaDouble(meta, "cfgScale") { guard c >= 1 else { throw PackageError.unsupportedRequestFeature("cfgScale must be ≥ 1") }; r.planner.cfgScale = Float(c) }
        if let rewrite = metaBool(meta, "rewriteCaption") { r.planner.useCotCaption = rewrite }
        if let s = metaDouble(meta, "shift") { guard s > 0 else { throw PackageError.unsupportedRequestFeature("shift must be positive") }; r.sampler.shift = s }
        return Plan(request: r)
    }

    /// "f# minor" / "Bb Minor" / "C" → the planner's spelling ("F# minor"; a bare note is major). Nil when it is not one of the 70.
    nonisolated static func canonicalKeyscale(_ text: String) -> String? {
        var parts = text.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard !parts.isEmpty, parts.count <= 2 else { return nil }
        if parts.count == 1 { parts.append("major") }
        let note = parts[0].prefix(1).uppercased() + parts[0].dropFirst()
        let mode = parts[1].lowercased()
        let candidate = "\(note) \(mode)"
        return ACEStepConstants.validKeyscales.first { $0 == candidate }
    }

    /// "4/4" → 4, "6" → 6; nil for anything but the planner's 2 / 3 / 4 / 6 over 4.
    nonisolated static func timeSignatureNumerator(_ text: String) -> Int? {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = t.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard let n = Int(parts[0]), ACEStepConstants.validTimeSignatures.contains(n) else { return nil }
        if parts.count == 2 { guard parts[1] == "4" else { return nil } }
        else if parts.count > 2 { return nil }
        return n
    }

    /// The anchors the planner settled on (its CoT, overridden by whatever the request pinned), as the response reports them.
    nonisolated static func anchors(from m: PlannerMetadata, request r: ACEStepRequest) -> MusicAnchors? {
        let bpm = r.bpm ?? m.bpm
        let key = r.keyscale ?? m.keyscale.flatMap { $0 == "N/A" ? nil : $0 }
        let ts = r.timesignature.map(String.init) ?? m.timesignature.flatMap { $0 == "N/A" ? nil : $0 }
        let a = MusicAnchors(bpm: bpm.map(Double.init), key: key, timeSignature: ts.map { $0.contains("/") ? $0 : "\($0)/4" })
        return a.isEmpty ? nil : a
    }

    nonisolated static func metaDouble(_ meta: MetaData, _ key: String) -> Double? {
        switch meta[key] {
        case .double(let v)?: return v
        case .int(let v)?: return Double(v)
        case .string(let s)?: return Double(s)
        default: return nil
        }
    }
    nonisolated static func metaBool(_ meta: MetaData, _ key: String) -> Bool? {
        switch meta[key] {
        case .bool(let v)?: return v
        case .int(let v)?: return v != 0
        case .string(let s)?: return ["true", "1", "yes"].contains(s.lowercased()) ? true : (["false", "0", "no"].contains(s.lowercased()) ? false : nil)
        default: return nil
        }
    }
}

extension ACEStepPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(ACEStepPackage.self) }
}

/// Wrapper-level errors (weight resolution, tier, a Core failure). Runtime request errors use `PackageError`.
public enum ACEStepPackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    case unsupportedQuant(String)
    case generationFailed(String)
    public var description: String {
        switch self {
        case .missingWeights(let s): return "ACE-Step 1.5: \(s)"
        case .unsupportedQuant(let s): return "ACE-Step 1.5: unsupported quant \(s)"
        case .generationFailed(let s): return "ACE-Step 1.5: \(s)"
        }
    }
}
