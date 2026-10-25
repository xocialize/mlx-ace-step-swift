// ConformanceTests.swift — the offline conformance gate for MLXACEStep: manifest (C1 / C7 / C8 / C-memory / C11 / provenance),
// the MAT gate per tier (auto-materialization declarations), the CAN gate (pre-cancelled run propagation, checkpoint cadence),
// the unloaded refusal, and the request plane (anchors, languages, lengths, metaData). Nothing here evals a kernel: the live
// run is `acestep-gates validate` (GPU, through MLXServeEngine).

import Foundation
import MLX
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXACEStep
@testable import ACEStepCore

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 — MIT weights with the Apache-2.0 Qwen3 bases declared beside them, MIT port code; the default policy admits.
    func testLicenseDeclaresBothWeightLicencesAndAdmits() {
        let license = ACEStepPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .mit)
        XCTAssertEqual(license.additionalWeightLicenses, [.apache2])
        XCTAssertEqual(license.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(license).isAdmitted)
        // No licence here carries a UI display duty; the three notices are informational (the HF-LICENSE footnote, the VAE
        // provenance footnote, the planner's caption / language drift).
        XCTAssertTrue(license.missingRequiredDisplayNotices.isEmpty)
        XCTAssertTrue(license.requiredNotices.isEmpty)
        XCTAssertEqual(license.notices?.count, 3)
    }

    /// C-memory — split footprints for the offered tiers, at least what `validate` measured through the engine.
    func testFootprintsAreTheMeasuredTiers() {
        let footprints = ACEStepPackage.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.bf16, .int8, .fp32])
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes, 10_910_000_000)   // bf16 phys after a song (V3 validate 10.91 GB)
        XCTAssertGreaterThanOrEqual(footprints[1].residentBytes, 9_600_000_000)    // int8 planner (9.60 GB)
        for f in footprints { XCTAssertGreaterThanOrEqual(f.peakActivationBytes, 7_430_000_000) }   // peak − post-prepare resident over a 120 s song
        XCTAssertTrue(CancellationConformance.longRunImplied(by: ACEStepPackage.manifest))
        // the lite planner is the same quant with a smaller working set: it hints its own resident bytes (FootprintConfigured)
        XCTAssertNil(ACEStepConfiguration().residentBytesHint)
        XCTAssertEqual(ACEStepConfiguration(planner: .lite0_6B).residentBytesHint, ACEStepPackage.litePlannerResidentBytes)
        XCTAssertGreaterThanOrEqual(ACEStepPackage.litePlannerResidentBytes, 8_530_000_000)   // 8.53 GB after a song
        XCTAssertNil(ACEStepConfiguration(quant: .fp32, planner: .lite0_6B).residentBytesHint)
    }

    /// C1 / C11 — the package serves `music` and nothing else, and its surface is born declared and coherent (1.56.0).
    func testCapabilityIsMusicWithAnHonestDeclaration() {
        XCTAssertEqual(Set(ACEStepPackage.manifest.capabilities), [.music])
        let surface = ACEStepPackage.manifest.surfaces.first { $0.capability == .music }
        XCTAssertEqual(surface?.name, "ace-step-1.5")
        let controls = surface?.musicControls
        XCTAssertEqual(controls?.features, [.anchors, .vocalLanguage, .referenceAudio, .sourceAudio, .timestamps])
        XCTAssertEqual(controls?.editModes, [.cover, .repaint, .extend])
        XCTAssertEqual(controls?.vocalLanguages.count, 50)
        XCTAssertTrue(controls?.vocalLanguages.contains("en") ?? false)
        XCTAssertTrue(controls?.vocalLanguages.contains("zh") ?? false)
        XCTAssertFalse(controls?.vocalLanguages.contains("unknown") ?? true)
        XCTAssertEqual(controls?.maxDurationSeconds, 600)
        XCTAssertEqual(controls?.declarationProblems, [])
        // only declared features are advertised as parameters (the `declaring(adapters:)` rule)
        let names = Set(surface?.parameters.map(\.name) ?? [])
        XCTAssertTrue(names.isSuperset(of: ["caption", "lyrics", "durationSeconds", "seed", "anchors", "vocalLanguage", "referenceAudio", "edit"]))
        XCTAssertTrue(names.isDisjoint(with: ["planning", "score", "guidanceScale"]))
    }

    func testProvenancePinsUpstream() {
        XCTAssertEqual(ACEStepPackage.manifest.provenance.sourceRepo, "ACE-Step/Ace-Step1.5")
        XCTAssertEqual(ACEStepPackage.manifest.provenance.revision.count, 40)
        XCTAssertEqual(ACEStepPackage.manifest.provenance.tier, 3)
    }
}

final class MaterializationConformanceTests: XCTestCase {

    /// MAT-1..5: a fresh (dir-less) configuration reports its four sources missing; explicit bundles satisfy — per offered tier.
    func testMaterializationGatePerTier() throws {
        for quant in ACEStepConfiguration.offeredQuants {
            let satisfied = try satisfiedConfiguration(quant: quant)
            defer { try? FileManager.default.removeItem(at: satisfied.bundleDirectory!) }
            let report = MaterializationConformance.check(
                freshConfiguration: ACEStepConfiguration(quant: quant, modelsRootDirectory: emptyStoreRoot()),
                satisfiedConfiguration: satisfied)
            XCTAssertTrue(report.passed, "\(quant): \(report.summary)")
        }
    }

    /// Four roles, the mlx-community repos the V4 publication targets; the lite planner swaps one of them.
    func testPublishedRepos() {
        let std = ACEStepConfiguration()
        XCTAssertEqual(std.weightSources.map(\.role), ["turbo", "planner", "text-encoder", "vae"])
        XCTAssertEqual(std.weightSources.map(\.repo), ["mlx-community/ACE-Step-1.5-turbo-bf16", "mlx-community/ACE-Step-1.5-planner-1.7B-bf16",
                                                       "mlx-community/ACE-Step-1.5-text-encoder-bf16", "mlx-community/ACE-Step-1.5-vae-fp32"])
        XCTAssertEqual(ACEStepConfiguration(planner: .lite0_6B).weightSources[1].repo, "mlx-community/ACE-Step-1.5-planner-0.6B-bf16")
        XCTAssertEqual(ACEStepConfiguration(quant: .int8).weightSources[1].repo, "mlx-community/ACE-Step-1.5-planner-1.7B-bf16")   // quantised at load (V3)
        XCTAssertEqual(std.quant, .bf16)
        XCTAssertEqual(std.planner, .standard1_7B)
        XCTAssertEqual(std.vaeChunkFrames, 512)
    }

    /// The declared file lists are the bundle contract `ACEStepPipeline.init` opens.
    func testDeclaredFilesAreTheBundleContract() {
        XCTAssertEqual(ACEStepConfiguration.turboFiles, ["model.safetensors", "config.json", "silence_latent.safetensors"])
        XCTAssertTrue(ACEStepConfiguration.qwen3Files.contains("tokenizer.json"))
        XCTAssertEqual(ACEStepConfiguration.vaeFiles, ["model.safetensors", "config.json"])
    }

    /// A partially present explicit layout reports exactly the missing roles.
    func testExplicitLayoutReportsMissingRoles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-partial-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        for name in [ACEStepBundles.turboName, ACEStepBundles.vaeName] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
            try Data().write(to: root.appendingPathComponent(name).appendingPathComponent("model.safetensors"))
        }
        XCTAssertEqual(ACEStepConfiguration(bundleDirectory: root).missingWeightSources(storeRoot: nil).map(\.role), ["planner", "text-encoder"])
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = ACEStepConfiguration(quant: .int8, planner: .lite0_6B, bundleDirectory: URL(fileURLWithPath: "/tmp/x"),
                                     plannerDirectory: URL(fileURLWithPath: "/tmp/p"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"), vaeChunkFrames: 256)
        let decoded = try JSONDecoder().decode(ACEStepConfiguration.self, from: JSONEncoder().encode(c))
        XCTAssertNil(decoded.bundleDirectory)
        XCTAssertNil(decoded.plannerDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertEqual(decoded.quant, .int8)
        XCTAssertEqual(decoded.planner, .lite0_6B)
        XCTAssertEqual(decoded.vaeChunkFrames, 256)
        let older = Data(#"{"quant":"bf16"}"#.utf8)
        let old = try JSONDecoder().decode(ACEStepConfiguration.self, from: older)
        XCTAssertEqual(old.planner, .standard1_7B)
        XCTAssertEqual(old.vaeChunkFrames, 512)
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("acestep-empty-store-\(UUID().uuidString)")
    }

    private func satisfiedConfiguration(quant: Quant) throws -> ACEStepConfiguration {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-explicit-\(UUID().uuidString)")
        for name in [ACEStepBundles.turboName, ACEStepBundles.plannerName, ACEStepBundles.textEncoderName, ACEStepBundles.vaeName] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
            try Data().write(to: root.appendingPathComponent(name).appendingPathComponent(ACEStepConfiguration.probeFile))
        }
        return ACEStepConfiguration(quant: quant, bundleDirectory: root)
    }
}

final class CancellationConformanceTests: XCTestCase {

    private func probe() -> MusicRequest {
        MusicRequest(caption: "cancellation probe", lyrics: "[Verse]\nla la la", durationSeconds: 30, seed: 1)
    }

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = ACEStepPackage(configuration: ACEStepConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: probe())
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — checkpoints at every planner token (plan), every DiT step (denoise) and every VAE window (decode), each on the
    /// same seam as the RunProgress report.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: ACEStepPackage.manifest,
            posture: .cadence([.init(phase: .plan, unit: .token, reportsRunProgress: true),
                               .init(phase: .denoise, unit: .step, reportsRunProgress: true),
                               .init(phase: .decode, unit: .chunk, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {

    private func request(duration: Double? = 120, anchors: MusicAnchors? = nil, language: String? = nil, lyrics: String = "[Verse]\nla la la",
                         metaData: MetaData = [:], planning: MusicPlanningMode? = nil, edit: MusicEdit? = nil, reference: Audio? = nil) -> MusicRequest {
        MusicRequest(caption: "A warm piano ballad", lyrics: lyrics, durationSeconds: duration, seed: 7, planning: planning,
                     referenceAudio: reference, edit: edit, anchors: anchors, vocalLanguage: language, metaData: metaData)
    }

    func testUnloadedPackageRejectsLegibly() async {
        let package = ACEStepPackage(configuration: ACEStepConfiguration())
        do {
            _ = try await package.run(request())
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    /// The core maps 1:1 onto the Core's request; empty lyrics become the instrumental tag; a nil language is the planner's.
    func testCoreMapsOntoThePipeline() throws {
        let p = try ACEStepPackage.plan(request(language: "EN"))
        XCTAssertEqual(p.request.caption, "A warm piano ballad")
        XCTAssertEqual(p.request.lyrics, "[Verse]\nla la la")
        XCTAssertEqual(p.request.duration, 120)
        XCTAssertEqual(p.request.seed, 7)
        XCTAssertEqual(p.request.language, "en")
        XCTAssertEqual(p.request.vaeChunk, 512)
        XCTAssertNil(try ACEStepPackage.plan(request()).request.language)
        XCTAssertEqual(try ACEStepPackage.plan(request(lyrics: "  ")).request.lyrics, "[Instrumental]")
        XCTAssertNil(try ACEStepPackage.plan(request(duration: nil)).request.duration)   // the planner picks the length
        XCTAssertEqual(try ACEStepPackage.plan(request(), vaeChunk: 256).request.vaeChunk, 256)
    }

    /// Anchors: whole bpm in 30…300, the planner's 70 keyscales (spelling normalised), N/4 time signatures.
    func testAnchorsMapAndRefuse() throws {
        let p = try ACEStepPackage.plan(request(anchors: MusicAnchors(bpm: 112, key: "f# minor", timeSignature: "6/4")))
        XCTAssertEqual(p.request.bpm, 112)
        XCTAssertEqual(p.request.keyscale, "F# minor")
        XCTAssertEqual(p.request.timesignature, 6)
        XCTAssertEqual(try ACEStepPackage.plan(request(anchors: MusicAnchors(key: "Bb"))).request.keyscale, "Bb major")
        XCTAssertEqual(try ACEStepPackage.plan(request(anchors: MusicAnchors(timeSignature: "3"))).request.timesignature, 3)
        XCTAssertThrowsError(try ACEStepPackage.plan(request(anchors: MusicAnchors(bpm: 900))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(anchors: MusicAnchors(bpm: 112.5))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(anchors: MusicAnchors(key: "H major"))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(anchors: MusicAnchors(key: "C dorian"))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(anchors: MusicAnchors(timeSignature: "5/4"))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(anchors: MusicAnchors(timeSignature: "6/8"))))
        XCTAssertNil(ACEStepPackage.canonicalKeyscale(""))
        XCTAssertEqual(ACEStepPackage.canonicalKeyscale("A♭ Minor"), "A♭ minor")
    }

    /// A 16-bit stereo 48 kHz WAV of `seconds` of a quiet tone — the request-side clip the plane tests map through.
    private func toneWAV(seconds: Double, rate: Int = 48_000, channels: Int = 2) -> Audio {
        let frames = Int(seconds * Double(rate))
        var samples = [Float](repeating: 0, count: frames * channels)
        for i in 0 ..< frames { let v = Float(sin(Double(i) * 2 * .pi * 220 / Double(rate))) * 0.3; for c in 0 ..< channels { samples[i * channels + c] = v } }
        return Audio(format: .wav, data: ACEStepAudioIO.encodeWAV16(interleaved: samples, channels: channels, sampleRate: rate), sampleRate: rate, channels: channels)
    }

    /// V3b: a reference clip and the three edits map onto the Core's request; timestamps are always on; the edit owns the length.
    func testReferenceAndEditsMapOntoThePipeline() throws {
        let clip = toneWAV(seconds: 12)
        let withRef = try ACEStepPackage.plan(request(reference: clip)).request
        XCTAssertEqual(withRef.referenceAudio?.sampleRate, 48_000)
        XCTAssertEqual(withRef.referenceAudio?.channels, 2)
        XCTAssertEqual(withRef.referenceAudio?.frames, 12 * 48_000)
        XCTAssertTrue(withRef.timestamps)
        let cover = try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .cover(strength: 0.5)))).request
        XCTAssertEqual(cover.edit?.kind, .cover(strength: 0.5))
        XCTAssertNil(cover.duration)
        let repaint = try ACEStepPackage.plan(request(duration: 12, edit: MusicEdit(source: clip, operation: .repaint(start: 2, end: 5)))).request
        XCTAssertEqual(repaint.edit?.kind, .repaint(start: 2, end: 5))
        let extend = try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .extend(leading: 3, trailing: 10)))).request
        XCTAssertEqual(extend.edit?.kind, .extend(leading: 3, trailing: 10))
        // a mono 16 kHz clip is taken (the Core folds it to stereo 48 kHz)
        XCTAssertEqual(try ACEStepPackage.plan(request(reference: toneWAV(seconds: 3, rate: 16_000, channels: 1))).request.referenceAudio?.sampleRate, 16_000)
    }

    /// V3b refusals: an unreadable clip, a non-wav, a range outside the source, an empty extension, a strength past 1, a
    /// durationSeconds that contradicts the edit, a result past 600 s.
    func testEditRefusals() {
        let clip = toneWAV(seconds: 12)
        XCTAssertThrowsError(try ACEStepPackage.plan(request(edit: MusicEdit(source: Audio(format: .wav, data: Data([1, 2, 3])), operation: .cover(strength: 0.5)))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(reference: Audio(format: .wav, data: Data([1])))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .repaint(start: 5, end: 20)))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .repaint(start: -2, end: 5)))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .extend(leading: 0, trailing: 0)))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .cover(strength: 1.5)))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: 90, edit: MusicEdit(source: clip, operation: .cover(strength: 1)))))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: nil, edit: MusicEdit(source: clip, operation: .extend(leading: 0, trailing: 600)))))
    }

    /// LRC lines map onto `MusicLyricTimestamp` in order with their ends.
    func testTimestampsMap() {
        let lines = [LyricLineStamp(text: "[Verse]", start: 0, end: 0, confidence: 0), LyricLineStamp(text: "Morning light", start: 0.32, end: 2.0, confidence: 0)]
        let mapped = ACEStepPackage.timestamps(LyricTimestamps(tokens: [], lines: lines, lrc: ""))
        XCTAssertEqual(mapped.map(\.text), ["[Verse]", "Morning light"])
        XCTAssertEqual(mapped[1].start, 0.32)
        XCTAssertEqual(mapped[1].end, 2.0)
    }

    /// Undeclared features and out-of-vocabulary values are refused legibly (the engine refuses them before admission too).
    func testUndeclaredFeaturesAndVocabulariesAreRefused() {
        XCTAssertThrowsError(try ACEStepPackage.plan(request(planning: .full)))
        XCTAssertThrowsError(try ACEStepPackage.plan(MusicRequest(caption: "c", lyrics: "l", score: "X:1")))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(language: "xx")))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: 700)))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(duration: 5)))
        XCTAssertThrowsError(try ACEStepPackage.plan(MusicRequest(caption: "  ", lyrics: "la")))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(metaData: ["temperature": .double(0)])))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(metaData: ["topP": .double(1.5)])))
        XCTAssertThrowsError(try ACEStepPackage.plan(request(metaData: ["cfgScale": .double(0.5)])))
    }

    /// The package levers ride metaData (C5) and default to upstream's values.
    func testMetaDataLevers() throws {
        let d = try ACEStepPackage.plan(request()).request
        XCTAssertEqual(d.planner.temperature, 0.85)
        XCTAssertEqual(d.planner.topP, 0.9)
        XCTAssertEqual(d.planner.cfgScale, 2.0)
        XCTAssertTrue(d.planner.useCotCaption)
        XCTAssertEqual(d.sampler.shift, 1.0)
        let r = try ACEStepPackage.plan(request(metaData: ["temperature": .double(0.7), "topP": .double(1), "cfgScale": .int(3), "rewriteCaption": .bool(false), "shift": .double(3)])).request
        XCTAssertEqual(r.planner.temperature, 0.7)
        XCTAssertNil(r.planner.topP)   // top-p 1 = off
        XCTAssertEqual(r.planner.cfgScale, 3)
        XCTAssertFalse(r.planner.useCotCaption)
        XCTAssertEqual(r.sampler.shift, 3)
    }

    /// The response's anchors are the planner's CoT, overridden by what the request pinned; "N" time signatures read as N/4.
    func testAnchorsReported() {
        var r = ACEStepRequest(caption: "c", lyrics: "l")
        let m = PlannerMetadata(bpm: 83, caption: nil, duration: 120, keyscale: "C major", language: "en", timesignature: "2")
        XCTAssertEqual(ACEStepPackage.anchors(from: m, request: r), MusicAnchors(bpm: 83, key: "C major", timeSignature: "2/4"))
        r.bpm = 112; r.keyscale = "F major"
        XCTAssertEqual(ACEStepPackage.anchors(from: m, request: r), MusicAnchors(bpm: 112, key: "F major", timeSignature: "2/4"))
        XCTAssertNil(ACEStepPackage.anchors(from: PlannerMetadata(keyscale: "N/A", timesignature: "N/A"), request: ACEStepRequest(caption: "c", lyrics: "l")))
    }

    func testWAVEncoding() {
        let data = ACEStepAudioIO.encodeWAV16(interleaved: [0, 0.5, -0.5, 2, -2, .nan], channels: 2, sampleRate: 48_000)
        XCTAssertEqual(data.count, 44 + 12)
        XCTAssertEqual(String(decoding: data[0 ..< 4], as: UTF8.self), "RIFF")
        XCTAssertEqual(data[22], 2)                                  // channels
        XCTAssertEqual(data[24 ..< 28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }, 48_000)
        let pcm = data.dropFirst(44).withUnsafeBytes { raw in raw.bindMemory(to: Int16.self).map { $0 } }
        XCTAssertEqual(pcm, [0, 16383, -16383, 32767, -32767, 0])
    }
}
