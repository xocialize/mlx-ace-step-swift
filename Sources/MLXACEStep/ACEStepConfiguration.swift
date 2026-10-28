// ACEStepConfiguration.swift — init-time configuration for `ACEStepPackage` (C9). Per-request caption / lyrics / length /
// seed / anchors / language ride the canonical `MusicRequest`.
//
// FOUR weight sources back the model, one per V1 bundle (`Tools/convert/convert.py`): the turbo DiT + condition encoder + FSQ
// detokenizer (`ACE-Step-1.5-turbo-bf16`, 4.5 GB), the 5 Hz planner (`ACE-Step-1.5-planner-1.7B-bf16`, 3.5 GB — or the 0.6B
// lite planner, 1.3 GB), the Qwen3-Embedding-0.6B text encoder in mlx-lm layout (1.1 GB) and the Oobleck VAE with weight norm
// folded (`ACE-Step-1.5-vae-fp32`, 0.6 GB). Tiers: `.int8` (the DEFAULT since V4b — the V2b gate's best planner tier, PER 0.108)
// loads the pre-quantised `ACE-Step-1.5-planner-1.7B-8bit` bundle (1.9 GB; affine, group 64, the layers' Linears, bit-equal to the
// quantise-at-load of the bf16 planner) — a local bf16 planner is still quantised at load; `.bf16` loads the bundles as shipped;
// `.fp32` upcasts everything (the parity lane, not a published tier). The VAE is fp32 on every tier.

import Foundation
import MLX
import MLXToolKit
import ACEStepCore

/// Which planner LM plans the song: the 1.7B (the default, V0's gate) or the 0.6B lite planner (the 16 GB tier, admitted by
/// the V2b `L` rule).
public enum ACEStepPlannerModel: String, Sendable, Codable, CaseIterable {
    case standard1_7B = "1.7B"
    case lite0_6B = "0.6B"
}

public struct ACEStepConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// `.int8` (the default: the pre-quantised 8-bit planner), `.bf16` (the planner as shipped), `.fp32` (parity work; upcasts the bundles).
    public var quant: Quant
    /// The planner LM.
    public var planner: ACEStepPlannerModel
    /// Pinned revision for every source; nil = main.
    public var revision: String?
    /// Explicit bundle root in the converter's layout (the four bundle folders side by side) — the dev escape hatch, never
    /// touches the network.
    public var bundleDirectory: URL?
    /// Explicit planner bundle (overrides the folder under `bundleDirectory`; the 0.6B converter output, for instance).
    public var plannerDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Oobleck decode window in latent frames (25 Hz). 512 is the memory knob V0 measured (18 GB working set vs 34 GB at 2048)
    /// and the value every footprint below was measured at; the decode is exact at any window (V2 lane D).
    public var vaeChunkFrames: Int

    public init(quant: Quant = .int8, planner: ACEStepPlannerModel = .standard1_7B, revision: String? = nil,
                bundleDirectory: URL? = nil, plannerDirectory: URL? = nil, modelsRootDirectory: URL? = nil, vaeChunkFrames: Int = 512) {
        self.quant = quant; self.planner = planner; self.revision = revision
        self.bundleDirectory = bundleDirectory; self.plannerDirectory = plannerDirectory; self.modelsRootDirectory = modelsRootDirectory
        self.vaeChunkFrames = vaeChunkFrames
    }

    // MARK: repos (the V4 publication targets; the names V3 declares)

    public static let turboRepo = "mlx-community/ACE-Step-1.5-turbo-bf16"
    public static let plannerRepo = "mlx-community/ACE-Step-1.5-planner-1.7B-bf16"
    /// The pre-quantised planner (V4b): affine 8-bit, group 64, the layers' Linears; the `.int8` tier's source.
    public static let plannerInt8Repo = "mlx-community/ACE-Step-1.5-planner-1.7B-8bit"
    public static let plannerLiteRepo = "mlx-community/ACE-Step-1.5-planner-0.6B-bf16"
    /// Qwen3-Embedding-0.6B in mlx-lm's `model.*` layout as the converter writes it (a bare `Qwen3Model` checkpoint that
    /// `mlx_lm.convert` cannot read), published under the ACE-Step name so it never collides with a differently laid-out
    /// community conversion of the same base.
    public static let textEncoderRepo = "mlx-community/ACE-Step-1.5-text-encoder-bf16"
    public static let vaeRepo = "mlx-community/ACE-Step-1.5-vae-fp32"

    /// The planner repo per tier: the lite planner is bf16 on every tier (quantised at load under `.int8`); the 1.7B planner is the
    /// pre-quantised `-8bit` bundle under `.int8` and the bf16 bundle otherwise.
    public var plannerRepoForTier: String {
        if planner == .lite0_6B { return Self.plannerLiteRepo }
        return quant == .int8 ? Self.plannerInt8Repo : Self.plannerRepo
    }

    /// The Core precision for this tier (nil = the bundles' bf16) and the planner quantisation applied at load.
    var dtype: DType? { quant == .fp32 ? .float32 : nil }
    var plannerBits: Int? {
        switch quant {
        case .int8: return 8
        case .int4: return 4
        default: return nil
        }
    }
    /// The tiers this package offers, the default first (`.int4` loads for measurement work but is refused by `load()` — V2b's Q4 rule).
    public static let offeredQuants: [Quant] = [.int8, .bf16, .fp32]

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case quant, planner, revision, vaeChunkFrames }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decodeIfPresent(Quant.self, forKey: .quant) ?? .int8
        planner = try c.decodeIfPresent(ACEStepPlannerModel.self, forKey: .planner) ?? .standard1_7B
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        vaeChunkFrames = try c.decodeIfPresent(Int.self, forKey: .vaeChunkFrames) ?? 512
    }
}

// MARK: - Footprint hint (the planner axis at the same quant)

extension ACEStepConfiguration: FootprintConfigured {
    /// The 0.6B planner tier shares `.bf16` with the 1.7B default but is 2.4 GB lighter resident (V3 validate: 8.15 GB after prepare,
    /// 8.53 after a song, vs 10.54 / 10.91) — the `QuantFootprint` quant match would charge it the 1.7B figure, so the lite planner
    /// declares its own resident bytes. The activation peak is the same (the VAE's, not the planner's), so no activation hint.
    public var residentBytesHint: UInt64? {
        planner == .lite0_6B && quant != .fp32 ? ACEStepPackage.litePlannerResidentBytes : nil
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension ACEStepConfiguration: WeightSourcing {
    /// The files `ACEStepPipeline.init` opens per bundle — the materialization contract.
    static let turboFiles = ACEStepBundles.turboFiles
    static let qwen3Files = ACEStepBundles.qwen3Files
    static let vaeFiles = ACEStepBundles.vaeFiles
    /// Representative file for the missing-probe (every bundle's largest).
    static let probeFile = "model.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "turbo", repo: Self.turboRepo, revision: revision, matching: Self.turboFiles),
         WeightSource(role: "planner", repo: plannerRepoForTier, revision: revision, matching: Self.qwen3Files),
         WeightSource(role: "text-encoder", repo: Self.textEncoderRepo, revision: revision, matching: Self.qwen3Files),
         WeightSource(role: "vae", repo: Self.vaeRepo, revision: revision, matching: Self.vaeFiles)]
    }

    /// The bundles as the explicit directories name them (nil without `bundleDirectory`).
    var explicitBundles: ACEStepBundles? {
        guard let root = bundleDirectory else {
            return nil
        }
        // under `.int8` the pre-quantised folder when it is there, else the bf16 folder (quantised at load — the dev layout)
        let int8 = root.appendingPathComponent(ACEStepBundles.plannerInt8Name)
        let standard = (quant == .int8 && FileManager.default.fileExists(atPath: int8.appendingPathComponent(Self.probeFile).path)) ? int8 : root.appendingPathComponent(ACEStepBundles.plannerName)
        let plannerDir = plannerDirectory ?? (planner == .lite0_6B ? root.appendingPathComponent(ACEStepBundles.plannerLiteName) : standard)
        return ACEStepBundles.under(root, planner: plannerDir)
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        if let b = explicitBundles {
            let present = [b.turbo, b.planner, b.textEncoder, b.vae].map { fm.fileExists(atPath: $0.appendingPathComponent(Self.probeFile).path) }
            return zip(weightSources, present).filter { !$0.1 }.map(\.0)
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// The four bundle directories `load()` opens: the explicit layout, else each repo's snapshot (hub-client layout) or flat
    /// directory (engine-executed layout) under the store. Nil when nothing resolves.
    public func resolvedBundles(storeRoot: URL?) -> ACEStepBundles? {
        if let b = explicitBundles { return b }
        let store = ModelStore(root: storeRoot)
        func dir(_ repo: String) -> URL? {
            if let snapshot = store.snapshotDirectory(for: repo, revision: revision),
               FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(Self.probeFile).path) { return snapshot }
            if let flat = store.directory(for: repo),
               FileManager.default.fileExists(atPath: flat.appendingPathComponent(Self.probeFile).path) { return flat }
            return nil
        }
        guard let t = dir(Self.turboRepo), let p = dir(plannerRepoForTier), let e = dir(Self.textEncoderRepo), let v = dir(Self.vaeRepo) else { return nil }
        return ACEStepBundles(turbo: t, planner: p, textEncoder: e, vae: v)
    }
}

// MARK: - Cold-start prewarm

extension ACEStepConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let b = resolvedBundles(storeRoot: modelsRootDirectory) else { return [] }
        return [b.turbo, b.planner, b.textEncoder, b.vae].map { $0.appendingPathComponent(Self.probeFile) }
    }
}
