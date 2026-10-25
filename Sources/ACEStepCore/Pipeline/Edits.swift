// Edits.swift — the source-audio tasks' geometry as `AceStepHandler` computes it: `prepare_padding_info` (a repaint range
// before the source pads it left, past its end pads it right — that is how upstream extends), `_build_chunk_masks_and_src_latents`
// (the explicit 0 / 1 chunk mask over the range, the source latents with the range replaced by the silence latent, the repaint
// mask), `_resolve_repaint_config("balanced", 0.5)` (injection ratio 0.5, 12 latent frames and 25 ms of crossfade) and
// `apply_repaint_waveform_splice` (the original source outside the range after the VAE, 25 ms ramps). A two-sided extension is
// the union of two ranges — upstream's single range generalised per contiguous run (declared divergence 4, MEASUREMENTS §V3b).
import Foundation
import MLX

/// What to do with a source clip (`MusicEditOperation`'s cases, in the Core's words).
public enum ACEStepEditKind: Sendable, Equatable {
    /// `task_type = "cover"`: the source re-performed; `strength` ∈ [0, 1] is `audio_cover_strength`.
    case cover(strength: Double)
    /// `task_type = "repaint"` over `[start, end]` seconds of the source.
    case repaint(start: Double, end: Double)
    /// `task_type = "repaint"` with the range outside the source: `leading` seconds prepended, `trailing` appended.
    case extend(leading: Double, trailing: Double)
}

public struct ACEStepEdit: Sendable {
    public var source: AudioClip
    public var kind: ACEStepEditKind
    public init(source: AudioClip, kind: ACEStepEditKind) { self.source = source; self.kind = kind }
}

/// `_resolve_repaint_config(mode, strength)` — the pipeline's defaults ("balanced", 0.5).
public struct RepaintConfig: Sendable, Equatable {
    public var injectionRatio: Double
    public var crossfadeFrames: Int
    public var waveformCrossfadeSeconds: Double
    public static let balanced = RepaintConfig(mode: "balanced", strength: 0.5)
    public init(mode: String, strength: Double) {
        let s = max(0, min(1, strength))
        switch mode {
        case "aggressive": injectionRatio = 0; crossfadeFrames = 0; waveformCrossfadeSeconds = 0
        case "conservative": injectionRatio = 1; crossfadeFrames = 25; waveformCrossfadeSeconds = 0.05
        default:
            let inv = 1 - s
            injectionRatio = inv; crossfadeFrames = Int((25 * inv).rounded(.toNearestOrEven)); waveformCrossfadeSeconds = 0.05 * inv
        }
    }
}

/// The padded source and the ranges to regenerate, in seconds of the padded clip.
public struct EditGeometry: Sendable, Equatable {
    public var leadingPadSeconds: Double
    public var trailingPadSeconds: Double
    /// `[start, end)` seconds of the PADDED clip to regenerate; empty for a cover.
    public var ranges: [(start: Double, end: Double)]
    public var sourceSeconds: Double
    public var paddedSeconds: Double { leadingPadSeconds + sourceSeconds + trailingPadSeconds }
    public var isCover: Bool { ranges.isEmpty }

    public static func == (a: EditGeometry, b: EditGeometry) -> Bool {
        a.leadingPadSeconds == b.leadingPadSeconds && a.trailingPadSeconds == b.trailingPadSeconds && a.sourceSeconds == b.sourceSeconds
            && a.ranges.map(\.start) == b.ranges.map(\.start) && a.ranges.map(\.end) == b.ranges.map(\.end)
    }

    /// `prepare_padding_info` for one item (`repainting_end` < 0 / nil = the source's end).
    public static func make(sourceSeconds d: Double, kind: ACEStepEditKind) -> EditGeometry {
        switch kind {
        case .cover:
            return EditGeometry(leadingPadSeconds: 0, trailingPadSeconds: 0, ranges: [], sourceSeconds: d)
        case .repaint(let start, let end):
            let actualEnd = end < 0 ? d : end
            let left = max(0, -start), right = max(0, actualEnd - d)
            return EditGeometry(leadingPadSeconds: left, trailingPadSeconds: right, ranges: [(start + left, actualEnd + left)], sourceSeconds: d)
        case .extend(let leading, let trailing):
            var ranges: [(start: Double, end: Double)] = []
            if leading > 0 { ranges.append((0, leading)) }
            if trailing > 0 { ranges.append((leading + d, leading + d + trailing)) }
            return EditGeometry(leadingPadSeconds: max(0, leading), trailingPadSeconds: max(0, trailing), ranges: ranges, sourceSeconds: d)
        }
    }

    /// The zero padding `prepare_padding_info` applies (`int(duration · 48000)` samples each side).
    public func pad(_ source: MLXArray, sampleRate: Int) -> MLXArray {
        let l = Int(leadingPadSeconds * Double(sampleRate)), r = Int(trailingPadSeconds * Double(sampleRate))
        var parts: [MLXArray] = []
        if l > 0 { parts.append(MLXArray.zeros([l, source.dim(1)], dtype: source.dtype)) }
        parts.append(source)
        if r > 0 { parts.append(MLXArray.zeros([r, source.dim(1)], dtype: source.dtype)) }
        return parts.count == 1 ? source : concatenated(parts, axis: 0)
    }

    /// `_build_chunk_masks_and_src_latents`'s latent ranges: `int(sec · 48000 // 1920)`, start clamped to `[0, T − 1]`, end to
    /// `[start + 1, T]`.
    public func latentRanges(frames t: Int, sampleRate: Int, hop: Int) -> [Range<Int>] {
        ranges.map { r in
            var s = Int((r.start * Double(sampleRate) / Double(hop)).rounded(.down))
            var e = Int((r.end * Double(sampleRate) / Double(hop)).rounded(.down))
            s = max(0, min(s, t - 1)); e = max(s + 1, min(e, t))
            return s ..< e
        }
    }
}

public enum RepaintMasks {
    /// (`chunk_mask` `[1, T]` 0 / 1 explicit, `src_latents` with the ranges replaced by the silence latent, `repaint_mask` `[1, T]`).
    public static func build(target: MLXArray, silence: MLXArray, ranges: [Range<Int>]) -> (chunkMask: MLXArray, sourceLatents: MLXArray, repaintMask: MLXArray) {
        let t = target.dim(1)
        var mask = [Bool](repeating: false, count: t)
        for r in ranges { for i in r { mask[i] = true } }
        let m = MLXArray(mask, [1, t])
        let sil = silence[..<t].asType(target.dtype).expandedDimensions(axis: 0)
        let src = MLX.where(m.expandedDimensions(axis: -1), sil, target)
        return (m.asType(target.dtype), src, m)
    }
}

public enum WaveformSplice {
    /// `_build_waveform_crossfade_mask` per range, union over the ranges.
    static func mask(samples: Int, ranges: [(Int, Int)], crossfade: Int) -> [Float] {
        var m = [Float](repeating: 0, count: samples)
        for (start, end) in ranges {
            for i in start ..< end { m[i] = 1 }
            if crossfade > 0 {
                let fs = max(start - crossfade, 0)
                if start - fs > 0 { let n = start - fs; for k in 0 ..< n { m[fs + k] = max(m[fs + k], Float(Double(k + 1) / Double(n + 1))) } }
                let fe = min(end + crossfade, samples)
                if fe - end > 0 { let n = fe - end; for k in 0 ..< n { m[end + k] = max(m[end + k], Float(1 - Double(k + 1) / Double(n + 1))) } }
            }
        }
        return m
    }

    /// `apply_repaint_waveform_splice`: `pred` `[S, C]` and the (padded) `source` `[S', C]` → the source outside the ranges, the
    /// prediction inside, linear ramps of `crossfadeSeconds` at each boundary; samples past the shorter of the two are the
    /// prediction's.
    public static func apply(pred: MLXArray, source: MLXArray, ranges: [(start: Double, end: Double)], sampleRate: Int, crossfadeSeconds: Double) -> MLXArray {
        let minSamples = min(pred.dim(0), source.dim(0))
        var sampleRanges: [(Int, Int)] = []
        for r in ranges {
            var s = Int(r.start * Double(sampleRate)), e = Int(r.end * Double(sampleRate))
            s = max(0, min(s, minSamples)); e = max(s, min(e, minSamples))
            if s == 0 && e >= minSamples { return pred }
            sampleRanges.append((s, e))
        }
        if sampleRanges.isEmpty { return pred }
        let m = MLXArray(mask(samples: minSamples, ranges: sampleRanges, crossfade: Int(crossfadeSeconds * Double(sampleRate))), [minSamples, 1]).asType(pred.dtype)
        let head = m * pred[..<minSamples] + (1 - m) * source[..<minSamples].asType(pred.dtype)
        return pred.dim(0) > minSamples ? concatenated([head, pred[minSamples...]], axis: 0) : head
    }
}
