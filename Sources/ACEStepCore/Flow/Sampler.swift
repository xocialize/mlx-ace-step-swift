// Sampler.swift — acestep/models/mlx/dit_generate.py `get_timestep_schedule` + `mlx_generate_diffusion` for the turbo model
// (no CFG: `guidance_scale` is ignored on turbo), and dcw_correction_mlx.py's native-Haar DCW. The pipeline's defaults are
// `inference_steps=8`, `shift=1.0` (the GenerationParams default → the linspace schedule [1, .875, …, .125]; shift 3 gives the
// turbo table), `infer_method="ode"`, Euler, DCW double 0.05 / 0.02. Scalars are computed in Double as Python does.
import Foundation
import MLX
import MLXRandom

public struct SamplerSettings: Sendable {
    public var steps: Int = 8
    public var shift: Double = 1.0
    public var timesteps: [Double]? = nil
    public var dcw: Bool = true
    public var dcwMode: String = "double"      // low | high | double | pix
    public var dcwScaler: Double = 0.05
    public var dcwHighScaler: Double = 0.02
    public init() {}
}

public enum Schedule {
    public static let validTimesteps: [Double] = [1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875, 0.8571428571428571, 0.8333333333333334,
                                                 0.7692307692307693, 0.75, 0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454, 0.5, 0.4,
                                                 0.375, 0.3, 0.25, 0.2222222222222222, 0.125]
    public static let shiftTimesteps: [Double: [Double]] = [
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
        3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
    ]
    /// `get_timestep_schedule(shift, timesteps, infer_steps)`.
    public static func timesteps(shift: Double, custom: [Double]?, steps: Int?) -> [Double] {
        if var list = custom {
            while let last = list.last, last == 0 { list.removeLast() }
            if !list.isEmpty {
                return list.prefix(20).map { t in validTimesteps.min { abs($0 - t) < abs($1 - t) }! }
            }
        }
        if let steps, steps > 0 {
            let raw = (0 ..< steps).map { 1.0 - Double($0) / Double(steps) }
            return shift == 1.0 ? raw : raw.map { shift * $0 / (1.0 + (shift - 1.0) * $0) }
        }
        let nearest = [1.0, 2.0, 3.0].min { abs($0 - shift) < abs($1 - shift) }!
        return shiftTimesteps[nearest]!
    }
}

public enum DCW {
    /// `_haar_dwt_1d` along T of `[B, T, C]` (zero-pads odd T).
    static func haar(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = x
        if x.dim(1) % 2 == 1 { x = concatenated([x, MLXArray.zeros([x.dim(0), 1, x.dim(2)], dtype: x.dtype)], axis: 1) }
        let even = x[0..., .stride(from: 0, by: 2), 0...], odd = x[0..., .stride(from: 1, by: 2), 0...]
        let s = Float(1.0 / 2.0.squareRoot())
        return ((even + odd) * s, (even - odd) * s)
    }
    static func inverse(_ low: MLXArray, _ high: MLXArray, length: Int) -> MLXArray {
        let s = Float(1.0 / 2.0.squareRoot())
        let even = (low + high) * s, odd = (low - high) * s
        return stacked([even, odd], axis: 2).reshaped(even.dim(0), -1, even.dim(2))[0..., ..<length, 0...]
    }
    /// `apply_mlx_dcw(x_next, denoised, t_curr, …)`.
    public static func apply(_ x: MLXArray, denoised: MLXArray, t: Double, mode: String, scaler: Double, highScaler: Double) -> MLXArray {
        let lowS = t * scaler, highS = (1.0 - t) * scaler, doubleHighS = (1.0 - t) * highScaler
        switch mode {
        case "pix": return scaler == 0 ? x : x + Float(scaler) * (x - denoised)
        case "low" where lowS == 0, "high" where highS == 0: return x
        case "double" where lowS == 0 && doubleHighS == 0: return x
        default: break
        }
        var (xl, xh) = haar(x)
        let (yl, yh) = haar(denoised)
        switch mode {
        case "low": xl = xl + Float(lowS) * (xl - yl)
        case "high": xh = xh + Float(highS) * (xh - yh)
        case "double":
            if lowS != 0 { xl = xl + Float(lowS) * (xl - yl) }
            if doubleHighS != 0 { xh = xh + Float(doubleHighS) * (xh - yh) }
        default: fatalError("invalid dcw_mode \(mode)")
        }
        return inverse(xl, xh, length: x.dim(1))
    }
}

/// `repaint_mask_np` + `clean_src_latents_np` + `repaint_crossfade_frames` + `repaint_injection_ratio` of `mlx_generate_diffusion`:
/// outside the mask the state is re-set to `t_next·noise + (1 − t_next)·source` for the first `round(ratio · steps)` steps, and
/// the final latents are blended into the source across `crossfadeFrames` on each side of every masked run.
public struct RepaintPlan {
    /// `[1, T]` — true where the model generates, false where the source is kept.
    public var mask: MLXArray
    public var cleanSource: MLXArray
    public var crossfadeFrames: Int
    public var injectionRatio: Double
    public init(mask: MLXArray, cleanSource: MLXArray, crossfadeFrames: Int, injectionRatio: Double) {
        self.mask = mask; self.cleanSource = cleanSource; self.crossfadeFrames = crossfadeFrames; self.injectionRatio = injectionRatio
    }
}

/// `audio_cover_strength` < 1: from step `int(steps · strength)` on, the non-cover conditions (default instruction, silence
/// context) replace the cover ones, with a fresh cross-attention cache.
public struct CoverSwitch {
    public var atStep: Int
    public var cache: CrossAttentionCache
    public var context: MLXArray
    public init(atStep: Int, cache: CrossAttentionCache, context: MLXArray) { self.atStep = atStep; self.cache = cache; self.context = context }
}

public enum Repaint {
    /// `_mlx_repaint_step_injection`.
    static func inject(_ x: MLXArray, source: MLXArray, mask: MLXArray, tNext: Double, noise: MLXArray) -> MLXArray {
        let zt = Float(tNext) * noise.asType(x.dtype) + Float(1.0 - tNext) * source.asType(x.dtype)
        return MLX.where(mask.expandedDimensions(axis: -1), x, zt)
    }

    /// `_mlx_repaint_boundary_blend`'s soft mask, per contiguous masked run (upstream ramps the first and last edge of one
    /// run; for one run this is that mask exactly): linear ramps of `crossfade` frames into the kept region on each side.
    public static func softMask(_ mask: [Bool], crossfade: Int) -> [Float] {
        var soft = mask.map { $0 ? Float(1) : Float(0) }
        let t = mask.count
        guard crossfade > 0, mask.contains(true), mask.contains(false) else { return soft }
        var i = 0
        while i < t {
            guard mask[i] else { i += 1; continue }
            var right = i
            while right < t && mask[right] { right += 1 }
            let left = i
            let fs = max(left - crossfade, 0)
            if left - fs > 0 {
                let n = left - fs
                for k in 0 ..< n { soft[fs + k] = Float(Double(k + 1) / Double(n + 1)) }
            }
            let fe = min(right + crossfade, t)
            if fe - right > 0 {
                let n = fe - right
                for k in 0 ..< n { soft[right + k] = Float(1 - Double(k + 1) / Double(n + 1)) }
            }
            i = right
        }
        return soft
    }

    static func blend(_ x: MLXArray, source: MLXArray, mask: MLXArray, crossfade: Int) -> MLXArray {
        let soft = softMask(mask.asArray(Bool.self), crossfade: crossfade)
        let m = MLXArray(soft, [1, soft.count, 1]).asType(x.dtype)
        return m * x + (1 - m) * source.asType(x.dtype)
    }
}

public enum DiffusionSampler {
    /// `_draw_noise(seed)`: `mx.random.normal((B, T, C), key=mx.random.key(seed))` — bit-identical across the bindings.
    public static func noise(seed: UInt64, frames: Int, channels: Int) -> MLXArray {
        MLXRandom.normal([1, frames, channels], key: MLXRandom.key(seed))
    }

    /// The turbo loop: Euler over the schedule, the final step `x − v·t`, DCW after every step. `noise` is the start state.
    /// `onStep(done, total)` runs after every step; whatever it throws (a cancellation) is rethrown unchanged.
    public static func solve(decoder: AceStepDiTModel, cache: CrossAttentionCache, context: MLXArray, noise: MLXArray,
                             settings: SamplerSettings, repaint: RepaintPlan? = nil, cover: CoverSwitch? = nil,
                             onStep: ((Int, Int) throws -> Void)? = nil) throws -> MLXArray {
        let schedule = Schedule.timesteps(shift: settings.shift, custom: settings.timesteps, steps: settings.steps)
        let dcwActive = settings.dcw && (settings.dcwScaler != 0 || (settings.dcwMode == "double" && settings.dcwHighScaler != 0))
        var x = noise.asType(context.dtype)
        let b = x.dim(0)
        var cache = cache, context = context
        var switched = false
        // `injection_cutoff = round(repaint_injection_ratio * num_steps)` — Python's round, half to even
        let injectionCutoff = repaint.map { Int(($0.injectionRatio * Double(schedule.count)).rounded(.toNearestOrEven)) } ?? 0
        for (i, t) in schedule.enumerated() {
            if let cover, i >= cover.atStep, !switched { switched = true; cache = cover.cache; context = cover.context }
            let tArr = MLXArray(Array(repeating: Float(t), count: b)).asType(x.dtype)
            let v = decoder(x, timestep: tArr, timestepR: tArr, contextLatents: context, cache: cache)
            eval(v)
            let before = x
            if i == schedule.count - 1 {
                x = x - v * MLXArray(Float(t)).asType(x.dtype)
            } else {
                let dt = t - schedule[i + 1]
                x = x - v * MLXArray(Float(dt)).asType(x.dtype)
            }
            eval(x)
            if dcwActive {
                let denoised = before - v * MLXArray(Float(t)).asType(x.dtype)
                x = DCW.apply(x, denoised: denoised, t: t, mode: settings.dcwMode, scaler: settings.dcwScaler, highScaler: settings.dcwHighScaler)
                eval(x)
            }
            if let repaint, i < injectionCutoff {
                let tAfter = i < schedule.count - 1 ? schedule[i + 1] : 0.0
                x = Repaint.inject(x, source: repaint.cleanSource, mask: repaint.mask, tNext: tAfter, noise: noise)
                eval(x)
            }
            try onStep?(i + 1, schedule.count)
        }
        if let repaint, repaint.crossfadeFrames > 0 {
            x = Repaint.blend(x, source: repaint.cleanSource, mask: repaint.mask, crossfade: repaint.crossfadeFrames)
            eval(x)
        }
        return x
    }
}
