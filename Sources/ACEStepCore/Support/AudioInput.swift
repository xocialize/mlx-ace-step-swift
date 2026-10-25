// AudioInput.swift — the loader half of acestep's `IoAudioMixin` (`_normalize_audio_to_stereo_48k`, `is_silence`,
// `process_reference_audio`'s 30 s clip) and the torchaudio resampler it calls (`Resample(sr, 48000)` =
// `_get_sinc_resample_kernel` + `_apply_sinc_resample_kernel`, sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99: the
// kernel in Double as torch builds it in float64, the convolution in fp32 as torch runs it). Declared divergence (V3b):
// the reference clip's three segment offsets come from a SplitMix64 seeded by the request seed, not Python's unseeded `random`.
import Foundation
import MLX

/// Host-side PCM: `frames × channels` interleaved, any rate.
public struct AudioClip: Sendable, Equatable {
    public var interleaved: [Float]
    public var channels: Int
    public var sampleRate: Int
    public init(interleaved: [Float], channels: Int, sampleRate: Int) {
        self.interleaved = interleaved; self.channels = channels; self.sampleRate = sampleRate
    }
    public var frames: Int { channels > 0 ? interleaved.count / channels : 0 }
    public var seconds: Double { sampleRate > 0 ? Double(frames) / Double(sampleRate) : 0 }
}

/// torchaudio `functional.resample` with its defaults.
public enum Resampler {
    public static let lowpassFilterWidth = 6
    public static let rolloff = 0.99

    static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

    /// `_get_sinc_resample_kernel` → (kernel `[new, K, 1]` fp32 in MLX's conv layout, width, reduced orig, reduced new).
    public static func kernel(from orig: Int, to new: Int) -> (MLXArray, Int, Int, Int) {
        let g = gcd(orig, new)
        let o = orig / g, n = new / g
        let base = Double(min(o, n)) * rolloff
        let width = Int((Double(lowpassFilterWidth) * Double(o) / base).rounded(.up))
        let k = 2 * width + o
        var taps = [Float](repeating: 0, count: n * k)
        let scale = base / Double(o)
        let w = Double(lowpassFilterWidth)
        for j in 0 ..< n {
            for i in 0 ..< k {
                let idx = Double(-width + i) / Double(o)
                var t = (Double(-j) / Double(n) + idx) * base
                t = min(max(t, -w), w)
                let window = pow(cos(t * Double.pi / w / 2), 2)
                t *= Double.pi
                let sinc = t == 0 ? 1.0 : sin(t) / t
                taps[j * k + i] = Float(sinc * window * scale)
            }
        }
        return (MLXArray(taps, [n, k, 1]), width, o, n)
    }

    /// `[C, S]` at `orig` Hz → `[C, ceil(S·new/orig)]` at `new` Hz.
    public static func resample(_ x: MLXArray, from orig: Int, to new: Int) -> MLXArray {
        if orig == new { return x }
        let (kernel, width, o, n) = kernel(from: orig, to: new)
        let (c, s) = (x.dim(0), x.dim(1))
        let padded = MLX.padded(x.asType(.float32), widths: [IntOrPair((0, 0)), IntOrPair((width, width + o))])
        let y = conv1d(padded.expandedDimensions(axis: -1), kernel, stride: o)          // [C, Lout, n]
        let flat = y.reshaped(c, -1)
        let target = Int((Double(n) * Double(s) / Double(o)).rounded(.up))
        return flat[0..., ..<target]
    }
}

/// SplitMix64 — the seeded stand-in for Python's `random.randint` in the reference-segment picker.
public struct SplitMix64 {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// `random.randint(a, b)`: inclusive on both ends.
    public mutating func randint(_ a: Int, _ b: Int) -> Int { b <= a ? a : a + Int(next() % UInt64(b - a + 1)) }
}

public enum AudioLoader {
    public static let sampleRate = 48000
    public static let referenceSeconds = 30
    public static let segmentSeconds = 10

    /// `_normalize_audio_to_stereo_48k`: mono duplicated, more than two channels cut to two, resampled to 48 kHz, clamped to
    /// ±1 → `[S, 2]` fp32.
    public static func stereo48k(_ clip: AudioClip) throws -> MLXArray {
        guard clip.channels >= 1, clip.frames > 0, clip.sampleRate > 0 else { throw ACEStepError.invalidInput("empty audio") }
        var x = MLXArray(clip.interleaved, [clip.frames, clip.channels]).transposed(1, 0)   // [C, S]
        if x.dim(0) == 1 { x = concatenated([x, x], axis: 0) }
        x = x[..<2]
        if clip.sampleRate != sampleRate { x = Resampler.resample(x, from: clip.sampleRate, to: sampleRate) }
        let out = MLX.clip(x, min: -1, max: 1).transposed(1, 0)
        eval(out)
        return out
    }

    /// `is_silence`: every sample under 1e-6 in magnitude.
    public static func isSilence(_ audio: MLXArray) -> Bool { (abs(audio) .< Float(1e-6)).all().item(Bool.self) }

    /// `process_reference_audio` after the loader: repeated to ≥ 30 s, then a 10 s segment from each third (front / middle /
    /// back), concatenated → exactly 30 s `[1_440_000, 2]`. The offsets are `randint(0, third − 10 s)` from a SplitMix64 seeded
    /// by `seed`; a clip of exactly 30 s gives offsets 0 on either side, which is what the oracle lane feeds.
    public static func referenceClip(_ audio: MLXArray, seed: UInt64) -> MLXArray {
        let target = referenceSeconds * sampleRate, segment = segmentSeconds * sampleRate
        var a = audio
        if a.dim(0) < target {
            let repeats = (target + a.dim(0) - 1) / a.dim(0)
            a = tiled(a, repetitions: [repeats, 1])
        }
        let total = a.dim(0)
        let third = total / 3
        var rng = SplitMix64(seed: seed)
        let front = rng.randint(0, max(0, third - segment))
        let middle = third + rng.randint(0, max(0, third - segment))
        let back = 2 * third + rng.randint(0, max(0, (total - 2 * third) - segment))
        return concatenated([a[front ..< front + segment], a[middle ..< middle + segment], a[back ..< back + segment]], axis: 0)
    }
}
