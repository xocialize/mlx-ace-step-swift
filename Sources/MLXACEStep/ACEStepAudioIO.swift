// ACEStepAudioIO.swift — canonical `Audio` (.wav) both ways: the Core's interleaved fp32 stereo at 48 kHz → 16-bit PCM WAV bytes
// (the fleet's `encodeWAV16`, extended to N channels), and a request's `referenceAudio` / `edit.source` WAV bytes → the Core's
// `AudioClip` (any PCM / float WAV, any rate / channel count — the Core's loader folds it to stereo 48 kHz).

import Foundation
import MLXToolKit
import ACEStepCore

enum ACEStepAudioIO {
    /// `Audio(.wav)` → `AudioClip`; anything else is a legible refusal (C11: the surface takes `.wav` in).
    static func decode(_ audio: Audio, field: String) throws -> AudioClip {
        guard audio.format == .wav else { throw PackageError.unsupportedRequestFeature("\(field) — this surface takes .wav audio in (got \(audio.format))") }
        do { return try WAV.decode(audio.data) }
        catch { throw PackageError.unsupportedRequestFeature("\(field) — not a readable WAV: \(error)") }
    }

    /// Interleaved float [-1, 1] → 16-bit PCM WAV (× 32767, clamped; non-finite samples become silence).
    static func encodeWAV16(interleaved samples: [Float], channels: Int, sampleRate: Int) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = samples.count * 2
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + bytes)); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(UInt16(channels)); u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * channels * 2)); u16(UInt16(channels * 2)); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(bytes))
        d.reserveCapacity(d.count + bytes)
        var pcm = [Int16](repeating: 0, count: samples.count)
        for (i, s) in samples.enumerated() {
            pcm[i] = Int16(max(-1, min(1, s.isFinite ? s : 0)) * 32767)
        }
        pcm.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
        return d
    }
}
