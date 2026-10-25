// WAV.swift — PCM WAV out for songs: 16-bit interleaved (any channel count) and 32-bit float; a mono/averaging reader
// for the gate lanes. Lifted in shape from mlx-auk-swift (editx-gates lineage), extended to stereo.
import Foundation

public enum WAV {
    /// samples: (frames, channels) row-major → 16-bit PCM.
    public static func write16(_ samples: [Float], channels: Int, sampleRate: Int, to url: URL) throws {
        try encode16(samples, channels: channels, sampleRate: sampleRate).write(to: url)
    }

    public static func data16(_ samples: [Float], channels: Int, sampleRate: Int) -> Data {
        encode16(samples, channels: channels, sampleRate: sampleRate)
    }

    /// Reads any PCM WAV; channels are averaged to mono.
    public static func read(_ url: URL) throws -> ([Float], Int) {
        let d = try Data(contentsOf: url)
        func u32(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 | Int(d[o + 2]) << 16 | Int(d[o + 3]) << 24 }
        func u16(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
        var o = 12; var fmt = 1, channels = 1, rate = 0, bits = 16; var samples = [Float]()
        while o + 8 <= d.count {
            let id = String(bytes: d[o ..< o + 4], encoding: .ascii) ?? "", size = u32(o + 4); let body = o + 8
            if id == "fmt " { fmt = u16(body); channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14) }
            if id == "data" {
                let n = min(size, d.count - body) / (bits / 8)
                var acc: Float = 0
                for i in 0 ..< n {
                    let p = body + i * (bits / 8); var v: Float
                    if fmt == 3 && bits == 32 { v = d[p ..< p + 4].withUnsafeBytes { $0.loadUnaligned(as: Float.self) } }
                    else if bits == 16 { v = Float(Int16(bitPattern: UInt16(u16(p)))) / 32768 }
                    else if bits == 24 { let x = Int32(d[p]) | (Int32(d[p + 1]) << 8) | (Int32(d[p + 2]) << 16); v = Float((x << 8) >> 8) / 8388608 }
                    else { v = Float(Int32(bitPattern: UInt32(u32(p)))) / 2147483648 }
                    acc += v
                    if (i + 1) % channels == 0 { samples.append(acc / Float(channels)); acc = 0 }
                }
                break
            }
            o = body + size + (size & 1)
        }
        return (samples, rate)
    }
}

extension WAV {
    static func encode16(_ samples: [Float], channels: Int, sampleRate: Int) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { Swift.withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { Swift.withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let pcm = samples.map { Int16(Swift.max(-32768, Swift.min(32767, ($0 * 32767).rounded()))) }
        let blockAlign = channels * 2
        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + pcm.count * 2)); data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(UInt16(channels)); u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * blockAlign)); u16(UInt16(blockAlign)); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(pcm.count * 2))
        pcm.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        return data
    }
}

extension WAV {
    /// Decodes a RIFF/WAVE blob (PCM 8 / 16 / 24 / 32-bit, IEEE float 32 / 64, WAVE_FORMAT_EXTENSIBLE) into interleaved
    /// fp32 frames — the canonical `Audio(.wav)` of a request. Channels are kept; the loader folds them to stereo.
    public static func decode(_ d: Data) throws -> AudioClip {
        let bytes = [UInt8](d)
        func u32(_ o: Int) -> Int { Int(bytes[o]) | Int(bytes[o + 1]) << 8 | Int(bytes[o + 2]) << 16 | Int(bytes[o + 3]) << 24 }
        func u16(_ o: Int) -> Int { Int(bytes[o]) | Int(bytes[o + 1]) << 8 }
        guard bytes.count >= 12, String(bytes: bytes[0 ..< 4], encoding: .ascii) == "RIFF", String(bytes: bytes[8 ..< 12], encoding: .ascii) == "WAVE" else {
            throw ACEStepError.invalidInput("not a RIFF/WAVE file")
        }
        var o = 12; var fmt = 0, channels = 0, rate = 0, bits = 0
        var samples: [Float] = []
        var sawData = false
        while o + 8 <= bytes.count {
            let id = String(bytes: bytes[o ..< o + 4], encoding: .ascii) ?? ""
            let size = u32(o + 4); let body = o + 8
            if id == "fmt ", body + 16 <= bytes.count {
                fmt = u16(body); channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14)
                if fmt == 0xFFFE, size >= 40, body + 26 <= bytes.count { fmt = u16(body + 24) }   // EXTENSIBLE: the sub-format GUID's first two bytes
            }
            if id == "data" {
                guard channels > 0, rate > 0, bits > 0 else { throw ACEStepError.invalidInput("WAV data before a usable fmt chunk") }
                let bytesPer = bits / 8
                let n = min(size, bytes.count - body) / bytesPer
                samples.reserveCapacity(n)
                for i in 0 ..< n {
                    let p = body + i * bytesPer
                    let v: Float
                    switch (fmt, bits) {
                    case (3, 32): v = Float(bitPattern: UInt32(u32(p)))
                    case (3, 64): v = Float(Double(bitPattern: UInt64(u32(p)) | UInt64(u32(p + 4)) << 32))
                    case (1, 8): v = (Float(bytes[p]) - 128) / 128
                    case (1, 16): v = Float(Int16(bitPattern: UInt16(u16(p)))) / 32768
                    case (1, 24): let x = Int32(bytes[p]) | Int32(bytes[p + 1]) << 8 | Int32(bytes[p + 2]) << 16; v = Float((x << 8) >> 8) / 8_388_608
                    case (1, 32): v = Float(Int32(bitPattern: UInt32(u32(p)))) / 2_147_483_648
                    default: throw ACEStepError.invalidInput("unsupported WAV format \(fmt) at \(bits) bits")
                    }
                    samples.append(v)
                }
                sawData = true
                break
            }
            o = body + size + (size & 1)
        }
        guard sawData, !samples.isEmpty else { throw ACEStepError.invalidInput("WAV without samples") }
        let frames = samples.count / channels
        return AudioClip(interleaved: Array(samples.prefix(frames * channels)), channels: channels, sampleRate: rate)
    }
}
