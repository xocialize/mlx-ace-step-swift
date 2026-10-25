// Structural tests for the V3b pieces that need no kernel: the edit geometry (`prepare_padding_info`), the repaint config
// (`_resolve_repaint_config`), the soft masks (latent blend + waveform splice), the DTW, the LRC line format, the WAV decoder.
import Foundation
import Testing
@testable import ACEStepCore

@Test func editGeometryPadsAndRanges() {
    let cover = EditGeometry.make(sourceSeconds: 30, kind: .cover(strength: 1))
    #expect(cover.isCover && cover.paddedSeconds == 30)
    let repaint = EditGeometry.make(sourceSeconds: 30, kind: .repaint(start: 10, end: 20))
    #expect(repaint.ranges.count == 1 && repaint.ranges[0].start == 10 && repaint.ranges[0].end == 20 && repaint.paddedSeconds == 30)
    #expect(repaint.latentRanges(frames: 750, sampleRate: 48000, hop: 1920) == [250 ..< 500])
    let tail = EditGeometry.make(sourceSeconds: 30, kind: .extend(leading: 0, trailing: 10))
    #expect(tail.trailingPadSeconds == 10 && tail.ranges[0].start == 30 && tail.ranges[0].end == 40)
    #expect(tail.latentRanges(frames: 1000, sampleRate: 48000, hop: 1920) == [750 ..< 1000])
    let head = EditGeometry.make(sourceSeconds: 30, kind: .extend(leading: 10, trailing: 0))
    #expect(head.leadingPadSeconds == 10 && head.ranges[0].start == 0 && head.ranges[0].end == 10)
    let both = EditGeometry.make(sourceSeconds: 30, kind: .extend(leading: 5, trailing: 7))
    #expect(both.ranges.count == 2 && both.paddedSeconds == 42 && both.ranges[1].start == 35 && both.ranges[1].end == 42)
    // upstream's repaint past the end pads right and keeps the range
    let past = EditGeometry.make(sourceSeconds: 30, kind: .repaint(start: 25, end: 35))
    #expect(past.trailingPadSeconds == 5 && past.ranges[0].start == 25 && past.ranges[0].end == 35)
    // the latent clamp: a range ending past T clamps to T, a start at T clamps to T − 1
    #expect(EditGeometry.make(sourceSeconds: 30, kind: .repaint(start: 29.99, end: 30)).latentRanges(frames: 750, sampleRate: 48000, hop: 1920) == [749 ..< 750])
}

@Test func repaintConfigMatchesUpstream() {
    let b = RepaintConfig.balanced
    #expect(b.injectionRatio == 0.5 && b.crossfadeFrames == 12 && b.waveformCrossfadeSeconds == 0.025)   // round(12.5) = 12 in Python
    let a = RepaintConfig(mode: "aggressive", strength: 0.5)
    #expect(a.injectionRatio == 0 && a.crossfadeFrames == 0 && a.waveformCrossfadeSeconds == 0)
    let c = RepaintConfig(mode: "conservative", strength: 0.5)
    #expect(c.injectionRatio == 1 && c.crossfadeFrames == 25 && c.waveformCrossfadeSeconds == 0.05)
    #expect(RepaintConfig(mode: "balanced", strength: 0).crossfadeFrames == 25)
    #expect(RepaintConfig(mode: "balanced", strength: 1).crossfadeFrames == 0)
}

@Test func softMasksRamp() {
    var mask = [Bool](repeating: false, count: 20)
    for i in 8 ..< 12 { mask[i] = true }
    let soft = Repaint.softMask(mask, crossfade: 3)
    #expect(soft[8 ..< 12].allSatisfy { $0 == 1 } && soft[0 ..< 5].allSatisfy { $0 == 0 } && soft[15 ..< 20].allSatisfy { $0 == 0 })
    #expect(abs(soft[5] - 0.25) < 1e-6 && abs(soft[6] - 0.5) < 1e-6 && abs(soft[7] - 0.75) < 1e-6)   // linspace(0, 1, 5)[1:-1]
    #expect(abs(soft[12] - 0.75) < 1e-6 && abs(soft[13] - 0.5) < 1e-6 && abs(soft[14] - 0.25) < 1e-6)
    #expect(Repaint.softMask([Bool](repeating: true, count: 4), crossfade: 2) == [1, 1, 1, 1])
    let edge = Repaint.softMask([true, true, false, false, false, false], crossfade: 2)
    #expect(edge[0] == 1 && edge[1] == 1 && abs(edge[2] - 2.0 / 3) < 1e-6 && abs(edge[3] - 1.0 / 3) < 1e-6 && edge[4] == 0)
    let wave = WaveformSplice.mask(samples: 100, ranges: [(40, 60)], crossfade: 10)
    #expect(wave[50] == 1 && wave[20] == 0 && abs(wave[30] - 1.0 / 11) < 1e-6 && abs(wave[60] - 10.0 / 11) < 1e-6 && wave[70] == 0)
}

@Test func dtwFollowsTheDiagonal() {
    // a clean diagonal cost: the path visits every (i, i)
    let n = 5, m = 5
    var cost = [Double](repeating: 1, count: n * m)
    for i in 0 ..< n { cost[i * m + i] = 0 }
    let (t, f) = LyricAligner.dtw(cost: cost, n: n, m: m)
    #expect(t == [0, 1, 2, 3, 4] && f == [0, 1, 2, 3, 4])
    // more frames than tokens: the last token stretches
    let (t2, f2) = LyricAligner.dtw(cost: [0, 1, 1, 1, 1, 0, 0, 0], n: 2, m: 4)
    #expect(t2.first == 0 && f2.first == 0 && t2.last == 1 && f2.last == 3 && t2.count == f2.count)
}

@Test func lrcFormat() {
    let lines = [LyricLineStamp(text: "[Verse]", start: 0, end: 0, confidence: 0),
                 LyricLineStamp(text: "Morning light", start: 0.32, end: 2, confidence: 0),
                 LyricLineStamp(text: "later", start: 83.456, end: 90, confidence: 0)]
    #expect(LyricAligner.lrc(lines) == "[00:00.00][Verse]\n[00:00.32]Morning light\n[01:23.46]later")
    #expect(LyricAligner.lyricRows(lyricIDs: [1, 2, 3, 9, 9, 151643, 151643], headerLength: 3) == 3 ..< 5)
    #expect(LyricAligner.lyricRows(lyricIDs: [1, 2, 3, 9], headerLength: 3) == 3 ..< 4)
}

@Test func wavDecodeRoundTrips() throws {
    let samples: [Float] = [0, 0.5, -0.5, 0.25, 1, -1]
    let data = WAV.data16(samples, channels: 2, sampleRate: 44100)
    let clip = try WAV.decode(data)
    #expect(clip.channels == 2 && clip.sampleRate == 44100 && clip.frames == 3)
    for (a, b) in zip(clip.interleaved, samples) { #expect(abs(a - b) < 1.0 / 32767 + 1e-6) }
    #expect(throws: (any Error).self) { try WAV.decode(Data([1, 2, 3])) }
}
