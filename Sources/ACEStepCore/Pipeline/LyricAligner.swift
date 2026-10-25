// LyricAligner.swift — `LyricTimestampMixin.get_lyric_timestamp` + `acestep/core/scoring/dit_alignment.py` (`MusicStampsAligner`)
// + `_dtw.py`, op for op: the lyric rows of the cross-attention probabilities of the configured heads (2B DiT:
// `{2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}`) at `t = 1 / inference_steps` on `x = t·noise + (1 − t)·latents`;
// bidirectional consensus (row softmax × column softmax, −violence × the row and column medians, ReLU, square, a global
// z-score, a median filter) averaged over the heads; `dtw_cpu` on the negated matrix; per-token start / end from the path
// (a token without a path step inherits the previous end); sentences split on a token containing a newline, decoded jointly;
// `[mm:ss.xx]text` lines. Scalars in Double where Python holds floats; the matrices in fp32 as torch keeps them.
import Foundation
import MLX

public struct LyricTokenStamp: Sendable, Equatable {
    public var tokenID: Int
    public var text: String
    public var start: Double
    public var end: Double
}

public struct LyricLineStamp: Sendable, Equatable {
    public var text: String
    public var start: Double
    public var end: Double
    public var confidence: Double
}

public struct LyricTimestamps: Sendable, Equatable {
    public var tokens: [LyricTokenStamp]
    public var lines: [LyricLineStamp]
    public var lrc: String
}

public enum LyricAligner {
    /// `_DEFAULT_LAYERS_CONFIG` for the 2B DiT (24 layers / 16 heads), in the order Python iterates the dict.
    public static let defaultHeads: [(layer: Int, heads: [Int])] = [(2, [6]), (3, [10, 11]), (4, [3]), (5, [8, 9]), (6, [8])]
    public static let violenceLevel: Float = 2.0
    public static let medianFilterWidth = 1
    public static let endOfText = 151643

    /// `_extract_lyric_segment`: the lyric rows of the packed conditions — after the `# Languages … # Lyric` header, up to the
    /// first `<|endoftext|>` (the one `_format_lyrics` writes into the text).
    public static func lyricRows(lyricIDs: [Int], headerLength: Int) -> Range<Int> {
        let end = lyricIDs.firstIndex(of: endOfText) ?? lyricIDs.count
        return headerLength ..< max(headerLength, end)
    }

    /// The `[Hsel, Ltokens, Lq]` stack the aligner reads: `captured[layer][head]` `[1, Lq, Lk]` → transposed, the lyric rows.
    public static func stack(_ captured: [Int: [Int: MLXArray]], heads: [(layer: Int, heads: [Int])], rows: Range<Int>) -> MLXArray {
        var mats: [MLXArray] = []
        for (layer, hs) in heads {
            for h in hs { if let m = captured[layer]?[h] { mats.append(m[0].transposed(1, 0)[rows, 0...]) } }
        }
        return stacked(mats, axis: 0)
    }

    /// torch.quantile(0.5, dim, keepdim=True): the midpoint of the two middle order statistics on an even count.
    static func median(_ x: MLXArray, axis: Int) -> MLXArray {
        let n = x.dim(axis)
        let s = sorted(x, axis: axis)
        if n % 2 == 1 { return take(s, MLXArray([Int32((n - 1) / 2)]), axis: axis) }
        let lo = take(s, MLXArray([Int32(n / 2 - 1)]), axis: axis), hi = take(s, MLXArray([Int32(n / 2)]), axis: axis)
        return lo + (hi - lo) * 0.5
    }

    /// `_dtw.median_filter`: reflect-padded sliding median along the last axis (width 1 is the identity, as upstream runs it).
    static func medianFilter(_ x: MLXArray, width: Int) -> MLXArray {
        let pad = width / 2
        let m = x.dim(-1)
        if pad == 0 || m <= pad { return x }
        let left = x[.ellipsis, 1 ... pad][.ellipsis, .stride(by: -1)]
        let right = x[.ellipsis, (m - 1 - pad) ..< (m - 1)][.ellipsis, .stride(by: -1)]
        let padded = concatenated([left, x, right], axis: -1)
        let windows = stacked((0 ..< width).map { padded[.ellipsis, $0 ..< ($0 + m)] }, axis: -1)
        return take(sorted(windows, axis: -1), MLXArray([Int32(width / 2)]), axis: -1).squeezed(axis: -1)
    }

    /// `_apply_bidirectional_consensus` on the head stack `[H, N, M]` → (`calc_matrix` `[N, M]`, `energy_matrix` `[N, M]`).
    public static func consensus(_ weights: MLXArray, violence: Float = violenceLevel, medianWidth: Int = medianFilterWidth) -> (MLXArray, MLXArray) {
        let w = weights.asType(.float32)
        let rowProb = softmax(w, axis: -1), colProb = softmax(w, axis: -2)
        var processed = rowProb * colProb
        processed = maximum(processed - violence * median(processed, axis: -1), 0)
        processed = maximum(processed - violence * median(processed, axis: -2), 0)
        processed = processed * processed
        let energy = processed.mean(axis: 0)
        let mean = processed.mean()
        let std = sqrt(((processed - mean) * (processed - mean)).mean())
        var z = (processed - mean) / (std + Float(1e-9))
        z = medianFilter(z, width: medianWidth)
        let calc = z.mean(axis: 0)
        eval(calc, energy)
        return (calc, energy)
    }

    /// `dtw_cpu` + `_backtrace` on an `[N, M]` cost (row-major Doubles) → the path's (text index, time index) pairs.
    public static func dtw(cost x: [Double], n: Int, m: Int) -> (text: [Int], time: [Int]) {
        let w = m + 1
        var cost = [Double](repeating: .infinity, count: (n + 1) * w)
        var trace = [Int8](repeating: -1, count: (n + 1) * w)
        cost[0] = 0
        for j in 1 ... max(m, 1) where j <= m {
            for i in 1 ... max(n, 1) where i <= n {
                let c0 = cost[(i - 1) * w + (j - 1)], c1 = cost[(i - 1) * w + j], c2 = cost[i * w + (j - 1)]
                let c: Double, t: Int8
                if c0 < c1 && c0 < c2 { c = c0; t = 0 } else if c1 < c0 && c1 < c2 { c = c1; t = 1 } else { c = c2; t = 2 }
                cost[i * w + j] = x[(i - 1) * m + (j - 1)] + c
                trace[i * w + j] = t
            }
        }
        for j in 0 ..< w { trace[j] = 2 }
        for i in 0 ... n { trace[i * w] = 1 }
        var text: [Int] = [], time: [Int] = []
        var i = n, j = m
        while i > 0 || j > 0 {
            text.append(i - 1); time.append(j - 1)
            let t = trace[i * w + j]
            if t == 0 { i -= 1; j -= 1 } else if t == 1 { i -= 1 } else if t == 2 { j -= 1 } else { break }
        }
        return (text.reversed(), time.reversed())
    }

    /// `_decode_tokens_incrementally`: each token's own text as the bytes it adds to the running decode (an incomplete UTF-8
    /// sequence contributes "").
    static func incrementalTexts(_ ids: [Int], tokenizer: BPETokenizer) -> [String] {
        var out: [String] = []
        var previous: [UInt8] = []
        for i in 0 ..< ids.count {
            let current = Array(tokenizer.decode(Array(ids[0 ... i])).utf8)
            var text = ""
            if current.count >= previous.count { text = String(validating: current[previous.count...], as: UTF8.self) ?? "" }
            out.append(text)
            previous = current
        }
        return out
    }

    /// Python's `round(x, 3)` for the magnitudes here (half to even on the scaled value).
    static func round3(_ x: Double) -> Double { (x * 1000).rounded(.toNearestOrEven) / 1000 }

    /// `token_timestamps`.
    public static func tokenStamps(calc: MLXArray, tokens: [Int], duration: Double, tokenizer: BPETokenizer) -> [LyricTokenStamp] {
        let (n, m) = (calc.dim(0), calc.dim(1))
        let cost = calc.asType(.float32).asArray(Float.self).map { -Double($0) }
        let (textIdx, timeIdx) = dtw(cost: cost, n: n, m: m)
        let secondsPerFrame = duration / Double(m)
        let texts = incrementalTexts(tokens, tokenizer: tokenizer)
        var out: [LyricTokenStamp] = []
        var firstTime = [Int](repeating: -1, count: n), lastTime = [Int](repeating: -1, count: n)
        for (t, f) in zip(textIdx, timeIdx) where t >= 0 && t < n {
            if firstTime[t] < 0 { firstTime[t] = f }
            lastTime[t] = f
        }
        for i in 0 ..< n {
            var start: Double, end: Double
            if firstTime[i] < 0 {
                start = out.last?.end ?? 0; end = start
            } else {
                start = Double(firstTime[i]) * secondsPerFrame; end = Double(lastTime[i]) * secondsPerFrame
            }
            if end < start { end = start }
            out.append(LyricTokenStamp(tokenID: tokens[i], text: texts[i], start: start, end: end))
        }
        return out
    }

    /// `sentence_timestamps`: tokens grouped up to each token whose text contains a newline, decoded together.
    public static func lineStamps(_ tokens: [LyricTokenStamp], tokenizer: BPETokenizer) -> [LyricLineStamp] {
        var lines: [LyricLineStamp] = []
        var current: [LyricTokenStamp] = []
        func flush() {
            guard !current.isEmpty else { return }
            let text = tokenizer.decode(current.map(\.tokenID)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                lines.append(LyricLineStamp(text: text, start: round3(current[0].start), end: round3(current[current.count - 1].end), confidence: 0))
            }
            current.removeAll()
        }
        for t in tokens {
            current.append(t)
            if t.text.contains("\n") { flush() }
        }
        flush()
        return lines
    }

    /// `format_lrc` (start times only).
    public static func lrc(_ lines: [LyricLineStamp]) -> String {
        lines.map { line in
            let minutes = Int((line.start / 60).rounded(.down))
            let seconds = line.start.truncatingRemainder(dividingBy: 60)
            return String(format: "[%02d:%05.2f]", minutes, seconds) + line.text
        }.joined(separator: "\n")
    }

    /// `stamps_align_info` + `get_timestamps_and_lrc` on the head stack `[H, N, M]`.
    public static func align(stack: MLXArray, tokens: [Int], duration: Double, tokenizer: BPETokenizer) -> LyricTimestamps {
        let (calc, _) = consensus(stack)
        let t = tokenStamps(calc: calc, tokens: tokens, duration: duration, tokenizer: tokenizer)
        let l = lineStamps(t, tokenizer: tokenizer)
        return LyricTimestamps(tokens: t, lines: l, lrc: lrc(l))
    }
}
