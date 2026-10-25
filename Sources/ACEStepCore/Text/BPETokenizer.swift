// BPETokenizer.swift — the Qwen2 byte-level BPE as Hugging Face `tokenizers` runs it for the two ACE-Step tokenizers
// (acestep-5Hz-lm-1.7B: 151 643 ranks + 65 561 added tokens incl. <|audio_code_N|>; Qwen3-Embedding-0.6B: 151 643 ranks + 26
// added tokens and a TemplateProcessing post-processor that appends <|endoftext|>). Read from tokenizer.json:
//   added tokens are split off first (exact content match, no normalisation — all of them are `<…>` strings), the remaining
//   text is NFC-normalised, pre-tokenised with the Qwen regex (`Split`, isolated), each piece mapped byte → unicode (GPT-2
//   bytes_to_unicode) and merged by lowest merge rank first; ids are the vocab's. Decoding reverses it. Gated id-exact on
//   every golden string (lane T0).
import Foundation

public final class BPETokenizer: @unchecked Sendable {
    public let vocabulary: [String: Int]
    private let byRank: [Int: String]
    private let mergeRanks: [String: Int]          // "a b" → rank
    public let addedTokens: [String: Int]
    private let addedByID: [Int: String]
    private let addedStarts: Set<Character>
    private let addedMaxLength: Int
    public let appendsEOS: Int?                    // the embedding tokenizer's TemplateProcessing special token
    public let eosTokenID: Int?
    public let padTokenID: Int?
    public let vocabularySize: Int                 // ids are < this (vocab + added)
    private let pattern: NSRegularExpression
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()
    private static let byteToUnicode: [UInt8: Character] = makeByteToUnicode()
    private static let unicodeToByte: [Character: UInt8] = Dictionary(uniqueKeysWithValues: byteToUnicode.map { ($1, $0) })

    static let patternString =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    public init(directory: URL) throws {
        let data = try ConfigIO.json(directory.appendingPathComponent("tokenizer.json"))
        guard let model = data["model"] as? [String: Any], (model["type"] as? String) == "BPE",
              let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any] else {
            throw ACEStepError.badConfig("tokenizer.json: not a BPE model")
        }
        vocabulary = vocab
        byRank = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        var ranks = [String: Int](); ranks.reserveCapacity(merges.count)
        for (i, m) in merges.enumerated() {
            if let pair = m as? [String], pair.count == 2 { ranks[pair[0] + " " + pair[1]] = i }
            else if let s = m as? String { ranks[s] = i }
        }
        mergeRanks = ranks
        var added = [String: Int](), addedByID = [Int: String](), starts = Set<Character>(), maxLen = 0
        for t in (data["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = t["content"] as? String, let id = t["id"] as? Int else { continue }
            added[content] = id; addedByID[id] = content
            if let f = content.first { starts.insert(f) }
            maxLen = max(maxLen, content.count)
        }
        addedTokens = added; self.addedByID = addedByID; addedStarts = starts; addedMaxLength = maxLen
        vocabularySize = max(vocab.count, (addedByID.keys.max() ?? -1) + 1)
        // TemplateProcessing: `single: [Sequence A, SpecialToken <|endoftext|>]` on the embedding tokenizer.
        var eosAppend: Int?
        if let post = data["post_processor"] as? [String: Any] {
            let processors = (post["processors"] as? [[String: Any]]) ?? [post]
            for p in processors where (p["type"] as? String) == "TemplateProcessing" {
                for item in (p["single"] as? [[String: Any]]) ?? [] {
                    if let special = item["SpecialToken"] as? [String: Any], let id = special["id"] as? String { eosAppend = added[id] }
                }
            }
        }
        appendsEOS = eosAppend
        let config = (try? ConfigIO.json(directory.appendingPathComponent("tokenizer_config.json"))) ?? [:]
        func special(_ key: String) -> Int? {
            if let s = config[key] as? String { return added[s] }
            if let d = config[key] as? [String: Any], let s = d["content"] as? String { return added[s] }
            return nil
        }
        eosTokenID = special("eos_token"); padTokenID = special("pad_token")
        pattern = try NSRegularExpression(pattern: Self.patternString)
    }

    // MARK: encode

    /// `tokenizer(text)` / `encode(text, add_special_tokens:)`: added tokens split off, NFC, regex, BPE; with
    /// `addSpecialTokens` the embedding tokenizer appends its <|endoftext|>. `maxLength` truncates as HF does (the special
    /// token counts toward the length).
    public func encode(_ text: String, addSpecialTokens: Bool = true, maxLength: Int? = nil) -> [Int] {
        var ids: [Int] = []
        for segment in splitAddedTokens(text) {
            switch segment {
            case .added(let id): ids.append(id)
            case .text(let s): ids += encodeOrdinary(s)
            }
        }
        let tail = addSpecialTokens ? appendsEOS : nil
        if let maxLength {
            let room = maxLength - (tail == nil ? 0 : 1)
            if ids.count > room { ids = Array(ids.prefix(max(0, room))) }
        }
        if let tail { ids.append(tail) }
        return ids
    }

    private enum Segment { case added(Int), text(String) }

    private func splitAddedTokens(_ text: String) -> [Segment] {
        guard !addedTokens.isEmpty else { return [.text(text)] }
        let chars = Array(text)
        var out: [Segment] = []
        var pending = ""
        var i = 0
        while i < chars.count {
            if addedStarts.contains(chars[i]) {
                var matched: (Int, Int)? = nil     // (length, id); shortest closing match wins — every added token is `<…>`
                var j = i + 1
                while j < chars.count, j - i < addedMaxLength {
                    if chars[j] == ">" {
                        let candidate = String(chars[i ... j])
                        if let id = addedTokens[candidate] { matched = (j - i + 1, id); break }
                    }
                    j += 1
                }
                if let (length, id) = matched {
                    if !pending.isEmpty { out.append(.text(pending)); pending = "" }
                    out.append(.added(id)); i += length; continue
                }
            }
            pending.append(chars[i]); i += 1
        }
        if !pending.isEmpty { out.append(.text(pending)) }
        return out
    }

    /// NFC → regex pieces → byte-level BPE, no added tokens recognised (`add_special_tokens=False` on plain text).
    public func encodeOrdinary(_ text: String) -> [Int] {
        let nfc = text.precomposedStringWithCanonicalMapping
        let ns = nfc as NSString
        var ids: [Int] = []
        for match in pattern.matches(in: nfc, range: NSRange(location: 0, length: ns.length)) {
            ids += bpe(ns.substring(with: match.range))
        }
        return ids
    }

    private func bpe(_ piece: String) -> [Int] {
        lock.lock(); if let hit = cache[piece] { lock.unlock(); return hit }; lock.unlock()
        var word: [String] = piece.utf8.map { String(Self.byteToUnicode[$0]!) }
        while word.count > 1 {
            var best = Int.max, at = -1
            for i in 0 ..< word.count - 1 {
                if let r = mergeRanks[word[i] + " " + word[i + 1]], r < best { best = r; at = i }
            }
            if at < 0 { break }
            let a = word[at], b = word[at + 1], merged = a + b
            var next: [String] = []; next.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == a, word[i + 1] == b { next.append(merged); i += 2 } else { next.append(word[i]); i += 1 }
            }
            word = next
        }
        let ids = word.map { vocabulary[$0] ?? -1 }
        lock.lock(); cache[piece] = ids; lock.unlock()
        return ids
    }

    // MARK: decode

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var out = ""
        var bytes: [UInt8] = []
        func flush() { if !bytes.isEmpty { out += String(decoding: bytes, as: UTF8.self); bytes.removeAll() } }
        for id in ids {
            if let s = addedByID[id] { flush(); if !skipSpecialTokens { out += s } }
            else if let tok = byRank[id] { for ch in tok { if let b = Self.unicodeToByte[ch] { bytes.append(b) } } }
        }
        flush()
        return out
    }

    public func isAdded(_ id: Int) -> Bool { addedByID[id] != nil }
    public func addedContent(_ id: Int) -> String? { addedByID[id] }

    private static func makeByteToUnicode() -> [UInt8: Character] {
        var bs: [Int] = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) { bs.append(b); cs.append(256 + n); n += 1 }
        var map = [UInt8: Character]()
        for (b, c) in zip(bs, cs) { map[UInt8(b)] = Character(UnicodeScalar(c)!) }
        return map
    }
}
