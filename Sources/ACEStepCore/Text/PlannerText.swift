// PlannerText.swift — the planner's text, both ways, 1:1 with acestep/llm_inference.py: the Qwen chat prompts
// (`build_formatted_prompt`, `build_formatted_prompt_with_cot`, `_build_unconditional_prompt`), the YAML reasoning block fed
// back for the codes phase (`_format_metadata_as_cot` = PyYAML `yaml.dump(cot_items, allow_unicode=True, sort_keys=True)`,
// re-emitted here with PyYAML's scalar analysis, plain / single-quoted styles and 80-column folding), `parse_lm_output`
// and `MetadataConstrainedLogitsProcessor.postprocess_caption`. Constants from acestep/constants.py.
import Foundation

public enum ACEStepConstants {
    public static let validLanguages: [String] = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
        "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
        "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
    ]
    public static let keyscaleNotes = ["A", "B", "C", "D", "E", "F", "G"]
    public static let keyscaleAccidentals = ["", "#", "b", "♯", "♭"]
    public static let keyscaleModes = ["major", "minor"]
    /// 7 × 5 × 2 = 70 keyscales, in the set-insertion order upstream builds them (order matters nowhere; it is a set).
    public static let validKeyscales: [String] = keyscaleNotes.flatMap { n in keyscaleAccidentals.flatMap { a in keyscaleModes.map { "\(n)\(a) \($0)" } } }
    public static let bpmRange = 30 ... 300
    public static let durationRange = 10 ... 600
    public static let validTimeSignatures = [2, 3, 4, 6]
    public static let maxAudioCode = 63999
    public static let defaultDiTInstruction = "Fill the audio semantic mask based on the given conditions:"
    public static let defaultLMInstruction = "Generate audio semantic tokens based on the given conditions:"
    public static let coverInstruction = "Generate audio semantic tokens based on the given conditions:"
    public static let repaintInstruction = "Repaint the mask area based on the given conditions:"
    public static let noUserInput = "NO USER INPUT"
}

/// The planner's parsed reasoning (`parse_lm_output`): strings as upstream keeps them (bpm / duration parsed to Int when numeric).
public struct PlannerMetadata: Codable, Sendable, Equatable {
    public var bpm: Int?
    public var caption: String?
    public var duration: Int?
    public var keyscale: String?
    public var language: String?
    public var timesignature: String?
    public init(bpm: Int? = nil, caption: String? = nil, duration: Int? = nil, keyscale: String? = nil, language: String? = nil, timesignature: String? = nil) {
        self.bpm = bpm; self.caption = caption; self.duration = duration; self.keyscale = keyscale; self.language = language; self.timesignature = timesignature
    }
}

public enum PlannerText {
    /// Qwen's chat template for `[system, user]` + `add_generation_prompt=True`.
    public static func chat(system: String, user: String) -> String {
        "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    }
    public static var systemContent: String { "# Instruction\n\(ACEStepConstants.defaultLMInstruction)\n\n" }

    static func isMeaningful(negative: String) -> Bool {
        let t = negative.trimmingCharacters(in: .whitespacesAndNewlines)
        return !t.isEmpty && t != ACEStepConstants.noUserInput
    }

    /// `build_formatted_prompt(caption, lyrics, generation_phase="cot")` (conditional).
    public static func cotPrompt(caption: String, lyrics: String) -> String {
        chat(system: systemContent, user: "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n")
    }
    /// `build_formatted_prompt(..., is_negative_prompt=True, generation_phase="cot")`.
    public static func cotUnconditionalPrompt(lyrics: String, negative: String) -> String {
        let user = isMeaningful(negative: negative) ? "# Caption\n\(negative)\n\n# Lyric\n\(lyrics)\n" : "# Lyric\n\(lyrics)\n"
        return chat(system: systemContent, user: user)
    }
    /// `build_formatted_prompt_with_cot(caption, lyrics, cot_text)`: the assistant turn left open, `\n\n` before the first code.
    public static func codesPrompt(caption: String, lyrics: String, cotBlock: String) -> String {
        chat(system: systemContent, user: "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n") + cotBlock + "\n\n"
    }
    /// `build_formatted_prompt_with_cot(..., is_negative_prompt=True)`: the raw negative prompt as the user turn, empty reasoning.
    public static func codesUnconditionalPrompt(negative: String) -> String {
        chat(system: systemContent, user: isMeaningful(negative: negative) ? negative : ACEStepConstants.noUserInput) + "<think>\n\n</think>" + "\n\n"
    }

    // MARK: `_format_metadata_as_cot`

    /// `<think>\n{yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip()}\n</think>`; numeric strings become ints,
    /// a `N/4` time signature its numerator.
    public static func cotBlock(_ m: PlannerMetadata) -> String {
        var items: [(String, YAMLValue)] = []
        func put(_ key: String, _ v: Any?) {
            guard let v else { return }
            if let i = v as? Int { items.append((key, .int(i))); return }
            var s = v as! String
            if key == "timesignature", s.hasSuffix("/4") { s = String(s.split(separator: "/", omittingEmptySubsequences: false)[0]) }
            if !s.isEmpty, s.unicodeScalars.allSatisfy({ CharacterSet.decimalDigits.contains($0) }), let i = Int(s) { items.append((key, .int(i))) }
            else { items.append((key, .string(s))) }
        }
        put("bpm", m.bpm); put("caption", m.caption); put("duration", m.duration); put("keyscale", m.keyscale)
        put("language", m.language); put("timesignature", m.timesignature)
        let yaml = items.isEmpty ? "" : PyYAML.dumpMapping(items.sorted { $0.0 < $1.0 }).trimmingCharacters(in: .whitespacesAndNewlines)
        return "<think>\n\(yaml)\n</think>"
    }

    // MARK: `parse_lm_output`

    /// Returns the metadata and the audio codes found in `output` (`<|audio_code_N|>` occurrences, in order).
    public static func parse(_ output: String) -> (PlannerMetadata, [Int]) {
        var codes: [Int] = []
        var rest = Substring(output)
        while let open = rest.range(of: "<|audio_code_") {
            let tail = rest[open.upperBound...]
            guard let close = tail.range(of: "|>") else { break }
            if let v = Int(tail[..<close.lowerBound]) { codes.append(v) }
            rest = tail[close.upperBound...]
        }
        var reasoning: String
        if let s = output.range(of: "<think>"), let e = output.range(of: "</think>", range: s.upperBound ..< output.endIndex) {
            reasoning = String(output[s.upperBound ..< e.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let s = output.range(of: "<reasoning>"), let e = output.range(of: "</reasoning>", range: s.upperBound ..< output.endIndex) {
            reasoning = String(output[s.upperBound ..< e.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            reasoning = (output.components(separatedBy: "<|audio_code_").first ?? output).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var meta = PlannerMetadata()
        var key: String? = nil
        var lines: [String] = []
        func save() {
            defer { key = nil; lines = [] }
            guard let key, !lines.isEmpty else { return }
            let value = lines.joined(separator: "\n")
            let stripped = value.trimmingCharacters(in: .whitespacesAndNewlines)
            switch key {
            case "bpm": meta.bpm = Int(stripped)
            case "caption": meta.caption = postprocessCaption(value)
            case "duration": meta.duration = Int(stripped)
            case "keyscale": meta.keyscale = stripped
            case "language": meta.language = stripped
            case "timesignature": meta.timesignature = stripped
            default: break
            }
        }
        for line in reasoning.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("<") { continue }
            if let first = line.first, !first.isWhitespace, let colon = line.firstIndex(of: ":") {
                save()
                key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                let first = String(line[line.index(after: colon)...])
                if !first.trimmingCharacters(in: .whitespaces).isEmpty { lines.append(first) }
            } else if line.hasPrefix(" ") || line.hasPrefix("\t") {
                if key != nil { lines.append(line) }
            }
        }
        save()
        return (meta, codes)
    }

    /// `postprocess_caption`: every line stripped, empties dropped, joined with one space.
    public static func postprocessCaption(_ caption: String) -> String {
        caption.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.joined(separator: " ")
    }
}

public enum YAMLValue { case int(Int), string(String) }

/// The slice of PyYAML's Emitter (emitter.py) that `yaml.dump` of a flat `{str: int|str}` mapping exercises: block mapping
/// at the root, `best_indent` 2, `best_width` 80, `allow_unicode` True; plain scalars when the analysis and the implicit
/// resolver allow, else single-quoted (double-quoted only for characters neither style can carry).
public enum PyYAML {
    static let bestWidth = 80
    static let indent = 2

    public static func dumpMapping(_ items: [(String, YAMLValue)]) -> String {
        var out = ""
        for (key, value) in items {
            out += key + ":"
            var column = key.unicodeScalars.count + 1
            switch value {
            case .int(let i): out += " " + String(i)
            case .string(let s):
                let a = analyze(s)
                if !s.isEmpty, a.allowBlockPlain, !resolvesToNonString(s) {
                    out += writePlain(s, column: &column)
                } else if a.allowSingleQuoted {
                    out += writeSingleQuoted(s, column: &column)
                } else {
                    out += " " + writeDoubleQuoted(s)
                }
            }
            out += "\n"
        }
        return out
    }

    struct Analysis { var allowBlockPlain = true; var allowSingleQuoted = true; var multiline = false }

    static let whitespaceLike: Set<Unicode.Scalar> = ["\0", " ", "\t", "\r", "\n", "\u{85}", "\u{2028}", "\u{2029}"]
    static let lineBreaks: Set<Unicode.Scalar> = ["\n", "\u{85}", "\u{2028}", "\u{2029}"]

    static func analyze(_ scalar: String) -> Analysis {
        let s = Array(scalar.unicodeScalars)
        if s.isEmpty { return Analysis(allowBlockPlain: true, allowSingleQuoted: true) }
        var blockIndicators = false, lineBreaksSeen = false, special = false
        var leadingSpace = false, leadingBreak = false, trailingSpace = false, trailingBreak = false
        var breakSpace = false, spaceBreak = false
        if scalar.hasPrefix("---") || scalar.hasPrefix("...") { blockIndicators = true }
        var precededByWhitespace = true
        var followedByWhitespace = s.count == 1 || whitespaceLike.contains(s[1])
        var previousSpace = false, previousBreak = false
        var index = 0
        while index < s.count {
            let ch = s[index]
            if index == 0 {
                if "#,[]{}&*!|>'\"%@`".unicodeScalars.contains(ch) { blockIndicators = true }
                if (ch == "?" || ch == ":") && followedByWhitespace { blockIndicators = true }
                if ch == "-" && followedByWhitespace { blockIndicators = true }
            } else {
                if ch == ":" && followedByWhitespace { blockIndicators = true }
                if ch == "#" && precededByWhitespace { blockIndicators = true }
            }
            if lineBreaks.contains(ch) { lineBreaksSeen = true }
            if !(ch == "\n" || (0x20 ... 0x7E).contains(ch.value)) {
                let v = ch.value
                let unicodeOK = v == 0x85 || (0xA0 ... 0xD7FF).contains(v) || (0xE000 ... 0xFFFD).contains(v) || (0x10000 ..< 0x10FFFF).contains(v)
                if !(unicodeOK && v != 0xFEFF) { special = true }
            }
            if ch == " " {
                if index == 0 { leadingSpace = true }
                if index == s.count - 1 { trailingSpace = true }
                if previousBreak { breakSpace = true }
                previousSpace = true; previousBreak = false
            } else if lineBreaks.contains(ch) {
                if index == 0 { leadingBreak = true }
                if index == s.count - 1 { trailingBreak = true }
                if previousSpace { spaceBreak = true }
                previousSpace = false; previousBreak = true
            } else { previousSpace = false; previousBreak = false }
            index += 1
            precededByWhitespace = whitespaceLike.contains(ch)
            followedByWhitespace = index + 1 >= s.count || whitespaceLike.contains(s[index + 1])
        }
        var a = Analysis(multiline: lineBreaksSeen)
        if leadingSpace || leadingBreak || trailingSpace || trailingBreak { a.allowBlockPlain = false }
        if breakSpace { a.allowBlockPlain = false; a.allowSingleQuoted = false }
        if spaceBreak || special { a.allowBlockPlain = false; a.allowSingleQuoted = false }
        if lineBreaksSeen { a.allowBlockPlain = false }
        if blockIndicators { a.allowBlockPlain = false }
        return a
    }

    /// PyYAML's implicit resolvers (YAML 1.1): a plain scalar that would read back as bool / int / float / null /
    /// timestamp / merge / value is not a str, so the emitter quotes it.
    static let resolverPatterns: [String] = [
        #"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$"#,
        #"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$"#,
        #"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"#,
        #"^(?:~|null|Null|NULL|)$"#,
        #"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$"#,
        #"^(?:<<)$"#, #"^(?:=)$"#,
    ]
    static func resolvesToNonString(_ s: String) -> Bool {
        resolverPatterns.contains { s.range(of: $0, options: .regularExpression) != nil }
    }

    /// `write_plain(text, split=True)` after `key:` (the emitter is not in whitespace, so a space is written first).
    static func writePlain(_ text: String, column: inout Int) -> String {
        let s = Array(text.unicodeScalars)
        var out = " "; column += 1
        var spaces = false, breaks = false
        var start = 0, end = 0
        while end <= s.count {
            let ch: Unicode.Scalar? = end < s.count ? s[end] : nil
            if spaces {
                if ch != " " {
                    if start + 1 == end && column > bestWidth {
                        out += "\n" + String(repeating: " ", count: indent); column = indent
                    } else {
                        out += String(String.UnicodeScalarView(s[start ..< end])); column += end - start
                    }
                    start = end
                }
            } else if breaks {
                if ch == nil || !lineBreaks.contains(ch!) {
                    for br in s[start ..< end] { out += br == "\n" ? "\n" : String(br) }
                    if s[start] == "\n" { out += "" }   // write_line_break per break already emitted above
                    out += "\n" + String(repeating: " ", count: indent); column = indent
                    start = end
                }
            } else {
                if ch == nil || ch == " " || lineBreaks.contains(ch!) {
                    out += String(String.UnicodeScalarView(s[start ..< end])); column += end - start
                    start = end
                }
            }
            if let ch { spaces = ch == " "; breaks = lineBreaks.contains(ch) }
            end += 1
        }
        return out
    }

    /// `write_single_quoted(text, split=True)`: `write_indicator("'", True)` puts a space before the opening quote.
    static func writeSingleQuoted(_ text: String, column: inout Int) -> String {
        let s = Array(text.unicodeScalars)
        var out = " '"; column += 2
        var spaces = false, breaks = false
        var start = 0, end = 0
        while end <= s.count {
            let ch: Unicode.Scalar? = end < s.count ? s[end] : nil
            if spaces {
                if ch == nil || ch != " " {
                    if start + 1 == end && column > bestWidth && start != 0 && end != s.count {
                        out += "\n" + String(repeating: " ", count: indent); column = indent
                    } else {
                        out += String(String.UnicodeScalarView(s[start ..< end])); column += end - start
                    }
                    start = end
                }
            } else if breaks {
                if ch == nil || !lineBreaks.contains(ch!) {
                    for br in s[start ..< end] { out += br == "\n" ? "\n" : String(br) }
                    out += "\n" + String(repeating: " ", count: indent); column = indent
                    start = end
                }
            } else {
                if ch == nil || ch == " " || lineBreaks.contains(ch!) || ch == "'" {
                    if start < end { out += String(String.UnicodeScalarView(s[start ..< end])); column += end - start; start = end }
                }
            }
            if ch == "'" { out += "''"; column += 2; start = end + 1 }
            if let ch { spaces = ch == " "; breaks = lineBreaks.contains(ch) }
            end += 1
        }
        out += "'"; column += 1
        return out
    }

    /// Only reached for control characters; JSON escaping matches PyYAML's double-quoted style for them.
    static func writeDoubleQuoted(_ text: String) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: [text], options: [.withoutEscapingSlashes])) ?? Data()
        let json = String(decoding: data, as: UTF8.self)
        return String(json.dropFirst().dropLast())
    }
}
