// Structural tests that never eval an array (so they run under plain `swift test` even without the metallib): the YAML
// emitter against PyYAML behaviours, the schedule scalars, and the parser.
import Testing
@testable import ACEStepCore

@Test func yamlPlainAndQuoted() {
    #expect(PyYAML.dumpMapping([("bpm", .int(73))]) == "bpm: 73\n")
    #expect(PyYAML.dumpMapping([("keyscale", .string("B♭ minor"))]) == "keyscale: B♭ minor\n")
    #expect(PyYAML.dumpMapping([("language", .string("no"))]) == "language: 'no'\n")          // YAML 1.1 bool → quoted
    #expect(PyYAML.dumpMapping([("caption", .string("a: b"))]) == "caption: 'a: b'\n")        // block indicator
    #expect(PyYAML.dumpMapping([("caption", .string("it's fine"))]) == "caption: it's fine\n")
    #expect(PyYAML.dumpMapping([("caption", .string("'quoted"))]) == "caption: '''quoted'\n")
}

@Test func yamlFoldsAt80() {
    let words = Array(repeating: "word", count: 30).joined(separator: " ")
    let out = PyYAML.dumpMapping([("caption", .string(words))])
    let lines = out.split(separator: "\n")
    #expect(lines.count > 1)
    #expect(lines.dropFirst().allSatisfy { $0.hasPrefix("  word") })
    #expect(lines.allSatisfy { $0.count <= 80 + 5 })
}

@Test func scheduleScalars() {
    #expect(Schedule.timesteps(shift: 1.0, custom: nil, steps: 8) == [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125])
    #expect(Schedule.timesteps(shift: 3.0, custom: nil, steps: nil) == Schedule.shiftTimesteps[3.0]!)
    #expect(Schedule.timesteps(shift: 3.0, custom: [0.9, 0.5, 0.0], steps: nil) == [0.9, 0.5])
}

@Test func parseLMOutput() {
    let text = "<think>\nbpm: 103\ncaption: An upbeat track\n  that continues here.\nduration: 20\nkeyscale: D minor\nlanguage: zh\ntimesignature: 4\n</think>\n<|audio_code_5|><|audio_code_63999|>"
    let (m, codes) = PlannerText.parse(text)
    #expect(m.bpm == 103 && m.duration == 20 && m.keyscale == "D minor" && m.language == "zh" && m.timesignature == "4")
    #expect(m.caption == "An upbeat track that continues here.")
    #expect(codes == [5, 63999])
    #expect(PlannerText.cotBlock(m).hasPrefix("<think>\nbpm: 103\ncaption: An upbeat track that continues here.\nduration: 20\n"))
}
