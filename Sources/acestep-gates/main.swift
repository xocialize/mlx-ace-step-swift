// acestep-gates — the parity lanes of PORTING-SPEC.md §3 against the fp32 oracle (Tools/oracle-capture/dump_goldens.py).
//
//   acestep-gates [opts] tokenizer <golden> <bundles>        T0: ids on every golden string, CoT round trip, YAML block, prompts
//   acestep-gates [opts] text      <golden> <bundles>        T:  Qwen3-Embedding hidden states; lyric gather
//   acestep-gates [opts] cond      <golden> <bundles>        C:  packed encoder_hidden_states + mask
//   acestep-gates [opts] planner   <golden> <bundles>        P:  teacher-forced logits (argmax, tail relL2) ×3 prefixes
//   acestep-gates [opts] fsm       <golden> <bundles>        K:  the FSM's allowed sets along the oracle's CoT; the codes rule
//   acestep-gates [opts] detok     <golden> <bundles>        F:  FSQ + detokenizer hints
//   acestep-gates [opts] dit       <golden> <bundles>        N:  one velocity at t = 0.75
//   acestep-gates [opts] solve     <golden> <bundles>        S+E: noise / schedule bit-exact, the 8-step solve
//   acestep-gates [opts] vae       <golden> <bundles>        D:  full + chunked decodes
//   acestep-gates [opts] e2e       <golden> <bundles> [wav]  W:  codes + noise → Swift chain → waveform
//   acestep-gates [opts] all       <golden> <bundles>        every lane above
//   acestep-gates [opts] render    <bundles> <request.json> <out.wav>   a real generation (sampled; not a gate)
//   acestep-gates [opts] batch     <bundles> <prompts.json> <outdir>    V2b tier run: the 12 V0 songs → <id>.wav + manifest.jsonl
//                                                                        (the V0 harness's row shape, so harness/score.py scores it)
//   acestep-gates [opts] validate  <bundles> <prompts.json> <outdir> [audio-dir]   V3: MLXServeEngine register → prepare → run → evict,
//                                                                        phys_footprint at each step (the manifest's numbers); with
//                                                                        [audio-dir] (source_30s.wav + reference_30s.wav) the V3b
//                                                                        requests too: reference, cover, repaint, extend, timestamps
//   acestep-gates [opts] planner8 <V2 golden> <bundles> <8bit-dir> [<prompts.json> <p8.wav> <out.wav>]   V4b: the -8bit bundle vs quantise-at-load
//   acestep-gates [opts] resample|encoder|timbre|tokenize|cover|repaint|extend|lrc|v3b <goldens-v3b> <bundles>
//                                                                        V3b lanes (MEASUREMENTS §V3b) against dump_goldens_v3b.py;
//                                                                        `lrc` reads the V2 goldens beside (--v2 <dir>, default ../goldens)
// Options: --cpu, --dtype float32|bfloat16, --planner-bits N, --dit-bits N, --cache-gb G (default 1), --planner <dir> (another
// planner bundle, e.g. the 0.6B), --ids p01,p02 (batch: a subset), --tier NAME (batch: the label written to the manifest).
import Darwin
import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXToolKit
import MLXServeCore
import ACEStepCore
import MLXACEStep

var args = Array(CommandLine.arguments.dropFirst())
var useCPU = false
var dtype: DType = .float32
var plannerBits: Int? = nil, ditBits: Int? = nil
var cacheGB = 1.0
var plannerDir: URL? = nil
var onlyIDs: [String]? = nil
var tierLabel: String? = nil
var vaeChunk = 512
var v2Goldens: URL? = nil
var noTimestamps = false
while let a = args.first, a.hasPrefix("--") {
    args.removeFirst()
    switch a {
    case "--v2": v2Goldens = URL(fileURLWithPath: args.removeFirst())
    case "--no-decoder-evals": OobleckDecoder.evalBetweenBlocks = false   // V4b Q8-S diagnostic
    case "--no-timestamps": noTimestamps = true                           // V4b Q8-S diagnostic (batch / render without the LRC pass)
    case "--vae-chunk": vaeChunk = Int(args.removeFirst()) ?? 512
    case "--cpu": useCPU = true
    case "--dtype": dtype = ["bfloat16", "bf16"].contains(args.removeFirst()) ? .bfloat16 : .float32
    case "--planner-bits": plannerBits = Int(args.removeFirst())
    case "--dit-bits": ditBits = Int(args.removeFirst())
    case "--cache-gb": cacheGB = Double(args.removeFirst()) ?? 1
    case "--planner": plannerDir = URL(fileURLWithPath: args.removeFirst())
    case "--ids": onlyIDs = args.removeFirst().split(separator: ",").map(String.init)
    case "--tier": tierLabel = args.removeFirst()
    default: fatalError("unknown option \(a)")
    }
}
guard let cmd = args.first else { print("usage: acestep-gates [opts] tokenizer|text|cond|planner|fsm|detok|dit|solve|vae|e2e|all|render|batch|validate …"); exit(2) }
let rest = Array(args.dropFirst())
if useCPU { Device.setDefault(device: Device(.cpu)) }
Memory.cacheLimit = Int(cacheGB * Double(1 << 30))

struct Row { let name: String; let maxAbs: Float; let relL2: Float; let cos: Float; let pass: Bool? }
var rows = [Row]()
var failures = 0
@discardableResult
func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray, relL2Limit: Float? = nil, snrLimit: Float? = nil) -> Float {
    let g = got.asType(.float32), r = ref.asType(.float32)
    guard g.shape == r.shape else {
        print(String(format: "  %-28@ SHAPE MISMATCH got %@ ref %@", name as NSString, "\(g.shape)" as NSString, "\(r.shape)" as NSString))
        rows.append(Row(name: name, maxAbs: .infinity, relL2: .infinity, cos: 0, pass: false)); failures += 1; return .infinity
    }
    let d = g - r
    let maxAbs = abs(d).max().item(Float.self)
    let relL2 = (sqrt((d * d).sum()) / maximum(sqrt((r * r).sum()), MLXArray(Float(1e-30)))).item(Float.self)
    let cos = ((g * r).sum() / maximum(sqrt((g * g).sum()) * sqrt((r * r).sum()), MLXArray(Float(1e-30)))).item(Float.self)
    let snr = -20 * log10(max(relL2, 1e-30))
    var pass: Bool? = nil
    if let l = relL2Limit { pass = relL2 <= l }
    if let s = snrLimit { pass = (pass ?? true) && snr >= s }
    if pass == false { failures += 1 }
    rows.append(Row(name: name, maxAbs: maxAbs, relL2: relL2, cos: cos, pass: pass))
    let verdict = pass.map { $0 ? "PASS" : "FAIL" } ?? "info"
    print(String(format: "  %-28@ max|Δ| %.3e   relL2 %.3e   SNR %6.1f dB   cos %.7f   %@  %@", name as NSString, maxAbs, relL2, snr, cos, "\(g.shape)" as NSString, verdict as NSString))
    return relL2
}
func check(_ name: String, _ ok: Bool, _ detail: String = "") { if !ok { failures += 1 }; print("  \(name): \(ok ? "PASS" : "FAIL") \(detail)") }
func npy(_ dir: URL, _ name: String) throws -> MLXArray { try NPY.load(dir.appendingPathComponent("\(name).npy")) }
func ints(_ dir: URL, _ name: String) throws -> [Int] { try npy(dir, name).asArray(Int32.self).map { Int($0) } }
func url(_ s: String) -> URL { URL(fileURLWithPath: s) }
func json<T: Decodable>(_ type: T.Type, _ dir: URL, _ name: String) throws -> T { try JSONDecoder().decode(T.self, from: Data(contentsOf: dir.appendingPathComponent(name))) }
func elapsed(_ t: Date) -> String { String(format: "%.1fs", Date().timeIntervalSince(t)) }
func physFootprint() -> Double {
    var info = rusage_info_current(); let r = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0) } }
    return r == 0 ? Double(info.ri_phys_footprint) / 1e9 : .nan
}

struct Request: Decodable { let caption: String; let lyrics: String; let language: String; let duration: Double; let seed: UInt64; let frames: Int; let n_codes: Int }
struct TextGolden: Decodable { let text_prompt: String; let text_ids: [Int]; let lyric_prompt: String; let lyric_ids: [Int]; let caption: String; let instruction: String; let metas: String; let language: String; let lyrics: String }
struct PlannerSettingsGolden: Decodable { let temperature: Float; let cfg_scale: Float; let top_p: Float?; let target_duration: Double?; let user_metadata: [String: Int]? }
struct PlannerGolden: Decodable {
    let cot_prompt: String; let cot_prompt_ids: [Int]; let cot_output: String; let cot_output_ids: [Int]; let metadata: PlannerMetadataJSON; let cot_block: String
    let codes_prompt: String; let codes_prompt_ids: [Int]; let uncond_prompt: String; let uncond_prompt_ids: [Int]; let codes: [Int]; let codes_output_ids: [Int]
    let eos_token_id: Int; let audio_code_base: Int; let settings: PlannerSettingsGolden; let cot_output_reencoded_ids: [Int]?
}
struct PlannerMetadataJSON: Decodable { let bpm: Int?; let caption: String?; let duration: Int?; let keyscale: String?; let language: String?; let timesignature: String? }
struct FSMStep: Decodable { let i: Int; let token: Int; let state: String; let n_allowed: Int; let hash: String; let allowed: [Int]? }
struct FSMCodes: Decodable { let target_codes: Int; let n_allowed_with_eos: Int; let hash_with_eos: String; let hash_without_eos: String; let eos: Int }
struct FSMGolden: Decodable { let cot_steps: [FSMStep]; let final_state: String; let codes_phase: FSMCodes }
struct SamplerGolden: Decodable { let schedule: [Double]; let shift: Double; let steps: Int }

func fnv1a64(_ ids: [Int]) -> String {
    var h: UInt64 = 0xCBF2_9CE4_8422_2325
    for v in ids.sorted() {
        var x = UInt32(truncatingIfNeeded: v)
        for _ in 0 ..< 4 { h ^= UInt64(x & 0xFF); h = h &* 0x0000_0100_0000_01B3; x >>= 8 }
    }
    return String(format: "%016llx", h)
}

// MARK: bundle loading (lazy per lane)
final class Bundles {
    let dir: URL
    init(_ dir: URL) { self.dir = dir }
    lazy var modelAndSilence: (ACEStepModel, MLXArray) = { let t = Date(); let r = try! ACEStepModel.load(dir.appendingPathComponent("ACE-Step-1.5-turbo-bf16"), dtype: dtype); if let b = ditBits { r.0.quantizeDiT(groupSize: 64, bits: b) }; print("  loaded DiT bundle in \(elapsed(t))"); return r }()
    lazy var textEncoder: Qwen3LM = { let t = Date(); let m = try! Qwen3LM.load(dir.appendingPathComponent("Qwen3-Embedding-0.6B-bf16"), dtype: dtype); print("  loaded text encoder in \(elapsed(t))"); return m }()
    lazy var textTokenizer: BPETokenizer = try! BPETokenizer(directory: dir.appendingPathComponent("Qwen3-Embedding-0.6B-bf16"))
    var plannerBundle: URL { plannerDir ?? dir.appendingPathComponent("ACE-Step-1.5-planner-1.7B-bf16") }
    lazy var plannerTokenizer: BPETokenizer = try! BPETokenizer(directory: plannerBundle)
    lazy var planner: Planner = { let t = Date(); let p = try! Planner(directory: plannerBundle, dtype: dtype); if let b = plannerBits { p.model.quantizeLayers(groupSize: 64, bits: b) }; print("  loaded planner (\(plannerBundle.lastPathComponent)) in \(elapsed(t))"); return p }()
    lazy var vae: AutoencoderOobleck = { let t = Date(); let v = try! AutoencoderOobleck.load(dir.appendingPathComponent("ACE-Step-1.5-vae-fp32"), dtype: .float32); print("  loaded VAE in \(elapsed(t))"); return v }()
    func silence(_ n: Int) -> MLXArray { modelAndSilence.1[..<n].asType(modelAndSilence.0.dtype).expandedDimensions(axis: 0) }
}

// MARK: lanes
func tokenizerLane(_ g: URL, _ b: Bundles) throws {
    print("T0 tokenizer")
    let t = try json(TextGolden.self, g, "text.json"), p = try json(PlannerGolden.self, g, "planner.json"), r = try json(Request.self, g, "request.json")
    let emb = b.textTokenizer, lm = b.plannerTokenizer
    check("text ids (\(t.text_ids.count))", emb.encode(t.text_prompt, maxLength: 256) == t.text_ids)
    check("lyric ids (\(t.lyric_ids.count))", emb.encode(t.lyric_prompt, maxLength: 2048) == t.lyric_ids)
    check("cot prompt ids (\(p.cot_prompt_ids.count))", lm.encode(p.cot_prompt, addSpecialTokens: false) == p.cot_prompt_ids)
    check("codes prompt ids (\(p.codes_prompt_ids.count))", lm.encode(p.codes_prompt, addSpecialTokens: false) == p.codes_prompt_ids)
    check("uncond prompt ids (\(p.uncond_prompt_ids.count))", lm.encode(p.uncond_prompt, addSpecialTokens: false) == p.uncond_prompt_ids)
    check("cot output decode", lm.decode(p.cot_output_ids) == p.cot_output)
    // the FSM forces non-canonical splits ('b','pm' for "bpm:"), so the generated ids are not the canonical encoding — compare
    // against HF's own re-encoding of the decoded text instead.
    check("cot output re-encode == HF re-encode", lm.encode(p.cot_output, addSpecialTokens: false) == (p.cot_output_reencoded_ids ?? p.cot_output_ids))
    let (meta, _) = PlannerText.parse(p.cot_output)
    check("parse metadata", meta.bpm == p.metadata.bpm && meta.caption == p.metadata.caption && meta.duration == p.metadata.duration && meta.keyscale == p.metadata.keyscale && meta.language == p.metadata.language && meta.timesignature == p.metadata.timesignature)
    let block = PlannerText.cotBlock(meta)
    check("YAML cot block", block == p.cot_block, block == p.cot_block ? "" : "\n--- swift ---\n\(block)\n--- oracle ---\n\(p.cot_block)")
    check("cot prompt string", PlannerText.cotPrompt(caption: r.caption, lyrics: r.lyrics) == p.cot_prompt)
    check("codes prompt string", PlannerText.codesPrompt(caption: r.caption, lyrics: r.lyrics, cotBlock: p.cot_block) == p.codes_prompt)
    check("uncond prompt string", PlannerText.codesUnconditionalPrompt(negative: ACEStepConstants.noUserInput) == p.uncond_prompt)
    let metas = ACEStepPipeline.metas(bpm: meta.bpm, timesignature: meta.timesignature, keyscale: meta.keyscale, duration: r.duration)
    check("DiT text prompt string", ACEStepPipeline.textPrompt(instruction: ACEStepConstants.coverInstruction, caption: meta.caption ?? r.caption, metas: metas) == t.text_prompt)
    check("DiT lyric prompt string", ACEStepPipeline.lyricPrompt(lyrics: r.lyrics, language: r.language) == t.lyric_prompt)
    check("codes ids ↔ codes", p.codes_output_ids.dropLast().map { $0 - p.audio_code_base } == p.codes && p.codes_output_ids.last == p.eos_token_id)
}

func textLane(_ g: URL, _ b: Bundles) throws {
    print("T text encoder")
    let t = try json(TextGolden.self, g, "text.json")
    let enc = b.textEncoder
    let t0 = Date()
    let h = enc.model(MLXArray(t.text_ids.map { Int32($0) }).reshaped(1, -1), caches: nil); eval(h)
    print("  text encoder forward of \(t.text_ids.count) tokens in \(elapsed(t0))")
    compare("text_hidden", h, try npy(g, "text_hidden"), relL2Limit: 1e-3)
    let e = enc.model.embedTokens(MLXArray(t.lyric_ids.map { Int32($0) }).reshaped(1, -1))
    let r = compare("lyric_embed (gather)", e, try npy(g, "lyric_embed"))
    check("lyric gather exact", r == 0)
}

func condLane(_ g: URL, _ b: Bundles) throws {
    print("C condition encoder")
    let (m, _) = b.modelAndSilence
    let th = try npy(g, "text_hidden"), le = try npy(g, "lyric_embed")
    let refs = b.silence(m.config.timbre_fix_frame)
    let t0 = Date()
    let (enc, mask) = m.encoder(text: th.asType(m.dtype), textKeep: MLXArray.ones([1, th.dim(1)], dtype: .int32), lyrics: le.asType(m.dtype),
                                lyricKeep: MLXArray.ones([1, le.dim(1)], dtype: .int32), references: refs, referenceOrder: [0])
    eval(enc)
    print("  condition encoder in \(elapsed(t0)) → \(enc.shape)")
    compare("encoder_hidden", enc, try npy(g, "encoder_hidden"), relL2Limit: 1e-3)
    compare("encoder_hidden (direct recompute)", enc, try npy(g, "encoder_hidden_direct"))
    let refMask = try npy(g, "encoder_mask")
    check("encoder mask exact", mask.asType(.int32).shape == refMask.shape && (mask.asType(.int32) .== refMask.asType(.int32)).all().item(Bool.self))
}

func plannerLane(_ g: URL, _ b: Bundles) throws {
    print("P planner (teacher-forced)")
    let pl = b.planner
    for tag in ["cot", "codes", "uncond"] {
        let ids = try ints(g, "p_\(tag)_ids")
        let t0 = Date()
        let logits = pl.teacherForcedLogits(ids); eval(logits)
        let am = argMax(logits, axis: -1).asType(.int32)
        let refAm = try npy(g, "p_\(tag)_argmax").asType(.int32)
        let agree = (am .== refAm).sum().item(Int32.self)
        print("  \(tag): \(ids.count) tokens in \(elapsed(t0)); argmax agreement \(agree)/\(ids.count)")
        check("\(tag) argmax 100 %", Int(agree) == ids.count)
        let tail = try npy(g, "p_\(tag)_logits_tail")
        compare("\(tag) logits tail", logits[(logits.dim(0) - tail.dim(0))..., 0...], tail, relL2Limit: 1e-3)
    }
}

/// P2 — the GENERATION path (prefill → KV-cached single-token steps, the path `Planner.generate` runs) against the teacher-forced
/// full forward lane P gated: the same logits at every position of `codes prompt + the first 64 oracle codes`, and of
/// `CoT prompt + the oracle CoT`. Added at V2b (2026-10-24) after the bf16 tier's PER miss, rule stated before the first run:
/// argmax agreement 100 %, relL2 ≤ 1e-3 (fp32) — lane P's own rule, applied to the cached path.
func kvcacheLane(_ g: URL, _ b: Bundles) throws {
    print("P2 cached decode vs teacher-forced")
    let p = try json(PlannerGolden.self, g, "planner.json")
    let pl = b.planner
    for (tag, prompt, continuation) in [("codes", p.codes_prompt_ids, Array(p.codes_output_ids.prefix(64))), ("cot", p.cot_prompt_ids, p.cot_output_ids)] {
        let full = pl.teacherForcedLogits(prompt + continuation); eval(full)              // [L, V]
        let caches = pl.model.newCaches()
        var cached: [MLXArray] = [pl.prefill(prompt, caches: caches)]                       // logits after the prompt's last token
        for t in continuation.dropLast() {
            cached.append(pl.model.lastLogits(MLXArray([Int32(t)]).reshaped(1, 1), caches: caches))
        }
        let got = stacked(cached)                                                          // [n, V]
        let n0 = prompt.count
        let ref = full[(n0 - 1) ..< (n0 - 1 + continuation.count), 0...]
        let agree = (argMax(got, axis: -1) .== argMax(ref, axis: -1)).sum().item(Int32.self)
        check("\(tag): cached argmax == teacher-forced (\(continuation.count) positions)", Int(agree) == continuation.count, "agree \(agree)")
        compare("\(tag) cached logits", got, ref, relL2Limit: 1e-3)
        check("\(tag): cache offset == prompt + steps", caches[0].offset == prompt.count + continuation.count - 1, "\(caches[0].offset)")
    }
}

func fsmLane(_ g: URL, _ b: Bundles) throws {
    print("K constraint FSM")
    let p = try json(PlannerGolden.self, g, "planner.json"), f = try json(FSMGolden.self, g, "fsm.json"), r = try json(Request.self, g, "request.json")
    let pl = b.planner
    var user = UserMetadata()
    if let um = p.settings.user_metadata { if let d = um["duration"] { user.duration = String(d) }; if let v = um["bpm"] { user.bpm = String(v) } }
    pl.constraint.configure(phase: "cot", stopAtReasoning: true, targetDuration: nil, userMetadata: user, skipCaption: false, skipLanguage: false)
    let logits = pl.teacherForcedLogits(p.cot_prompt_ids + p.cot_output_ids); eval(logits)
    let n0 = p.cot_prompt_ids.count
    var mismatches = 0, stateMismatch = 0
    for step in f.cot_steps {
        let state = pl.constraint.state.rawValue
        if state != step.state { stateMismatch += 1; if stateMismatch <= 3 { print("    step \(step.i): state \(state) vs oracle \(step.state)") } }
        let c = pl.constraint.constrain(logits[n0 - 1 + step.i])
        let allowed = pl.constraint.allowedSet(c)
        let ok = allowed.count == step.n_allowed && fnv1a64(allowed) == step.hash
        if !ok { mismatches += 1; if mismatches <= 3 { print("    step \(step.i) (\(step.state)): swift allowed \(allowed.count) \(allowed.prefix(8)) vs oracle \(step.n_allowed) \(step.allowed?.prefix(8).map { "\($0)" } ?? [])") } }
        pl.constraint.update(with: step.token)
    }
    check("allowed sets exact at every CoT step (\(f.cot_steps.count))", mismatches == 0 && stateMismatch == 0, "mismatches \(mismatches), state mismatches \(stateMismatch)")
    check("final state", pl.constraint.state.rawValue == f.final_state, "\(pl.constraint.state.rawValue) vs \(f.final_state)")
    pl.constraint.configure(phase: "codes", stopAtReasoning: false, targetDuration: r.duration, userMetadata: UserMetadata(), skipCaption: true, skipLanguage: true, promptContainsThinkEnd: true)
    let codesAllowed = pl.constraint.allowedSet(pl.constraint.constrain(MLXArray.zeros([pl.constraint.vocabularySize])))
    check("codes phase: codes only, EOS blocked", fnv1a64(codesAllowed) == f.codes_phase.hash_without_eos && codesAllowed.count == f.codes_phase.n_allowed_with_eos - 1)
    for _ in 0 ..< f.codes_phase.target_codes { pl.constraint.update(with: p.audio_code_base) }
    let forced = pl.constraint.allowedSet(pl.constraint.constrain(MLXArray.zeros([pl.constraint.vocabularySize])))
    check("codes phase: EOS forced at target (\(f.codes_phase.target_codes))", forced == [f.codes_phase.eos])
    check("eos / code base", pl.constraint.eosTokenID == p.eos_token_id && pl.audioCodeBase == p.audio_code_base)
}

func detokLane(_ g: URL, _ b: Bundles) throws {
    print("F detokenizer")
    let (m, _) = b.modelAndSilence
    let p = try json(PlannerGolden.self, g, "planner.json")
    let t0 = Date()
    let h = m.hints(codes: p.codes); eval(h)
    print("  \(p.codes.count) codes → \(h.shape) in \(elapsed(t0))")
    if FileManager.default.fileExists(atPath: g.appendingPathComponent("quantized.npy").path) {
        let q = m.tokenizer.quantizer.output(fromIndices: MLXArray(p.codes.map { Int32($0) }).reshaped(1, -1), dtype: m.dtype)
        compare("FSQ quantized (info)", q, try npy(g, "quantized"))
    }
    compare("hints", h, try npy(g, "hints"), relL2Limit: 1e-3)
}

func ditLane(_ g: URL, _ b: Bundles) throws {
    print("N DiT velocity at t = 0.75")
    let (m, _) = b.modelAndSilence
    let enc = try npy(g, "encoder_hidden").asType(m.dtype), ctx = try npy(g, "context").asType(m.dtype), x = try npy(g, "dit_x_t075").asType(m.dtype)
    let cache = m.decoder.crossCache(enc)
    let t = MLXArray([Float(0.75)]).asType(m.dtype)
    let t0 = Date()
    let v = m.decoder(x, timestep: t, timestepR: t, contextLatents: ctx, cache: cache); eval(v)
    print("  velocity for \(x.shape) in \(elapsed(t0))")
    compare("velocity vs torch fp32", v, try npy(g, "dit_v_t075_torch"), relL2Limit: 1e-3)
    compare("velocity vs MLX fp32 (info)", v, try npy(g, "dit_v_t075_mlx"))
}

func solveLane(_ g: URL, _ b: Bundles) throws {
    print("S scalars + E 8-step solve")
    let (m, _) = b.modelAndSilence
    let r = try json(Request.self, g, "request.json"), s = try json(SamplerGolden.self, g, "sampler.json")
    let schedule = Schedule.timesteps(shift: s.shift, custom: nil, steps: s.steps)
    check("schedule bit-exact (shift \(s.shift), \(s.steps) steps)", schedule == s.schedule, "\(schedule)")
    let noise = DiffusionSampler.noise(seed: r.seed, frames: r.frames, channels: 64)
    compare("noise key(seed) (info)", noise, try npy(g, "noise"))
    let enc = try npy(g, "encoder_hidden").asType(m.dtype), ctx = try npy(g, "context").asType(m.dtype)
    var settings = SamplerSettings(); settings.shift = s.shift; settings.steps = s.steps
    let t0 = Date()
    let lat = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: try npy(g, "noise"), settings: settings)
    eval(lat)
    print("  \(s.steps)-step solve of \(r.frames) frames in \(elapsed(t0))")
    compare("latents (8-step, DCW)", lat, try npy(g, "latents"), relL2Limit: 5e-3)
}

func vaeLane(_ g: URL, _ b: Bundles) throws {
    print("D VAE")
    let vae = b.vae
    let lat = try npy(g, "latents")
    let t0 = Date()
    let full = vae.decode(lat); eval(full)
    print("  full decode of \(lat.dim(1)) frames → \(full.shape) in \(elapsed(t0))")
    compare("full vs diffusers fp32", full[0], try npy(g, "audio_torch_full"), snrLimit: 110)
    if FileManager.default.fileExists(atPath: g.appendingPathComponent("audio_mlx_full.npy").path) {
        compare("full vs MLX full (info)", full[0], try npy(g, "audio_mlx_full"))
    }
    let t1 = Date()
    let chunked = try vae.decodeChunked(lat, chunk: 512, overlap: 64); eval(chunked)
    print("  chunked 512/64 decode in \(elapsed(t1))")
    compare("chunked vs MLX chunked", chunked[0], try npy(g, "audio_mlx_chunked"), snrLimit: 110)
    compare("chunked vs full (swift)", chunked[0], full[0], relL2Limit: 1e-5)
}

func e2eLane(_ g: URL, _ b: Bundles, wav: String?) throws {
    print("W end to end (codes + noise → waveform)")
    let (m, _) = b.modelAndSilence
    let p = try json(PlannerGolden.self, g, "planner.json"), t = try json(TextGolden.self, g, "text.json"), r = try json(Request.self, g, "request.json")
    let s = try json(SamplerGolden.self, g, "sampler.json")
    let t0 = Date()
    let th = b.textEncoder.model(MLXArray(t.text_ids.map { Int32($0) }).reshaped(1, -1), caches: nil)
    let le = b.textEncoder.model.embedTokens(MLXArray(t.lyric_ids.map { Int32($0) }).reshaped(1, -1))
    let (enc, _) = m.encoder(text: th.asType(m.dtype), textKeep: MLXArray.ones([1, th.dim(1)], dtype: .int32), lyrics: le.asType(m.dtype),
                             lyricKeep: MLXArray.ones([1, le.dim(1)], dtype: .int32), references: b.silence(m.config.timbre_fix_frame), referenceOrder: [0])
    var src = m.hints(codes: p.codes)
    if src.dim(1) < r.frames { src = concatenated([src, b.silence(r.frames - src.dim(1))], axis: 1) }
    let ctx = concatenated([src[0..., ..<r.frames, 0...], MLXArray.ones([1, r.frames, 64], dtype: src.dtype)], axis: -1)
    eval(enc, ctx)
    print("  conditions in \(elapsed(t0))")
    var settings = SamplerSettings(); settings.shift = s.shift; settings.steps = s.steps
    let t1 = Date()
    let lat = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: r.seed, frames: r.frames, channels: 64), settings: settings)
    eval(lat)
    print("  solve in \(elapsed(t1))")
    compare("latents (swift chain)", lat, try npy(g, "latents"), relL2Limit: 5e-3)
    let t2 = Date()
    let audio = try b.vae.decodeChunked(lat, chunk: 512, overlap: 64)[0]; eval(audio)
    print("  VAE in \(elapsed(t2)); total \(elapsed(t0))")
    compare("waveform vs oracle (MLX path)", audio, try npy(g, "audio_mlx_chunked"), snrLimit: 75)
    let finished = ACEStepPipeline.finish(audio, normalize: true, db: -1)
    compare("final (−1 dBFS) vs oracle final (info)", finished, try npy(g, "audio_final"))
    if let wav { try WAV.write16(finished.asArray(Float.self), channels: 2, sampleRate: 48000, to: url(wav)); print("  wrote \(wav)") }
}


// MARK: - V3b lanes (MEASUREMENTS §V3b): reference audio, source audio (cover / repaint / extend), LRC — against dump_goldens_v3b.py

struct V3bCase: Decodable {
    let case_name: String?; let seed: UInt64; let frames: Int; let task_type: String?; let instruction: String?
    let repainting_start: Double?; let repainting_end: Double?; let audio_cover_strength: Double; let vocal_language: String?
    let caption: String?; let lyrics: String?; let text_ids: [Int]; let lyric_ids: [Int]; let non_cover_text_ids: [Int]?
    let audio_codes: String?; let text_prompt: String; let lyric_prompt: String
    enum CodingKeys: String, CodingKey { case case_name = "case", seed, frames, task_type, instruction, repainting_start, repainting_end, audio_cover_strength, vocal_language, caption, lyrics, text_ids, lyric_ids, non_cover_text_ids, audio_codes, text_prompt, lyric_prompt }
}
struct ResampleCase: Decodable { let orig_freq: Int; let new_freq: Int; let channels: Int; let input_samples: Int; let expected_samples: Int }
struct LRCTokenStamp: Decodable { let token_id: Int; let text: String; let start: Double; let end: Double }
struct LRCLineStamp: Decodable { let text: String; let start: Double; let end: Double }
struct LRCGolden: Decodable {
    let source_golden: String; let lrc_text: String; let token_stamps: [LRCTokenStamp]; let sentence_stamps: [LRCLineStamp]; let lyrics_tokens: [Int]
    let header_len: Int; let lyric_ids: [Int]; let vocal_language: String; let total_duration_seconds: Double; let inference_steps: Int; let seed: UInt64
    let custom_config: [String: [Int]]
}

func parseCodes(_ s: String) -> [Int] {
    var out: [Int] = []
    var rest = Substring(s)
    while let r = rest.range(of: "<|audio_code_") {
        rest = rest[r.upperBound...]
        guard let end = rest.firstIndex(of: "|") else { break }
        if let v = Int(rest[..<end]) { out.append(max(0, min(v, 63999))) }
        rest = rest[end...]
    }
    return out
}
func caseDir(_ root: URL, _ name: String) -> URL { root.appendingPathComponent(name) }
func audioNPY(_ dir: URL, _ name: String) throws -> MLXArray { try npy(dir, name) }   // [S, 2] fp32
func encodeKeys(seed: UInt64) -> (Int) -> MLXArray { { ACEStepPipeline.encodeKey(seed: seed, call: $0) } }

/// The conditions of a V3b case from its own ids (text / lyric prompts as the handler built them), with `references` or silence.
func caseConditions(_ b: Bundles, _ c: V3bCase, textIDs: [Int]? = nil, references: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let (m, _) = b.modelAndSilence
    let th = b.textEncoder.model(MLXArray((textIDs ?? c.text_ids).map { Int32($0) }).reshaped(1, -1), caches: nil)
    let le = b.textEncoder.model.embedTokens(MLXArray(c.lyric_ids.map { Int32($0) }).reshaped(1, -1))
    let refs = references ?? b.silence(m.config.timbre_fix_frame)
    return m.encoder(text: th.asType(m.dtype), textKeep: MLXArray.ones([1, th.dim(1)], dtype: .int32), lyrics: le.asType(m.dtype),
                     lyricKeep: MLXArray.ones([1, le.dim(1)], dtype: .int32), references: refs.asType(m.dtype), referenceOrder: [0])
}
func contextOf(_ latents: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let m = (mask ?? MLXArray.ones([1, latents.dim(1)], dtype: latents.dtype)).asType(latents.dtype)
    return concatenated([latents, broadcast(m.expandedDimensions(axis: -1), to: latents.shape)], axis: -1)
}
func settingsOf(_ b: Bundles) -> SamplerSettings { var s = SamplerSettings(); s.shift = 1.0; s.steps = 8; return s }

func resampleLane(_ root: URL, _ b: Bundles) throws {
    print("R resample (torchaudio sinc_interp_hann, width 6, rolloff 0.99)")
    for name in ["resample-44k1", "resample-16k-mono"] {
        let d = caseDir(root, name)
        let c = try json(ResampleCase.self, d, "case.json")
        let input = try npy(d, "input"), expected = try npy(d, "expected")
        let clip = AudioClip(interleaved: input.asArray(Float.self), channels: c.channels, sampleRate: c.orig_freq)
        let t0 = Date()
        let got = try AudioLoader.stereo48k(clip)
        print("  \(name): \(input.shape) @ \(c.orig_freq) → \(got.shape) in \(elapsed(t0))")
        check("\(name) length == \(c.expected_samples)", got.dim(0) == c.expected_samples, "\(got.dim(0))")
        compare("\(name) vs torchaudio", got, expected, relL2Limit: 1e-5)
    }
}

func encoderLane(_ root: URL, _ b: Bundles) throws {
    print("V VAE encoder (reference 30 s; chunked 40 s)")
    let vae = b.vae
    let inputs = caseDir(root, "_inputs"), timbre = caseDir(root, "timbre-folk-30s")
    let ref = try audioNPY(inputs, "reference_30s").expandedDimensions(axis: 0)
    let t0 = Date()
    let h = vae.encode(ref); eval(h)
    print("  encoder on \(ref.shape) → \(h.shape) in \(elapsed(t0)); phys \(String(format: "%.2f", physFootprint())) GB")
    compare("h = mean ‖ scale vs MLX fp32", h, try npy(timbre, "enc0_h"), snrLimit: 110)
    compare("mean vs diffusers fp32 (info)", split(h, parts: 2, axis: -1)[0][0], try npy(timbre, "refer_mean_diffusers"))
    let std = AutoencoderOobleck.softplus(split(h, parts: 2, axis: -1)[1]) + Float(1e-4)
    compare("std vs diffusers fp32 (info)", std[0], try npy(timbre, "refer_std_diffusers"))
    let c = try json(V3bCase.self, timbre, "case.json")
    let z = vae.sample(h, key: ACEStepPipeline.encodeKey(seed: c.seed, call: 0)); eval(z)
    compare("z (keyed sample) vs hooked oracle", z, try npy(timbre, "enc0_z"), snrLimit: 110)
    compare("z vs the pipeline's refer latents", z, try npy(timbre, "refer_latents"), snrLimit: 110)
    // chunked: the extend-tail case's padded 40 s source (two 30 s windows, 2 s overlap)
    let tail = caseDir(root, "extend-tail-folk-30s")
    let ct = try json(V3bCase.self, tail, "case.json")
    let src = try audioNPY(inputs, "source_30s")
    let geometry = EditGeometry.make(sourceSeconds: Double(src.dim(0)) / 48000, kind: .extend(leading: 0, trailing: 10))
    let padded = geometry.pad(src, sampleRate: 48000).expandedDimensions(axis: 0)
    let t1 = Date()
    let zc = try vae.encodeChunked(padded, key: encodeKeys(seed: ct.seed)); eval(zc)
    print("  chunked encode of \(padded.shape) → \(zc.shape) in \(elapsed(t1)); phys \(String(format: "%.2f", physFootprint())) GB")
    compare("chunked 30s/2s vs _mlx_encode_single", zc, try npy(tail, "target_latents"), snrLimit: 110)
    let z0 = vae.sample(vae.encode(padded[0..., ..<(28 * 48000), 0...]), key: ACEStepPipeline.encodeKey(seed: ct.seed, call: 0))
    compare("window 0 (28 s) vs enc0_z (info)", z0, try npy(tail, "enc0_z"))
}

func timbreLane(_ root: URL, _ b: Bundles) throws {
    print("B timbre (reference audio → conditions → the V2 folk song with its timbre)")
    let (m, _) = b.modelAndSilence
    let d = caseDir(root, "timbre-folk-30s"), inputs = caseDir(root, "_inputs")
    let c = try json(V3bCase.self, d, "case.json")
    let ref = try audioNPY(inputs, "reference_30s")
    let clip = AudioLoader.referenceClip(ref, seed: c.seed)
    check("30 s clip is the 30 s input (offsets 0)", (clip .== ref).all().item(Bool.self))
    let t0 = Date()
    let refs = try b.vae.encodeChunked(clip.expandedDimensions(axis: 0), key: encodeKeys(seed: c.seed)); eval(refs)
    let (enc, mask) = caseConditions(b, c, references: refs); eval(enc)
    print("  reference + conditions in \(elapsed(t0)) → \(enc.shape)")
    compare("encoder_hidden (timbre)", enc, try npy(d, "encoder_hidden"), relL2Limit: 1e-3)
    let refMask = try npy(d, "encoder_mask")
    check("encoder mask exact", mask.asType(.int32).shape == refMask.shape && (mask.asType(.int32) .== refMask.asType(.int32)).all().item(Bool.self))
    let codes = parseCodes(c.audio_codes ?? "")
    check("codes replayed (\(codes.count))", codes.count == 150)
    var src = m.hints(codes: codes)
    if src.dim(1) < c.frames { src = concatenated([src, b.silence(c.frames - src.dim(1))], axis: 1) }
    let ctx = contextOf(src[0..., ..<c.frames, 0...])
    compare("context", ctx, try npy(d, "context"), relL2Limit: 1e-3)
    let t1 = Date()
    let lat = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: c.seed, frames: c.frames, channels: 64), settings: settingsOf(b))
    eval(lat); print("  solve in \(elapsed(t1))")
    compare("latents (timbre, 8 steps)", lat, try npy(d, "pred_latents"), relL2Limit: 5e-3)
    let audio = ACEStepPipeline.clampPeak(try b.vae.decodeChunked(lat, chunk: 512, overlap: 64)[0]); eval(audio)
    compare("waveform vs oracle", audio, try npy(d, "audio_raw"), snrLimit: 75)
}

func tokenizeLane(_ root: URL, _ b: Bundles) throws {
    print("Q tokenize (FSQ indices of the source latents; the detokenized hints)")
    let (m, sil) = b.modelAndSilence
    let d = caseDir(root, "cover-half-folk-30s")
    for tag in ["tok0", "tok1"] {
        let x = try npy(d, "\(tag)_in")
        let t0 = Date()
        let (q, idx) = m.tokenize(x, silence: sil); eval(q, idx)
        let ref = try npy(d, "\(tag)_indices").asType(.int32).reshaped(1, -1)
        let same = (idx.asType(.int32) .== ref).sum().item(Int32.self)
        check("\(tag) indices exact (\(same)/\(ref.dim(1)), \(elapsed(t0)))", Int(same) == ref.dim(1))
        compare("\(tag) quantized (info)", q, try npy(d, "\(tag)_quantized"))
        let hints = m.detokenizer(q)[0..., ..<x.dim(1), 0...]
        compare("\(tag) hints", hints, try npy(d, "detok\(tag.dropFirst(3))_hints"), relL2Limit: 1e-3)
    }
}

func coverLane(_ root: URL, _ b: Bundles) throws {
    print("O cover (source → encode → tokenize → hints → the cover; O2 strength 0.5 with the non-cover switch)")
    let (m, sil) = b.modelAndSilence
    let inputs = caseDir(root, "_inputs")
    let src = try audioNPY(inputs, "source_30s")
    for name in ["cover-folk-30s", "cover-half-folk-30s"] {
        let d = caseDir(root, name)
        let c = try json(V3bCase.self, d, "case.json")
        let t0 = Date()
        let target = try b.vae.encodeChunked(src.expandedDimensions(axis: 0), key: encodeKeys(seed: c.seed)); eval(target)
        compare("\(name) target latents", target, try npy(d, "target_latents"), snrLimit: 110)
        let tgt = target.asType(m.dtype)
        let hints = m.coverHints(tgt, silence: sil)
        let ctx = contextOf(hints); eval(ctx)
        compare("\(name) context [hints ‖ 1]", ctx, try npy(d, "context"), relL2Limit: 1e-3)
        let (enc, _) = caseConditions(b, c); eval(enc)
        compare("\(name) conditions (cover instruction)", enc, try npy(d, "encoder_hidden"), relL2Limit: 1e-3)
        print("  \(name): encode + hints + conditions in \(elapsed(t0))")
        var cover: CoverSwitch? = nil
        if c.audio_cover_strength < 1, let nc = c.non_cover_text_ids {
            let (encNC, _) = caseConditions(b, c, textIDs: nc); eval(encNC)
            compare("\(name) non-cover conditions", encNC, try npy(d, "enc_non_cover"), relL2Limit: 1e-3)
            let ctxNC = contextOf(b.silence(c.frames))
            compare("\(name) non-cover context", ctxNC, try npy(d, "ctx_non_cover"), relL2Limit: 1e-3)
            cover = CoverSwitch(atStep: Int(8 * c.audio_cover_strength), cache: m.decoder.crossCache(encNC), context: ctxNC)
            check("\(name) switch at step \(cover!.atStep)", cover!.atStep == 4)
        }
        let t1 = Date()
        let lat = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: c.seed, frames: c.frames, channels: 64), settings: settingsOf(b), cover: cover)
        eval(lat); print("  \(name): solve in \(elapsed(t1))")
        compare("\(name) latents", lat, try npy(d, "pred_latents"), relL2Limit: 5e-3)
        let audio = ACEStepPipeline.clampPeak(try b.vae.decodeChunked(lat, chunk: 512, overlap: 64)[0]); eval(audio)
        compare("\(name) waveform vs oracle", audio, try npy(d, "audio_raw"), snrLimit: 75)
    }
}

func repaintLikeLane(_ root: URL, _ b: Bundles, cases: [(String, ACEStepEditKind)]) throws {
    let (m, sil) = b.modelAndSilence
    let inputs = caseDir(root, "_inputs")
    let src = try audioNPY(inputs, "source_30s")
    let cfg = RepaintConfig.balanced
    check("repaint config balanced/0.5 → ratio 0.5, 12 frames, 25 ms", cfg.injectionRatio == 0.5 && cfg.crossfadeFrames == 12 && cfg.waveformCrossfadeSeconds == 0.025)
    for (name, kind) in cases {
        let d = caseDir(root, name)
        let c = try json(V3bCase.self, d, "case.json")
        let g = EditGeometry.make(sourceSeconds: Double(src.dim(0)) / 48000, kind: kind)
        let padded = g.pad(src, sampleRate: 48000)
        let t0 = Date()
        let target = try b.vae.encodeChunked(padded.expandedDimensions(axis: 0), key: encodeKeys(seed: c.seed)); eval(target)
        compare("\(name) target latents", target, try npy(d, "target_latents"), snrLimit: 110)
        let frames = max(128, target.dim(1))
        check("\(name) frames \(frames)", frames == c.frames)
        let ranges = g.latentRanges(frames: frames, sampleRate: 48000, hop: 1920)
        let masks = RepaintMasks.build(target: target.asType(m.dtype), silence: sil, ranges: ranges)
        let refChunk = try npy(d, "chunk_mask").asType(.int32), refRepaint = try npy(d, "repaint_mask").asType(.int32)
        check("\(name) chunk mask exact (ranges \(ranges))", (masks.chunkMask.asType(.int32) .== refChunk).all().item(Bool.self))
        check("\(name) repaint mask exact", (masks.repaintMask.asType(.int32) .== refRepaint).all().item(Bool.self))
        compare("\(name) src latents (silence hole)", masks.sourceLatents, try npy(d, "src_latents"), relL2Limit: 1e-3)
        let ctx = contextOf(masks.sourceLatents, mask: masks.chunkMask); eval(ctx)
        compare("\(name) context", ctx, try npy(d, "context"), relL2Limit: 1e-3)
        let (enc, _) = caseConditions(b, c); eval(enc)
        compare("\(name) conditions (repaint instruction)", enc, try npy(d, "encoder_hidden"), relL2Limit: 1e-3)
        print("  \(name): encode + masks + conditions in \(elapsed(t0))")
        let plan = RepaintPlan(mask: masks.repaintMask, cleanSource: target.asType(m.dtype), crossfadeFrames: cfg.crossfadeFrames, injectionRatio: cfg.injectionRatio)
        let t1 = Date()
        let lat = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: c.seed, frames: frames, channels: 64), settings: settingsOf(b), repaint: plan)
        eval(lat); print("  \(name): solve in \(elapsed(t1))")
        compare("\(name) latents (injected + blended)", lat, try npy(d, "pred_latents"), relL2Limit: 5e-3)
        do {   // diagnostic: where does the latent error sit — inside the ranges, in the 12-frame ramps, or in the kept region
            let ref = try npy(d, "pred_latents")
            var regions: [(String, [Int])] = []
            for r in ranges {
                regions.append(("inside \(r)", Array(r)))
                regions.append(("left ramp", Array(max(0, r.lowerBound - cfg.crossfadeFrames) ..< r.lowerBound)))
                regions.append(("right ramp", Array(r.upperBound ..< min(frames, r.upperBound + cfg.crossfadeFrames))))
            }
            let masked = Set(regions.flatMap(\.1))
            regions.append(("kept", (0 ..< frames).filter { !masked.contains($0) }))
            for (label, idx) in regions where !idx.isEmpty {
                let i = MLXArray(idx.map { Int32($0) })
                let g = take(lat, i, axis: 1), rr = take(ref, i, axis: 1)
                let rel = (sqrt(((g - rr) * (g - rr)).sum()) / maximum(sqrt((rr * rr).sum()), MLXArray(Float(1e-30)))).item(Float.self)
                print(String(format: "    region %-18@ frames %5d  relL2 %.3e  max|Δ| %.3e", label as NSString, idx.count, rel, abs(g - rr).max().item(Float.self)))
            }
            let unblended = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: c.seed, frames: frames, channels: 64), settings: settingsOf(b), repaint: RepaintPlan(mask: masks.repaintMask, cleanSource: target.asType(m.dtype), crossfadeFrames: 0, injectionRatio: cfg.injectionRatio))
            compare("\(name) latents without the blend (info)", unblended, ref)
            let noInject = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: c.seed, frames: frames, channels: 64), settings: settingsOf(b), repaint: RepaintPlan(mask: masks.repaintMask, cleanSource: target.asType(m.dtype), crossfadeFrames: cfg.crossfadeFrames, injectionRatio: 0))
            compare("\(name) latents without the injection (info)", noInject, ref)
            // the decisive split: the Swift sampler on the ORACLE's own inputs (its conditions, context, clean source, mask)
            let oEnc = try npy(d, "encoder_hidden").asType(m.dtype), oCtx = try npy(d, "context").asType(m.dtype)
            let oSrc = try npy(d, "diffusion_clean_src").asType(m.dtype), oMask = try npy(d, "diffusion_repaint_mask").asType(.int32) .> 0
            let fromOracle = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(oEnc), context: oCtx, noise: try npy(d, "noise"), settings: settingsOf(b), repaint: RepaintPlan(mask: oMask, cleanSource: oSrc, crossfadeFrames: cfg.crossfadeFrames, injectionRatio: cfg.injectionRatio))
            compare("\(name) Swift sampler on the oracle's inputs (info)", fromOracle, ref)
            // sensitivity: the same solve with the context nudged by 1e-6 relative noise
            let nudged = ctx + ctx * MLXRandom.normal(ctx.shape, key: MLXRandom.key(7)).asType(ctx.dtype) * Float(1e-6)
            let sens = try DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: nudged, noise: DiffusionSampler.noise(seed: c.seed, frames: frames, channels: 64), settings: settingsOf(b), repaint: plan)
            compare("\(name) sensitivity to a 1e-6 context nudge (info)", sens, lat)
        }
        let decoded = ACEStepPipeline.clampPeak(try b.vae.decodeChunked(lat, chunk: 512, overlap: 64)[0]); eval(decoded)
        compare("\(name) decoded (pre-splice, info)", decoded, try npy(d, "audio_decoded"))
        let spliced = WaveformSplice.apply(pred: decoded, source: padded, ranges: g.ranges, sampleRate: 48000, crossfadeSeconds: cfg.waveformCrossfadeSeconds); eval(spliced)
        compare("\(name) spliced waveform vs oracle", spliced, try npy(d, "audio_raw"), snrLimit: 75)
    }
}
func repaintLane(_ root: URL, _ b: Bundles) throws {
    print("P repaint [10, 20] s (+ the declared diagnostics: [15, 25] s at seed 1004, [10, 20] s at seed 1005)")
    var cases: [(String, ACEStepEditKind)] = [("repaint-folk-30s", .repaint(start: 10, end: 20))]
    for (name, kind) in [("repaint-b-folk-30s", ACEStepEditKind.repaint(start: 15, end: 25)), ("repaint-c-folk-30s", .repaint(start: 10, end: 20))]
    where FileManager.default.fileExists(atPath: caseDir(root, name).appendingPathComponent("case.json").path) { cases.append((name, kind)) }
    try repaintLikeLane(root, b, cases: cases)
}
func extendLane(_ root: URL, _ b: Bundles) throws {
    print("X extend (tail +10 s; head +10 s) as repaint past the source")
    try repaintLikeLane(root, b, cases: [("extend-tail-folk-30s", .extend(leading: 0, trailing: 10)), ("extend-head-folk-30s", .extend(leading: 10, trailing: 0))])
}

func lrcLane(_ root: URL, _ b: Bundles) throws {
    print("L LRC (cross-attention at t = 1/8 → consensus → DTW → lines)")
    let (m, _) = b.modelAndSilence
    let v2 = v2Goldens ?? root.deletingLastPathComponent().appendingPathComponent("goldens")
    for name in ["lrc-folk-30s", "lrc-citypop-20s"] {
        let d = caseDir(root, name)
        let g = try json(LRCGolden.self, d, "lrc.json")
        let src = v2.appendingPathComponent(g.source_golden)
        let lat = try npy(src, "latents"), enc = try npy(src, "encoder_hidden").asType(m.dtype), ctx = try npy(src, "context").asType(m.dtype)
        let noise = try npy(d, "noise")
        let t = 1.0 / Double(g.inference_steps)
        let x = (Float(t) * noise + Float(1 - t) * lat).asType(m.dtype)
        let tArr = MLXArray([Float(t)]).asType(m.dtype)
        let t0 = Date()
        let wanted = Dictionary(uniqueKeysWithValues: LyricAligner.defaultHeads.map { ($0.layer, $0.heads) })
        let captured = m.decoder.crossAttentions(x, timestep: tArr, timestepR: tArr, contextLatents: ctx, cache: m.decoder.crossCache(enc), layers: wanted)
        let rows = LyricAligner.lyricRows(lyricIDs: g.lyric_ids, headerLength: g.header_len)
        check("\(name) lyric rows \(rows) == golden header \(g.header_len) + \(g.lyrics_tokens.count) tokens", rows.lowerBound == g.header_len && rows.count == g.lyrics_tokens.count)
        let header = b.textTokenizer.encode(ACEStepPipeline.lyricHeader(language: g.vocal_language), addSpecialTokens: false)
        check("\(name) header length from the tokenizer (\(header.count))", header.count == g.header_len)
        let stack = LyricAligner.stack(captured, heads: LyricAligner.defaultHeads, rows: rows); eval(stack)
        print("  \(name): 7 layers + heads in \(elapsed(t0)) → \(stack.shape)")
        // the golden holds every layer / head: [24, 16, Ltok, Lq]
        let all = try npy(d, "attention")
        var refs: [MLXArray] = []
        for (layer, hs) in LyricAligner.defaultHeads { for h in hs { refs.append(all[layer, h]) } }
        let refStack = stacked(refs, axis: 0)
        compare("\(name) head probabilities (info)", stack, refStack)
        let (calc, _) = LyricAligner.consensus(stack)
        compare("\(name) calc_matrix (info)", calc, try npy(d, "calc_matrix"))
        let (calcRef, _) = LyricAligner.consensus(refStack)
        compare("\(name) consensus on the oracle's heads (info)", calcRef, try npy(d, "calc_matrix"))
        let stamps = LyricAligner.align(stack: stack, tokens: Array(g.lyric_ids[rows]), duration: g.total_duration_seconds, tokenizer: b.textTokenizer)
        var off = 0
        for (a, r) in zip(stamps.tokens, g.token_stamps) where abs(a.start - r.start) > 1e-6 || abs(a.end - r.end) > 1e-6 { off += 1 }
        print("  \(name): \(stamps.tokens.count) tokens (\(g.token_stamps.count) golden), \(off) with a start / end off; \(stamps.lines.count) lines (\(g.sentence_stamps.count) golden)")
        check("\(name) token count", stamps.tokens.count == g.token_stamps.count)
        check("\(name) lrc_text string-equal", stamps.lrc == g.lrc_text, stamps.lrc == g.lrc_text ? "" : "\n--- swift ---\n\(stamps.lrc)\n--- oracle ---\n\(g.lrc_text)")
        // the aligner alone on the oracle's own heads (isolates the DiT from the alignment code)
        let fromOracle = LyricAligner.align(stack: refStack, tokens: g.lyrics_tokens, duration: g.total_duration_seconds, tokenizer: b.textTokenizer)
        check("\(name) aligner on the oracle's heads string-equal (info)", fromOracle.lrc == g.lrc_text)
        if fromOracle.lrc != g.lrc_text { failures -= 1 }   // informational row: not counted
    }
}


// MARK: - V4b: the pre-quantised 8-bit planner bundle vs the quantise-at-load path (MEASUREMENTS §V4b)
//   acestep-gates planner8 <V2 goldens case dir> <bundles> <8bit bundle dir> [<prompts.json> <reference p8 wav> <out wav>]

func planner8Lane(_ g: URL, _ b: Bundles, int8Dir: URL, song: (String, String, String)?) throws {
    let qc = try ConfigIO.load(Qwen3Config.self, from: int8Dir.appendingPathComponent("config.json")).quantization
    let bits = qc?.bits ?? 8, group = qc?.group_size ?? 64
    print("Q\(bits) the -\(bits)bit planner bundle (\(int8Dir.lastPathComponent)); TF32 \(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "default")")
    let t0 = Date()
    let atLoad = try Qwen3LM.load(b.plannerBundle, dtype: dtype); atLoad.quantizeLayers(groupSize: group, bits: bits); eval(atLoad.parameters())
    let pre = try Qwen3LM.load(int8Dir, dtype: dtype)
    check("bundle config carries quantization g\(group) / \(bits)-bit affine", pre.config.quantization == QuantizationConfig(group_size: group, bits: bits, mode: "affine"), "\(String(describing: pre.config.quantization))")
    check("loader built QuantizedLinears (isPreQuantized)", pre.isPreQuantized)
    print("  loaded both in \(elapsed(t0))")
    // Q8-T: every parameter bit-equal
    let a = Dictionary(uniqueKeysWithValues: atLoad.parameters().flattened()), p = Dictionary(uniqueKeysWithValues: pre.parameters().flattened())
    check("key sets equal (\(a.count) vs \(p.count))", Set(a.keys) == Set(p.keys), "missing \(Set(a.keys).subtracting(p.keys).count), extra \(Set(p.keys).subtracting(a.keys).count)")
    var equal = 0, differing: [String] = []
    var kinds: [String: Int] = [:]
    for (k, x) in a {
        guard let y = p[k] else { continue }
        let same = x.shape == y.shape && x.dtype == y.dtype && (x .== y).all().item(Bool.self)
        if same { equal += 1 } else { differing.append(k) }
        kinds[k.split(separator: ".").last.map(String.init) ?? "?", default: 0] += 1
    }
    check("Q\(bits)-T tensors bit-equal (\(equal)/\(a.count); \(kinds.sorted { $0.key < $1.key }))", differing.isEmpty, differing.prefix(5).joined(separator: ", "))
    let bytes = p.values.reduce(0) { $0 + $1.nbytes }
    print(String(format: "  -%dbit parameters %.2f GB in memory (bf16 planner %.2f GB)", bits, Double(bytes) / 1e9, Double(b.planner.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }) / 1e9))
    // Q8-P: teacher-forced logits identical on the V2 goldens' prefixes
    for tag in ["cot", "codes"] {
        let ids = try ints(g, "p_\(tag)_ids")
        let x = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
        let la = atLoad.logits(x, caches: nil).asType(.float32), lp = pre.logits(x, caches: nil).asType(.float32); eval(la, lp)
        let maxAbs = abs(la - lp).max().item(Float.self)
        let agree = (argMax(la, axis: -1) .== argMax(lp, axis: -1)).sum().item(Int32.self)
        check("Q\(bits)-P \(tag) logits identical over \(ids.count) positions (max|Δ| \(maxAbs), argmax \(agree)/\(ids.count))", maxAbs == 0)
        let gold = try npy(g, "p_\(tag)_argmax").asType(.int32)
        let vsGold = (argMax(lp, axis: -1)[0].asType(.int32) .== gold).sum().item(Int32.self)
        print("    vs the fp32 oracle argmax (info, 8-bit): \(vsGold)/\(ids.count)")
    }
    // Q8-G diagnostic (declared after the first Q8-S miss): the two planners' SAMPLED plans at the same seed — same CoT, same codes?
    // and the KV-cached generation path's last logits on the CoT prefix
    do {
        let ids = try ints(g, "p_cot_ids")
        let ca = atLoad.newCaches(), cp = pre.newCaches()
        let xa = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
        let la = atLoad.lastLogits(xa, caches: ca), lp = pre.lastLogits(xa, caches: cp); eval(la, lp)
        print("    Q\(bits)-G cached last logits on the CoT prefix: max|Δ| \(abs(la - lp).max().item(Float.self))")
        let pa = try Planner(directory: b.plannerBundle, dtype: dtype); pa.model.quantizeLayers(groupSize: group, bits: bits); eval(pa.model.parameters())
        let pp = try Planner(directory: int8Dir, dtype: dtype)
        let r = try json(Request.self, g, "request.json")
        pa.random = MLXRandom.RandomState(seed: r.seed); pp.random = MLXRandom.RandomState(seed: r.seed)
        let ra = try pa.plan(caption: r.caption, lyrics: r.lyrics, targetDuration: r.duration, userMetadata: UserMetadata(), settings: PlannerSettings())
        let rp = try pp.plan(caption: r.caption, lyrics: r.lyrics, targetDuration: r.duration, userMetadata: UserMetadata(), settings: PlannerSettings())
        print("    Q\(bits)-G plans at seed \(r.seed): CoT equal \(ra.cotText == rp.cotText), codes equal \(ra.codes == rp.codes) (\(ra.codes.count) / \(rp.codes.count)); first differing code at \(zip(ra.codes, rp.codes).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1)")
        if ra.cotText != rp.cotText { print("    at-load CoT: \(ra.cotText.prefix(200))\n    bundle  CoT: \(rp.cotText.prefix(200))") }
    }
    // Q8-S: the seeded song vs the V2b p8 take
    guard let (promptsPath, refWav, outWav) = song else { return }
    let rows = try JSONDecoder().decode(PromptsFile.self, from: Data(contentsOf: url(promptsPath))).prompts
    guard let r = rows.first(where: { $0.id.hasPrefix("p04") }) else { check("p04 row", false); return }
    let t1 = Date()
    let pipe = try ACEStepPipeline(bundles: b.dir, dtype: dtype, plannerBits: nil, ditBits: nil, planner: int8Dir)
    check("pipeline took the pre-quantised planner without quantising at load", pipe.planner.model.isPreQuantized)
    var req = ACEStepRequest(caption: r.caption, lyrics: r.lyrics, language: r.language, duration: r.duration, seed: r.seed)
    req.timestamps = false
    let songOut = try pipe.generate(req)
    print("  p04 at seed \(r.seed): \(songOut.codes.count) codes, \(songOut.frames) frames, \(elapsed(t1)); bpm \(songOut.metadata.bpm ?? -1), key \(songOut.metadata.keyscale ?? "?")")
    try WAV.write16(songOut.audio, channels: 2, sampleRate: 48000, to: url(outWav))
    let mine = try WAV.decode(try Data(contentsOf: url(outWav))), ref = try WAV.decode(try Data(contentsOf: url(refWav)))
    check("Q\(bits)-S lengths equal (\(mine.interleaved.count) vs \(ref.interleaved.count) samples)", mine.interleaved.count == ref.interleaved.count)
    let n = min(mine.interleaved.count, ref.interleaved.count)
    var maxLSB = 0, off = 0
    for i in 0 ..< n { let d = Int((mine.interleaved[i] * 32768).rounded()) - Int((ref.interleaved[i] * 32768).rounded()); if d != 0 { off += 1; maxLSB = max(maxLSB, abs(d)) } }
    check("Q\(bits)-S sample-exact vs the V2b p\(bits) take (\(off) samples differ, max \(maxLSB) LSB)", off == 0)
}

struct RenderRequest: Decodable { let caption: String; let lyrics: String; let language: String?; let duration: Double?; let seed: UInt64?; let shift: Double? }
func renderLane(_ bundles: URL, _ requestPath: String, _ out: String) throws {
    print("render (a sampled generation; not a gate)")
    let rr = try JSONDecoder().decode(RenderRequest.self, from: Data(contentsOf: url(requestPath)))
    let t0 = Date()
    let pipe = try ACEStepPipeline(bundles: bundles, dtype: dtype, plannerBits: plannerBits, ditBits: ditBits, planner: plannerDir)
    print(String(format: "  loaded in %@; phys %.2f GB; MLX active %.2f GB", elapsed(t0) as NSString, physFootprint(), Double(Memory.activeMemory) / 1e9))
    var req = ACEStepRequest(caption: rr.caption, lyrics: rr.lyrics, language: rr.language ?? "en", duration: rr.duration, seed: rr.seed ?? 0)
    if let sh = rr.shift { req.sampler.shift = sh }
    req.vaeChunk = vaeChunk
    print("  VAE chunk \(vaeChunk) frames")
    var lastPhase = ""
    var phasePeak: [String: Int] = [:]
    let song = try pipe.generate(req) { phase, done, total in
        if phase != lastPhase {
            // MLX active / peak at the phase boundary — the per-phase activation trace (the peak is cumulative, so a phase's own
            // contribution is the increase it adds)
            print(String(format: "\n  → %@ starts: MLX active %.2f GB, peak so far %.2f GB, phys %.2f GB", phase as NSString, Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9, physFootprint()))
            lastPhase = phase
        }
        phasePeak[phase] = max(phasePeak[phase] ?? 0, Memory.peakMemory)
        FileHandle.standardError.write("\r  \(phase) \(done)/\(total)   ".data(using: .utf8)!)
    }
    print("")
    print("  peak MLX memory reached within each phase (cumulative): " + phasePeak.sorted { $0.key < $1.key }.map { String(format: "%@ %.2f GB", $0.key as NSString, Double($0.value) / 1e9) }.joined(separator: ", "))
    try WAV.write16(song.audio, channels: 2, sampleRate: 48000, to: url(out))
    print("  metadata: \(song.metadata)")
    print("  codes: \(song.codes.count) → \(song.frames) frames = \(Double(song.frames) / 25) s")
    print("  timings: \(song.timings.sorted { $0.key < $1.key }.map { "\($0.key) \(String(format: "%.1f", $0.value))s" }.joined(separator: ", "))")
    print(String(format: "  memory: MLX peak %.2f GB, active %.2f GB; phys_footprint now %.2f GB", Double(Memory.peakMemory) / 1e9, Double(Memory.activeMemory) / 1e9, physFootprint()))
    print("  wrote \(out)")
}

// MARK: - batch (V2b): the 12 V0 songs through one resident pipeline, the V0 harness's manifest row shape

struct PromptRow: Decodable { let id: String; let language: String; let duration: Double; let seed: UInt64; let caption: String; let lyrics: String }
struct PromptsFile: Decodable { let prompts: [PromptRow] }

/// `phys_footprint` sampled every 100 ms on a thread (the V0 harness's `FootprintSampler`): the peak and a (t, bytes) series.
final class PhysSampler {
    private let lock = NSLock()
    private var stopped = false
    private(set) var peak: UInt64 = 0
    private(set) var series: [(Double, UInt64)] = []
    private let t0 = Date()
    private var thread: Thread?
    static func read() -> UInt64 {
        var info = rusage_info_current()
        let r = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0) } }
        return r == 0 ? info.ri_phys_footprint : 0
    }
    init(period: TimeInterval = 0.1) {
        let th = Thread { [self] in
            while true {
                lock.lock(); let done = stopped; lock.unlock()
                if done { break }
                let b = Self.read()
                lock.lock(); peak = max(peak, b); series.append((Date().timeIntervalSince(t0), b)); lock.unlock()
                Thread.sleep(forTimeInterval: period)
            }
        }
        th.qualityOfService = .utility
        thread = th
        th.start()
    }
    /// A synchronous reading folded into the peak.
    @discardableResult func mark() -> UInt64 { let b = Self.read(); lock.lock(); peak = max(peak, b); lock.unlock(); return b }
    /// Resets the per-song peak to the current reading.
    func resetPeak() { lock.lock(); peak = Self.read(); lock.unlock() }
    func stop() -> UInt64 { lock.lock(); stopped = true; lock.unlock(); thread = nil; return peak }
}

func appendJSONL(_ row: [String: Any], to file: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys, .withoutEscapingSlashes])
    if !FileManager.default.fileExists(atPath: file.path) { try Data().write(to: file) }
    let h = try FileHandle(forWritingTo: file); defer { try? h.close() }
    try h.seekToEnd(); try h.write(contentsOf: data + Data("\n".utf8))
}
func doneIDs(_ file: URL) -> Set<String> {
    guard let text = try? String(contentsOf: file, encoding: .utf8) else { return [] }
    return Set(text.split(separator: "\n").compactMap { line -> String? in
        guard let d = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { return nil }
        return d["id"] as? String
    })
}
func wavSeconds(samples: Int, channels: Int, rate: Int) -> Double { Double(samples / channels) / Double(rate) }

func batchLane(_ bundles: URL, _ promptsPath: String, _ outDir: String) throws {
    let tier = tierLabel ?? "\(dtype == .float32 ? "fp32" : "bf16")\(plannerBits.map { "-p\($0)" } ?? "")\(plannerDir != nil ? "-" + plannerDir!.lastPathComponent : "")"
    let out = url(outDir)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let manifest = out.appendingPathComponent("manifest.jsonl")
    print("batch tier \(tier): bundles \(bundles.path), planner \(plannerDir?.lastPathComponent ?? ACEStepBundles.plannerName), planner-bits \(plannerBits.map(String.init) ?? "none"), dtype \(dtype), cache \(cacheGB) GiB, VAE chunk 512 → \(out.path)")
    let sampler = PhysSampler()
    let t0 = Date()
    let pipe = try ACEStepPipeline(bundles: bundles, dtype: dtype, plannerBits: plannerBits, ditBits: ditBits, planner: plannerDir)
    eval(pipe.model.parameters()); Memory.clearCache()
    let loadS = Date().timeIntervalSince(t0)
    Thread.sleep(forTimeInterval: 0.3)
    let physAfterLoad = sampler.mark()
    print(String(format: "  loaded in %.1fs; phys after load %.2f GB; MLX active %.2f GB", loadS, Double(physAfterLoad) / 1e9, Double(Memory.activeMemory) / 1e9))
    var rows = try JSONDecoder().decode(PromptsFile.self, from: Data(contentsOf: url(promptsPath))).prompts
    if let onlyIDs { rows = rows.filter { r in onlyIDs.contains { r.id.hasPrefix($0) } } }
    let done = doneIDs(manifest)
    var seriesPeak: UInt64 = physAfterLoad
    for r in rows {
        if done.contains(r.id) { print("  skip \(r.id) (done)"); continue }
        let physBefore = sampler.mark()
        seriesPeak = max(seriesPeak, sampler.peak)
        sampler.resetPeak()
        let t = Date()
        var req = ACEStepRequest(caption: r.caption, lyrics: r.lyrics, language: r.language, duration: r.duration, seed: r.seed)
        if noTimestamps { req.timestamps = false }
        req.vaeChunk = 512
        do {
            let song = try pipe.generate(req) { phase, done, total in
                if done == total || done % 50 == 0 { FileHandle.standardError.write("\r  \(r.id) \(phase) \(done)/\(total)   ".data(using: .utf8)!) }
            }
            let wall = Date().timeIntervalSince(t)
            Thread.sleep(forTimeInterval: 0.3)
            let physPeak = max(sampler.mark(), sampler.peak)      // the thread's 100 ms peak over this song, folded with a final reading
            seriesPeak = max(seriesPeak, physPeak)
            let wav = out.appendingPathComponent("\(r.id).wav")
            try WAV.write16(song.audio, channels: 2, sampleRate: 48000, to: wav)
            let audioS = wavSeconds(samples: song.audio.count, channels: 2, rate: 48000)
            let m = song.metadata
            var lmMeta: [String: Any] = ["n_codes": song.codes.count, "truncated": song.truncated, "cot_tokens": song.cotText.count, "language_used": song.language]
            if let v = m.bpm { lmMeta["bpm"] = v }; if let v = m.caption { lmMeta["caption"] = v }; if let v = m.duration { lmMeta["duration"] = v }
            if let v = m.keyscale { lmMeta["keyscale"] = v }; if let v = m.language { lmMeta["language"] = v }; if let v = m.timesignature { lmMeta["timesignature"] = v }
            let tc: [String: Any] = ["lm_phase1_time": song.timings["lm_phase1"] ?? 0, "lm_phase2_time": song.timings["lm_phase2"] ?? 0,
                                     "lm_total_time": (song.timings["lm_phase1"] ?? 0) + (song.timings["lm_phase2"] ?? 0),
                                     "conditions_time_cost": song.timings["conditions"] ?? 0, "dit_diffusion_time_cost": song.timings["diffusion"] ?? 0,
                                     "dit_vae_decode_time_cost": song.timings["vae"] ?? 0, "pipeline_total_time": song.timings["total"] ?? 0]
            let row: [String: Any] = ["id": r.id, "ok": true, "wav": wav.path, "audio_s": audioS, "sr": 48000, "channels": 2, "wall_s": wall, "rtf": audioS / wall,
                                      "time_costs": tc, "phys_before": physBefore, "phys_peak": physPeak, "mx_active_peak": Memory.peakMemory,
                                      "mx_active_now": Memory.activeMemory, "mx_cache_now": Memory.cacheMemory, "seed": r.seed, "duration_req": r.duration,
                                      "lm_meta": lmMeta, "status": "ok", "tier": tier]
            try appendJSONL(row, to: manifest)
            print(String(format: "\n  %@: %.1fs audio in %.1fs (RTF %.2f); planner %.1f + %.1f s, DiT %.1f s, VAE %.1f s; phys peak %.2f GB; MLX peak %.2f GB; bpm %@ %@ ts %@%@",
                         r.id as NSString, audioS, wall, audioS / wall, song.timings["lm_phase1"] ?? 0, song.timings["lm_phase2"] ?? 0, song.timings["diffusion"] ?? 0,
                         song.timings["vae"] ?? 0, Double(physPeak) / 1e9, Double(Memory.peakMemory) / 1e9, (m.bpm.map(String.init) ?? "?") as NSString,
                         (m.keyscale ?? "?") as NSString, (m.timesignature ?? "?") as NSString, (song.truncated ? " TRUNCATED" : "") as NSString))
        } catch {
            let wall = Date().timeIntervalSince(t)
            try appendJSONL(["id": r.id, "ok": false, "error": String(describing: error), "wall_s": wall, "tier": tier], to: manifest)
            print("\n  \(r.id) FAILED: \(error)")
        }
    }
    let finalPeak = max(seriesPeak, sampler.stop())
    try appendJSONL(["id": "_run", "load_s": loadS, "phys_after_load": physAfterLoad, "phys_series_peak": finalPeak, "cache_limit_gb": cacheGB,
                     "tier": tier, "dtype": "\(dtype)", "planner_bits": plannerBits ?? 0, "planner_dir": plannerDir?.lastPathComponent ?? ACEStepBundles.plannerName,
                     "vae_chunk": 512, "lm_backend": "swift", "use_mlx_dit": true, "use_mlx_vae": true, "n_series": sampler.series.count], to: manifest)
    let series = sampler.series.map { [$0.0, Double($0.1)] }
    try JSONSerialization.data(withJSONObject: series).write(to: out.appendingPathComponent("phys_series.json"))
    print(String(format: "DONE tier %@; run phys peak %.2f GB; phys now %.2f GB", tier as NSString, Double(finalPeak) / 1e9, physFootprint()))
}

// MARK: - validate (V3): the consumer path — MLXServeEngine (default policy) register → prepare → run → evict, phys_footprint
// at each step (the manifest's C-memory numbers). The bundles are given explicitly (the engine's MAT path is not exercised here).

/// A `--planner <dir>` whose config.json carries mlx-lm's `quantization` block (the -8bit bundle).
func plannerDirIsPreQuantized() -> Bool { plannerDirBits() != nil }
func plannerDirBits() -> Int? {
    guard let d = plannerDir, let c = try? ConfigIO.json(d.appendingPathComponent("config.json")), let q = c["quantization"] as? [String: Any] else { return nil }
    return q["bits"] as? Int
}

func validateLane(_ bundles: URL, _ promptsPath: String, _ outDir: String, audioDir: String? = nil) async throws {
    let bits = plannerBits ?? plannerDirBits()
    let quant: Quant = bits == 8 ? .int8 : (bits == 4 ? .int4 : (dtype == .float32 ? .fp32 : .bf16))
    let planner: ACEStepPlannerModel = plannerDir?.lastPathComponent.contains("0.6B") == true ? .lite0_6B : .standard1_7B
    let plannerOverride: URL? = plannerDirIsPreQuantized() && planner == .standard1_7B ? nil : plannerDir   // the -8bit folder resolves by name under .int8
    let out = url(outDir)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    print("validate: MLXServeEngine → register → prepare → run (\(quant), planner \(planner)) → evict; bundles \(bundles.path)")
    let engine = MLXServeEngine()   // .permissiveOnly — MIT + Apache-2.0 must admit with no advisory
    let t0 = Date()
    let configuration = ACEStepConfiguration(quant: quant, planner: planner, bundleDirectory: bundles, plannerDirectory: plannerOverride ?? plannerDir, vaeChunkFrames: vaeChunk)
    print("  planner bundle: \(configuration.resolvedBundles(storeRoot: nil)?.planner.lastPathComponent ?? "?")")
    print("  VAE chunk \(vaeChunk) frames")
    let id = try await engine.register(ACEStepPackage.registration, configuration: configuration, id: PackageID("ace-step-1.5"))
    let advisories = await engine.licenseAdvisories
    check("registered \(id.rawValue) under .permissiveOnly; licence advisories: \(advisories.count)", advisories.isEmpty)
    let needs = await engine.needsDownload(.music, package: id)
    check("needsDownload == \(needs) with the explicit bundles", !needs)
    let sampler = PhysSampler()
    let before = sampler.mark()
    try await engine.prepare(.music, package: id)
    Thread.sleep(forTimeInterval: 0.3)
    let afterLoad = sampler.mark()
    let charged = await engine.memory.residentBytes
    print(String(format: "  [VAL] prepared in %.1fs; phys_footprint %.2f → %.2f GB (resident +%.2f); engine charges %.2f GB", Date().timeIntervalSince(t0),
                 Double(before) / 1e9, Double(afterLoad) / 1e9, Double(afterLoad - before) / 1e9, Double(charged) / 1e9))
    let rows = try JSONDecoder().decode(PromptsFile.self, from: Data(contentsOf: url(promptsPath))).prompts
    let picks = (onlyIDs ?? ["p04", "p11"]).compactMap { p in rows.first { $0.id.hasPrefix(p) } }
    var peak = afterLoad
    var afterSong = afterLoad
    for (i, r) in picks.enumerated() {
        sampler.resetPeak()
        // the second song pins an anchor the planner must honour (a declared feature) and leaves the language to the planner
        let anchors: MusicAnchors? = i == 1 ? MusicAnchors(bpm: 112, key: "F major") : nil
        let request = MusicRequest(caption: r.caption, lyrics: r.lyrics, durationSeconds: r.duration, seed: r.seed,
                                   anchors: anchors, vocalLanguage: i == 1 ? nil : r.language)
        var phases: [String: Int] = [:]
        let t = Date()
        let response = try await RunProgress.$sink.withValue({ report in phases[report.phase.rawValue, default: 0] += 1 }) {
            try await engine.run(request, package: id)
        }
        let wall = Date().timeIntervalSince(t)
        Thread.sleep(forTimeInterval: 0.3)
        peak = max(peak, sampler.peak)
        afterSong = sampler.mark()
        guard let m = response as? MusicResponse else { check("MusicResponse", false, "got \(type(of: response))"); return }
        let samples = m.audio.data.dropFirst(44).withUnsafeBytes { raw -> [Float] in raw.bindMemory(to: Int16.self).map { Float($0) / 32768 } }
        let secs = Double(samples.count) / Double((m.audio.sampleRate ?? 48_000) * (m.audio.channels ?? 2))
        let rms = 20 * log10(max(sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(max(samples.count, 1))), 1e-9))
        let peakDB = 20 * log10(max(samples.map(abs).max() ?? 0, 1e-9))
        check(String(format: "  [RUN] %@: %.1fs audio (%d Hz × %d ch), peak %.1f / rms %.1f dBFS, %.1fs wall (RTF %.2f), anchors %@, truncated %@, phases %@, phys %.2f GB",
                     r.id as NSString, secs, m.audio.sampleRate ?? 0, m.audio.channels ?? 0, peakDB, rms, wall, secs / wall,
                     String(describing: m.anchors) as NSString, "\(m.truncated)" as NSString, "\(phases.sorted { $0.key < $1.key })" as NSString, Double(afterSong) / 1e9),
              secs > r.duration - 5 && secs <= r.duration + 5 && rms > -40 && rms < -3 && m.audio.sampleRate == 48_000 && m.audio.channels == 2
                  && (phases["plan"] ?? 0) > 100 && phases["denoise"] == 8 && (phases["decode"] ?? 0) >= 1 && !m.truncated)
        if let anchors { check("  pinned bpm honoured (\(m.anchors?.bpm ?? -1) vs \(anchors.bpm ?? -1))", m.anchors?.bpm == anchors.bpm) }
        try m.audio.data.write(to: out.appendingPathComponent("\(r.id)__engine_\(quant)\(planner == .lite0_6B ? "-lm06" : "")\(vaeChunk == 512 ? "" : "-c\(vaeChunk)").wav"))
    }
    // V3b: one request of each declared kind — a timbre reference, a cover, a repaint, an extension — each returning timestamps
    let probe = picks.first ?? rows[0]
    if let audioDir {
        let sourceWAV = try Data(contentsOf: url(audioDir).appendingPathComponent("source_30s.wav"))
        let referenceWAV = try Data(contentsOf: url(audioDir).appendingPathComponent("reference_30s.wav"))
        let source = Audio(format: .wav, data: sourceWAV), reference = Audio(format: .wav, data: referenceWAV)
        let v3b: [(String, MusicRequest, Double)] = [
            ("referenceAudio 30 s", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, durationSeconds: 30, seed: probe.seed, referenceAudio: reference, vocalLanguage: probe.language), 30),
            ("cover strength 0.5", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, seed: probe.seed, edit: MusicEdit(source: source, operation: .cover(strength: 0.5)), vocalLanguage: probe.language), 30),
            ("repaint 10–20 s", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, seed: probe.seed, edit: MusicEdit(source: source, operation: .repaint(start: 10, end: 20)), vocalLanguage: probe.language), 30),
            ("extend +10 s", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, seed: probe.seed, edit: MusicEdit(source: source, operation: .extend(trailing: 10)), vocalLanguage: probe.language), 40),
        ]
        for (label, request, expected) in v3b {
            sampler.resetPeak()
            var phases: [String: Int] = [:]
            let t = Date()
            let response = try await RunProgress.$sink.withValue({ report in phases[report.phase.rawValue, default: 0] += 1 }) {
                try await engine.run(request, package: id)
            }
            let wall = Date().timeIntervalSince(t)
            Thread.sleep(forTimeInterval: 0.3)
            peak = max(peak, sampler.peak)
            afterSong = sampler.mark()
            guard let m = response as? MusicResponse else { check("MusicResponse", false, "got \(type(of: response))"); return }
            let samples = m.audio.data.dropFirst(44).withUnsafeBytes { raw -> [Float] in raw.bindMemory(to: Int16.self).map { Float($0) / 32768 } }
            let secs = Double(samples.count) / Double((m.audio.sampleRate ?? 48_000) * (m.audio.channels ?? 2))
            let rms = 20 * log10(max(sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(max(samples.count, 1))), 1e-9))
            let stamps = m.timestamps ?? []
            let ordered = zip(stamps, stamps.dropFirst()).allSatisfy { $0.start <= $1.start } && stamps.allSatisfy { $0.start >= 0 && ($0.end ?? $0.start) <= secs + 0.5 }
            check(String(format: "  [RUN V3b] %@: %.1fs audio, rms %.1f dBFS, %.1fs wall, %d timestamp lines (first: %@), phases %@, phys peak %.2f GB",
                         label as NSString, secs, rms, wall, stamps.count, (stamps.first.map { "\($0.text) @ \($0.start)" } ?? "—") as NSString,
                         "\(phases.sorted { $0.key < $1.key })" as NSString, Double(sampler.peak) / 1e9),
                  abs(secs - expected) < 1 && rms > -40 && rms < -3 && !stamps.isEmpty && ordered && (phases["encode"] ?? 0) >= 1 && phases["denoise"] == 8)
            try m.audio.data.write(to: out.appendingPathComponent("\(probe.id)__engine_\(quant)_\(label.split(separator: " ").first!).wav"))
            if label.hasPrefix("extend"), let lrc = m.timestamps { try lrc.map { String(format: "[%.2f–%.2f] %@", $0.start, $0.end ?? -1, $0.text) }.joined(separator: "\n").write(to: out.appendingPathComponent("\(probe.id)__engine_\(quant)_extend.lrc.txt"), atomically: true, encoding: .utf8) }
        }
    }
    // the engine refuses an undeclared feature / out-of-vocabulary value BEFORE admission (MusicControls.refusal), and the
    // package refuses a durationSeconds that contradicts an edit
    for (label, bad) in [
        ("planning (undeclared)", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, planning: .full)),
        ("score (undeclared)", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, score: "X:1")),
        ("edit with an unreadable clip", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, edit: MusicEdit(source: Audio(format: .wav, data: Data([1, 2, 3])), operation: .cover(strength: 0.5)))),
        ("referenceAudio unreadable", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, referenceAudio: Audio(format: .wav, data: Data([1, 2, 3])))),
        ("edit + durationSeconds 90 (contradicts the source)", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, durationSeconds: 90, edit: MusicEdit(source: Audio(format: .wav, data: (try? Data(contentsOf: url(audioDir ?? "/nonexistent").appendingPathComponent("source_30s.wav"))) ?? Data([1])), operation: .cover(strength: 1)))),
        ("vocalLanguage xx", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, vocalLanguage: "xx")),
        ("durationSeconds 700", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, durationSeconds: 700)),
        ("anchors bpm 900", MusicRequest(caption: probe.caption, lyrics: probe.lyrics, anchors: MusicAnchors(bpm: 900))),
    ] {
        do { _ = try await engine.run(bad, package: id); check("  refused: \(label)", false, "ADMITTED") }
        catch { check("  refused: \(label)", true, String(describing: error).prefix(110).description) }
    }
    await engine.evict(package: id)
    Thread.sleep(forTimeInterval: 0.3)
    let afterEvict = sampler.mark()
    _ = sampler.stop()
    print(String(format: "  [VAL] %@%@: resident %.2f GB (phys after prepare − before), %.2f GB after a song; peak phys %.2f GB over the songs (activation %.2f GB over the post-song resident, %.2f over the post-prepare); after evict %.2f GB",
                 "\(quant)" as NSString, (planner == .lite0_6B ? " + 0.6B planner" : "") as NSString, Double(afterLoad - before) / 1e9, Double(afterSong - before) / 1e9,
                 Double(peak) / 1e9, Double(peak) / 1e9 - Double(afterSong) / 1e9, Double(peak) / 1e9 - Double(afterLoad) / 1e9, Double(afterEvict) / 1e9))
    print(failures == 0 ? "RESULT: PASS" : "RESULT: FAIL (\(failures) checks)")
}

do {
    switch cmd {
    case "render":
        guard rest.count == 3 else { print("render <bundles> <request.json> <out.wav>"); exit(2) }
        try renderLane(url(rest[0]), rest[1], rest[2])
    case "batch":
        guard rest.count == 3 else { print("batch <bundles> <prompts.json> <outdir>"); exit(2) }
        try batchLane(url(rest[0]), rest[1], rest[2])
    case "planner8":
        guard rest.count == 3 || rest.count == 6 else { print("planner8 <golden> <bundles> <8bit-dir> [<prompts.json> <reference.wav> <out.wav>]"); exit(2) }
        let g = url(rest[0]), b = Bundles(url(rest[1]))
        print("golden \(g.lastPathComponent); dtype \(dtype); TF32 \(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "default")")
        try planner8Lane(g, b, int8Dir: url(rest[2]), song: rest.count == 6 ? (rest[3], rest[4], rest[5]) : nil)
        print(String(format: "memory: MLX peak %.2f GB; phys_footprint %.2f GB", Double(Memory.peakMemory) / 1e9, physFootprint()))
        print(failures == 0 ? "RESULT: PASS" : "RESULT: FAIL (\(failures) checks)")
        exit(failures == 0 ? 0 : 1)
    case "validate":
        guard rest.count == 3 || rest.count == 4 else { print("validate <bundles> <prompts.json> <outdir> [audio-dir]"); exit(2) }
        try await validateLane(url(rest[0]), rest[1], rest[2], audioDir: rest.count == 4 ? rest[3] : nil)
        exit(failures == 0 ? 0 : 1)
    default:
        guard rest.count >= 2 else { print("<lane> <golden> <bundles> [wav]"); exit(2) }
        let g = url(rest[0]), b = Bundles(url(rest[1]))
        print("golden \(g.lastPathComponent); dtype \(dtype); device \(useCPU ? "cpu" : "gpu"); TF32 \(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "default")")
        let v3b = ["resample", "encoder", "timbre", "tokenize", "cover", "repaint", "extend", "lrc"]
        let lanes: [String] = cmd == "all" ? ["tokenizer", "text", "cond", "planner", "kvcache", "fsm", "detok", "dit", "solve", "vae", "e2e"] : (cmd == "v3b" ? v3b : [cmd])
        for lane in lanes {
            switch lane {
            case "resample": try resampleLane(g, b)
            case "encoder": try encoderLane(g, b)
            case "timbre": try timbreLane(g, b)
            case "tokenize": try tokenizeLane(g, b)
            case "cover": try coverLane(g, b)
            case "repaint": try repaintLane(g, b)
            case "extend": try extendLane(g, b)
            case "lrc": try lrcLane(g, b)
            case "tokenizer": try tokenizerLane(g, b)
            case "text": try textLane(g, b)
            case "cond": try condLane(g, b)
            case "planner": try plannerLane(g, b)
            case "kvcache": try kvcacheLane(g, b)
            case "fsm": try fsmLane(g, b)
            case "detok": try detokLane(g, b)
            case "dit": try ditLane(g, b)
            case "solve": try solveLane(g, b)
            case "vae": try vaeLane(g, b)
            case "e2e": try e2eLane(g, b, wav: rest.count > 2 ? rest[2] : nil)
            default: print("unknown lane \(lane)"); exit(2)
            }
        }
        print(String(format: "memory: MLX peak %.2f GB; phys_footprint %.2f GB", Double(Memory.peakMemory) / 1e9, physFootprint()))
        print(failures == 0 ? "RESULT: PASS" : "RESULT: FAIL (\(failures) checks)")
        exit(failures == 0 ? 0 : 1)
    }
} catch {
    print("error: \(error)"); exit(1)
}
