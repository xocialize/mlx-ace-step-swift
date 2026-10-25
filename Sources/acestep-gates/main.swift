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
// Options: --cpu, --dtype float32|bfloat16, --planner-bits N, --dit-bits N, --cache-gb G (default 1).
import Darwin
import Foundation
import MLX
import MLXNN
import MLXRandom
import ACEStepCore

var args = Array(CommandLine.arguments.dropFirst())
var useCPU = false
var dtype: DType = .float32
var plannerBits: Int? = nil, ditBits: Int? = nil
var cacheGB = 1.0
while let a = args.first, a.hasPrefix("--") {
    args.removeFirst()
    switch a {
    case "--cpu": useCPU = true
    case "--dtype": dtype = ["bfloat16", "bf16"].contains(args.removeFirst()) ? .bfloat16 : .float32
    case "--planner-bits": plannerBits = Int(args.removeFirst())
    case "--dit-bits": ditBits = Int(args.removeFirst())
    case "--cache-gb": cacheGB = Double(args.removeFirst()) ?? 1
    default: fatalError("unknown option \(a)")
    }
}
guard let cmd = args.first else { print("usage: acestep-gates [opts] tokenizer|text|cond|planner|fsm|detok|dit|solve|vae|e2e|all|render …"); exit(2) }
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
    lazy var plannerTokenizer: BPETokenizer = try! BPETokenizer(directory: dir.appendingPathComponent("ACE-Step-1.5-planner-1.7B-bf16"))
    lazy var planner: Planner = { let t = Date(); let p = try! Planner(directory: dir.appendingPathComponent("ACE-Step-1.5-planner-1.7B-bf16"), dtype: dtype); if let b = plannerBits { p.model.quantizeLayers(groupSize: 64, bits: b) }; print("  loaded planner in \(elapsed(t))"); return p }()
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
    let lat = DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: try npy(g, "noise"), settings: settings)
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
    let chunked = vae.decodeChunked(lat, chunk: 512, overlap: 64); eval(chunked)
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
    let lat = DiffusionSampler.solve(decoder: m.decoder, cache: m.decoder.crossCache(enc), context: ctx, noise: DiffusionSampler.noise(seed: r.seed, frames: r.frames, channels: 64), settings: settings)
    eval(lat)
    print("  solve in \(elapsed(t1))")
    compare("latents (swift chain)", lat, try npy(g, "latents"), relL2Limit: 5e-3)
    let t2 = Date()
    let audio = b.vae.decodeChunked(lat, chunk: 512, overlap: 64)[0]; eval(audio)
    print("  VAE in \(elapsed(t2)); total \(elapsed(t0))")
    compare("waveform vs oracle (MLX path)", audio, try npy(g, "audio_mlx_chunked"), snrLimit: 75)
    let finished = ACEStepPipeline.finish(audio, normalize: true, db: -1)
    compare("final (−1 dBFS) vs oracle final (info)", finished, try npy(g, "audio_final"))
    if let wav { try WAV.write16(finished.asArray(Float.self), channels: 2, sampleRate: 48000, to: url(wav)); print("  wrote \(wav)") }
}

struct RenderRequest: Decodable { let caption: String; let lyrics: String; let language: String?; let duration: Double?; let seed: UInt64?; let shift: Double? }
func renderLane(_ bundles: URL, _ requestPath: String, _ out: String) throws {
    print("render (a sampled generation; not a gate)")
    let rr = try JSONDecoder().decode(RenderRequest.self, from: Data(contentsOf: url(requestPath)))
    let t0 = Date()
    let pipe = try ACEStepPipeline(bundles: bundles, dtype: dtype, plannerBits: plannerBits, ditBits: ditBits)
    print(String(format: "  loaded in %@; phys %.2f GB; MLX active %.2f GB", elapsed(t0) as NSString, physFootprint(), Double(Memory.activeMemory) / 1e9))
    var req = ACEStepRequest(caption: rr.caption, lyrics: rr.lyrics, language: rr.language ?? "en", duration: rr.duration, seed: rr.seed ?? 0)
    if let sh = rr.shift { req.sampler.shift = sh }
    let song = try pipe.generate(req) { phase, f in FileHandle.standardError.write("\r  \(phase) \(Int(f * 100))%   ".data(using: .utf8)!) }
    print("")
    try WAV.write16(song.audio, channels: 2, sampleRate: 48000, to: url(out))
    print("  metadata: \(song.metadata)")
    print("  codes: \(song.codes.count) → \(song.frames) frames = \(Double(song.frames) / 25) s")
    print("  timings: \(song.timings.sorted { $0.key < $1.key }.map { "\($0.key) \(String(format: "%.1f", $0.value))s" }.joined(separator: ", "))")
    print(String(format: "  memory: MLX peak %.2f GB, active %.2f GB; phys_footprint now %.2f GB", Double(Memory.peakMemory) / 1e9, Double(Memory.activeMemory) / 1e9, physFootprint()))
    print("  wrote \(out)")
}

do {
    switch cmd {
    case "render":
        guard rest.count == 3 else { print("render <bundles> <request.json> <out.wav>"); exit(2) }
        try renderLane(url(rest[0]), rest[1], rest[2])
    default:
        guard rest.count >= 2 else { print("<lane> <golden> <bundles> [wav]"); exit(2) }
        let g = url(rest[0]), b = Bundles(url(rest[1]))
        print("golden \(g.lastPathComponent); dtype \(dtype); device \(useCPU ? "cpu" : "gpu"); TF32 \(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "default")")
        let lanes: [String] = cmd == "all" ? ["tokenizer", "text", "cond", "planner", "fsm", "detok", "dit", "solve", "vae", "e2e"] : [cmd]
        for lane in lanes {
            switch lane {
            case "tokenizer": try tokenizerLane(g, b)
            case "text": try textLane(g, b)
            case "cond": try condLane(g, b)
            case "planner": try plannerLane(g, b)
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
