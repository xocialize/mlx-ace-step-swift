// AudioTokens.swift — the 5 Hz semantic codes and the modules either side: `AceStepAudioTokenizer` (64→2048 proj, the
// 2-layer `AttentionPooler` with its CLS token over 5-frame patches, `ResidualFSQ` levels [8,8,8,5,5,5] one quantizer with
// preserve_symmetry + hard clamp — vector_quantize_pytorch 1.32.0) and `AudioTokenDetokenizer` (2048→2048 embed, 5 learned
// patch tokens per code, 2 encoder layers, norm, proj_out 2048→64 → 25 Hz "LM hints").
import Foundation
import MLX
import MLXFast
import MLXNN

/// `ResidualFSQ(dim=2048, levels=[8,8,8,5,5,5], num_quantizers=1)` with its projections.
public final class ResidualFSQ: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "project_out") var projectOut: Linear
    public let levels: [Int]
    /// Constant tables kept out of the Module's parameter reflection (they are not checkpoint keys).
    final class Tables {
        let levelsF: MLXArray         // [1, 1, d]
        let basis: MLXArray           // [1, 1, d] int32 cumprod
        let softClamp: MLXArray       // 1 + 1/(L−1)
        init(_ levels: [Int]) {
            let d = levels.count
            levelsF = MLXArray(levels.map { Float($0) }, [1, 1, d])
            var b: [Int32] = [], run: Int32 = 1
            for l in levels { b.append(run); run *= Int32(l) }
            basis = MLXArray(b, [1, 1, d])
            softClamp = MLXArray(levels.map { 1 + 1 / Float($0 - 1) }, [1, 1, d])
        }
    }
    let tables: Tables
    var levelsF: MLXArray { tables.levelsF }
    var basis: MLXArray { tables.basis }
    var softClamp: MLXArray { tables.softClamp }
    init(_ c: ACEStepConfig) {
        levels = c.fsq_input_levels
        let d = levels.count
        _projectIn.wrappedValue = Linear(c.fsq_dim, d, bias: true)
        _projectOut.wrappedValue = Linear(d, c.fsq_dim, bias: true)
        tables = Tables(levels)
    }
    public var codebookSize: Int { levels.reduce(1, *) }

    /// `get_output_from_indices(indices [B, T, 1])`: level indices `(idx // basis) % levels` → `zhat·2/(L−1) − 1` → project_out.
    public func output(fromIndices codes: MLXArray, dtype: DType) -> MLXArray {
        let idx = codes.asType(.int32).expandedDimensions(axis: -1)             // [B, T, 1]
        // `indices // basis % levels` — MLX's `/` on integer arrays is true division, so floorDivide keeps the level indices integral.
        let level = floorDivide(idx, basis) % MLXArray(levels.map { Int32($0) }, [1, 1, levels.count])
        let z = level.asType(.float32) * (2 / (levelsF - 1)) - 1
        return projectOut(z.asType(dtype))
    }

    /// `forward(x [B, T, 2048])` → (quantized `[B, T, 2048]`, indices `[B, T]`): soft clamp, symmetry-preserving bound with
    /// the hard clamp, `codes_to_indices`.
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var z = projectIn(x).asType(.float32)
        z = tanh(z / softClamp) * softClamp
        z = clip(z, min: -1, max: 1)
        let bracket = floor((levelsF - 1) * (z + 1) / 2 + 0.5)
        let codes = (2 / (levelsF - 1)) * bracket - 1
        let indices = ((codes + 1) / (2 / (levelsF - 1)) * basis.asType(.float32)).sum(axis: -1).round().asType(.int32)
        return (projectOut(codes.asType(x.dtype)), indices)
    }
}

/// `AttentionPooler`: `[B, T, P, D]` → `[B, T, D]` via a prepended CLS token through 2 encoder layers.
public final class AttentionPooler: Module {
    let config: ACEStepConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_token") var specialToken: MLXArray
    @ModuleInfo(key: "layers") var layers: [AceStepEncoderLayer]
    init(_ c: ACEStepConfig) {
        config = c
        _embedTokens.wrappedValue = Linear(c.hidden_size, c.hidden_size, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _specialToken.wrappedValue = MLXArray.zeros([1, 1, c.hidden_size])
        _layers.wrappedValue = (0 ..< c.num_attention_pooler_hidden_layers).map { AceStepEncoderLayer(c, index: $0) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, p, d) = (x.dim(0), x.dim(1), x.dim(2), config.hidden_size)
        var h = embedTokens(x)
        h = concatenated([broadcast(specialToken.asType(h.dtype), to: [b, t, 1, d]), h], axis: 2).reshaped(b * t, p + 1, d)
        h = norm(EncoderStack.run(layers, h, keep: nil, config: config))
        return h[0..., 0, 0...].reshaped(b, t, d)
    }
}

public final class AceStepAudioTokenizer: Module {
    @ModuleInfo(key: "audio_acoustic_proj") var audioAcousticProj: Linear
    @ModuleInfo(key: "attention_pooler") var attentionPooler: AttentionPooler
    @ModuleInfo(key: "quantizer") public var quantizer: ResidualFSQ
    init(_ c: ACEStepConfig) {
        _audioAcousticProj.wrappedValue = Linear(c.audio_acoustic_hidden_dim, c.hidden_size, bias: true)
        _attentionPooler.wrappedValue = AttentionPooler(c)
        _quantizer.wrappedValue = ResidualFSQ(c)
    }
    /// `forward(x [B, T5, 5, 64])` → (quantized `[B, T5, 2048]`, indices `[B, T5]`).
    public func callAsFunction(_ patches: MLXArray) -> (MLXArray, MLXArray) {
        quantizer(attentionPooler(audioAcousticProj(patches)))
    }
}

public final class AudioTokenDetokenizer: Module {
    let config: ACEStepConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_tokens") var specialTokens: MLXArray
    @ModuleInfo(key: "layers") var layers: [AceStepEncoderLayer]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    init(_ c: ACEStepConfig) {
        config = c
        _embedTokens.wrappedValue = Linear(c.hidden_size, c.hidden_size, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _specialTokens.wrappedValue = MLXArray.zeros([1, c.pool_window_size, c.hidden_size])
        _layers.wrappedValue = (0 ..< c.num_attention_pooler_hidden_layers).map { AceStepEncoderLayer(c, index: $0) }
        _projOut.wrappedValue = Linear(c.hidden_size, c.audio_acoustic_hidden_dim, bias: true)
    }
    /// quantized `[B, T5, 2048]` → `[B, T5·5, 64]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, p, d) = (x.dim(0), x.dim(1), config.pool_window_size, config.hidden_size)
        var h = embedTokens(x).expandedDimensions(axis: 2)                       // [B, T, 1, D]
        h = broadcast(h, to: [b, t, p, d]) + specialTokens.asType(h.dtype).reshaped(1, 1, p, d)
        h = h.reshaped(b * t, p, d)
        h = projOut(norm(EncoderStack.run(layers, h, keep: nil, config: config)))
        return h.reshaped(b, t * p, config.audio_acoustic_hidden_dim)
    }
}
