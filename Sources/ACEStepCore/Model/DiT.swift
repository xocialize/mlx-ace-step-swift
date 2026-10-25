// DiT.swift — `AceStepDiTModel` (modeling_acestep_v15_turbo.py; the official MLX port is acestep/models/mlx/dit_model.py and
// this follows it op for op): `TimestepEmbedding` ×2 (t and t−r), Conv1d patch embedding (k2 s2, 192→2048), 24 `AceStepDiTLayer`s
// alternating sliding-128 / full bidirectional self-attention with adaLN (6-way scale_shift_table + temb), cross-attention to
// the packed conditions (K/V computed once per condition, reused every step), SwiGLU, the 2-way output adaLN, ConvTranspose1d
// de-patchify (2048→64) and the crop back to the input length.
import Foundation
import MLX
import MLXFast
import MLXNN

public final class TimestepEmbedding: Module {
    let inChannels: Int, dimension: Int, scale: Float
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "time_proj") var timeProj: Linear
    init(inChannels: Int = 256, dimension: Int, scale: Float = 1000) {
        self.inChannels = inChannels; self.dimension = dimension; self.scale = scale
        _linear1.wrappedValue = Linear(inChannels, dimension, bias: true)
        _linear2.wrappedValue = Linear(dimension, dimension, bias: true)
        _timeProj.wrappedValue = Linear(dimension, dimension * 6, bias: true)
    }
    /// `timestep_embedding`: `t·1000`, freqs = exp(−ln 10000 · i/half), [cos ; sin] (cos first).
    func sinusoidal(_ t: MLXArray) -> MLXArray {
        let half = inChannels / 2
        let freqs = exp(-Float(log(10000.0)) * MLXArray(Int32(0) ..< Int32(half)).asType(.float32) / Float(half))
        let args = (t * scale)[0..., .newAxis].asType(.float32) * freqs[.newAxis, 0...]
        return concatenated([cos(args), sin(args)], axis: -1)
    }
    /// t `[B]` → (temb `[B, D]`, timestep_proj `[B, 6, D]`).
    func callAsFunction(_ t: MLXArray) -> (MLXArray, MLXArray) {
        let temb = linear2(silu(linear1(sinusoidal(t).asType(t.dtype))))
        let proj = timeProj(silu(temb)).reshaped(t.dim(0), 6, -1)
        return (temb, proj)
    }
}

public final class AceStepDiTLayer: Module {
    public let sliding: Bool
    @ModuleInfo(key: "self_attn_norm") var selfAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: AceStepAttention
    @ModuleInfo(key: "cross_attn_norm") var crossAttnNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: AceStepAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: AceStepMLP
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    init(_ c: ACEStepConfig, index: Int) {
        sliding = c.isSliding(layer: index)
        _selfAttnNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _selfAttn.wrappedValue = AceStepAttention(c)
        _crossAttnNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _crossAttn.wrappedValue = AceStepAttention(c)
        _mlpNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _mlp.wrappedValue = AceStepMLP(c)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, c.hidden_size])
    }
    func callAsFunction(_ x: MLXArray, temb: MLXArray, selfMask: MLXFast.ScaledDotProductAttentionMaskMode, rope: RotaryTable,
                        cross: (MLXArray, MLXArray)) -> MLXArray {
        let parts = split(scaleShiftTable + temb, parts: 6, axis: 1)      // each [B, 1, D]
        let (shiftMSA, scaleMSA, gateMSA, cShift, cScale, cGate) = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
        var h = x
        let normed = selfAttnNorm(h) * (1 + scaleMSA) + shiftMSA
        h = h + selfAttn(normed, mask: sliding ? selfMask : .none, rope: rope) * gateMSA
        h = h + crossAttn(crossAttnNorm(h), mask: .none, keysValues: cross)
        let normed2 = mlpNorm(h) * (1 + cScale) + cShift
        h = h + mlp(normed2) * cGate
        return h
    }
}

/// Cross-attention K/V per layer for one condition sequence (upstream's `EncoderDecoderCache` after the first step).
public struct CrossAttentionCache { public var layers: [(MLXArray, MLXArray)] }

public final class AceStepDiTModel: Module {
    public let config: ACEStepConfig
    @ModuleInfo(key: "layers") var layers: [AceStepDiTLayer]
    @ModuleInfo(key: "proj_in") var projIn: Conv1d
    @ModuleInfo(key: "time_embed") var timeEmbed: TimestepEmbedding
    @ModuleInfo(key: "time_embed_r") var timeEmbedR: TimestepEmbedding
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: Linear
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm
    @ModuleInfo(key: "proj_out") var projOut: ConvTransposed1d
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(_ c: ACEStepConfig) {
        config = c
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { AceStepDiTLayer(c, index: $0) }
        _projIn.wrappedValue = Conv1d(inputChannels: c.in_channels, outputChannels: c.hidden_size, kernelSize: c.patch_size, stride: c.patch_size)
        _timeEmbed.wrappedValue = TimestepEmbedding(dimension: c.hidden_size)
        _timeEmbedR.wrappedValue = TimestepEmbedding(dimension: c.hidden_size)
        _conditionEmbedder.wrappedValue = Linear(c.hidden_size, c.hidden_size, bias: true)
        _normOut.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _projOut.wrappedValue = ConvTransposed1d(inputChannels: c.hidden_size, outputChannels: c.audio_acoustic_hidden_dim,
                                                 kernelSize: c.patch_size, stride: c.patch_size)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, c.hidden_size])
    }

    /// `condition_embedder` + every layer's cross K/V for `encoderHiddenStates` `[B, Lenc, 2048]` — once per condition.
    public func crossCache(_ encoderHiddenStates: MLXArray) -> CrossAttentionCache {
        let projected = conditionEmbedder(encoderHiddenStates)
        return CrossAttentionCache(layers: layers.map { $0.crossAttn.crossKeysValues(projected) })
    }

    /// Velocity `[B, T, 64]` for `hiddenStates` `[B, T, 64]` at `timestep` / `timestepR` `[B]` with `contextLatents` `[B, T, 128]`.
    public func callAsFunction(_ hiddenStates: MLXArray, timestep: MLXArray, timestepR: MLXArray, contextLatents: MLXArray,
                               cache: CrossAttentionCache) -> MLXArray {
        let (tembT, projT) = timeEmbed(timestep)
        let (tembR, projR) = timeEmbedR(timestep - timestepR)
        let temb = tembT + tembR, timestepProj = projT + projR
        var h = concatenated([contextLatents, hiddenStates], axis: -1)
        let originalLength = h.dim(1)
        let pad = (config.patch_size - originalLength % config.patch_size) % config.patch_size
        if pad > 0 { h = concatenated([h, MLXArray.zeros([h.dim(0), pad, h.dim(2)], dtype: h.dtype)], axis: 1) }
        h = projIn(h)
        let length = h.dim(1)
        let rope = RotaryTable(length: length, headDim: config.head_dim, theta: Float(config.rope_theta))
        let slidingMask = AttentionMasks.bidirectional(length: length, window: config.use_sliding_window ? config.sliding_window : nil, dtype: h.dtype)
        for (i, layer) in layers.enumerated() {
            h = layer(h, temb: timestepProj, selfMask: slidingMask, rope: rope, cross: cache.layers[i])
        }
        let out = split(scaleShiftTable + temb.expandedDimensions(axis: 1), parts: 2, axis: 1)
        h = normOut(h) * (1 + out[1]) + out[0]
        return projOut(h)[0..., ..<originalLength, 0...]
    }
}
