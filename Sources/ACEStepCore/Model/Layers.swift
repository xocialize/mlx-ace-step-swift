// Layers.swift — modeling_acestep_v15_turbo.py's shared pieces: `AceStepAttention` (GQA 16/8 × 128 with q/k RMSNorm; self-
// attention with rotate-half RoPE, or cross-attention over cached K/V), `Qwen3MLP`, `AceStepEncoderLayer` (pre-norm,
// bidirectional), `create_4d_mask` (bidirectional full / sliding-window additive masks, -1e9 as upstream's MLX decoder uses)
// and `Qwen3RotaryEmbedding` (inv_freq = 1/θ^(2i/d), cos/sin of [freqs ; freqs], computed in fp32 MLX ops like dit_model.py).
import Foundation
import MLX
import MLXFast
import MLXNN

/// Rotate-half RoPE tables for one sequence length (positions 0 … L−1), `[1, 1, L, head_dim]`.
public struct RotaryTable {
    public let cos: MLXArray
    public let sin: MLXArray
    public init(length: Int, headDim: Int, theta: Float) {
        let invFreq = 1.0 / MLX.pow(MLXArray(theta), MLXArray(stride(from: 0, to: headDim, by: 2).map { Int32($0) }).asType(.float32) / Float(headDim))
        let positions = MLXArray(Int32(0) ..< Int32(length)).asType(.float32)
        let freqs = positions.reshaped(length, 1) * invFreq.reshaped(1, -1)
        let both = concatenated([freqs, freqs], axis: -1)
        cos = MLX.cos(both).reshaped(1, 1, length, headDim)
        sin = MLX.sin(both).reshaped(1, 1, length, headDim)
    }
    /// `apply_rotary_pos_emb`: `x·cos + rotate_half(x)·sin`, x `[B, H, L, D]`.
    public func apply(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos.asType(x.dtype) + rotated * sin.asType(x.dtype)
    }
}

public enum AttentionMasks {
    /// `create_4d_mask(is_causal=False)`: `.none` when nothing is masked, else an additive `[B or 1, 1, L, L]` with −1e9.
    /// `window`: bidirectional sliding window (|i−j| ≤ window); `keep`: `[B, L]` key padding (1 = real).
    public static func bidirectional(length: Int, window: Int?, keep: MLXArray? = nil, dtype: DType) -> MLXFast.ScaledDotProductAttentionMaskMode {
        var allowed: MLXArray? = nil
        if let window, window < length - 1 {
            let idx = MLXArray(Int32(0) ..< Int32(length))
            let distance = abs(idx.reshaped(length, 1) - idx.reshaped(1, length))
            allowed = (distance .<= MLXArray(Int32(window))).reshaped(1, 1, length, length)
        }
        if let keep, keep.min().item(Int32.self) <= 0 {
            let keys = (keep.asType(.int32) .> 0).reshaped(keep.dim(0), 1, 1, length)
            allowed = allowed.map { $0 .&& keys } ?? broadcast(keys, to: [keep.dim(0), 1, length, length])
        }
        guard let allowed else { return .none }
        return .array(MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9))).asType(dtype))
    }
}

/// `AceStepAttention`: self-attention (RoPE) or cross-attention (K/V from encoder states, cacheable).
public final class AceStepAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: ACEStepConfig) {
        heads = c.num_attention_heads; kvHeads = c.num_key_value_heads; headDim = c.head_dim
        scale = 1 / Float(c.head_dim).squareRoot()
        _qProj.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: c.attention_bias)
        _kProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: c.attention_bias)
        _vProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: c.attention_bias)
        _oProj.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: c.attention_bias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rms_norm_eps))
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rms_norm_eps))
    }

    /// Cross-attention keys / values for `encoder` `[B, Lenc, D]` (normed K, raw V), reusable across diffusion steps.
    public func crossKeysValues(_ encoder: MLXArray) -> (MLXArray, MLXArray) {
        let (b, l) = (encoder.dim(0), encoder.dim(1))
        let k = kNorm(kProj(encoder).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(encoder).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        return (k, v)
    }

    /// Self-attention with `rope`, or cross-attention when `keysValues` is given (no RoPE, as upstream).
    public func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, rope: RotaryTable? = nil,
                               keysValues: (MLXArray, MLXArray)? = nil) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var k: MLXArray, v: MLXArray
        if let keysValues { (k, v) = keysValues } else {
            k = kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
            v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
            if let rope { q = rope.apply(q); k = rope.apply(k) }
        }
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

/// `Qwen3MLP` as the ACE-Step modules use it.
public final class AceStepMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(_ c: ACEStepConfig) {
        _gateProj.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _upProj.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _downProj.wrappedValue = Linear(c.intermediate_size, c.hidden_size, bias: false)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

/// `AceStepEncoderLayer`: pre-norm self-attention + MLP, bidirectional; `layer_types[i]` picks the sliding or full mask.
public final class AceStepEncoderLayer: Module {
    public let sliding: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: AceStepAttention
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: AceStepMLP
    init(_ c: ACEStepConfig, index: Int) {
        sliding = c.isSliding(layer: index)
        _selfAttn.wrappedValue = AceStepAttention(c)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _mlp.wrappedValue = AceStepMLP(c)
    }
    func callAsFunction(_ x: MLXArray, full: MLXFast.ScaledDotProductAttentionMaskMode, slidingMask: MLXFast.ScaledDotProductAttentionMaskMode,
                        rope: RotaryTable) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: sliding ? slidingMask : full, rope: rope)
        return h + mlp(postAttentionLayernorm(h))
    }
}

/// A stack of encoder layers run the way every AceStep encoder runs them: full + sliding masks from one `keep` mask.
enum EncoderStack {
    static func run(_ layers: [AceStepEncoderLayer], _ x: MLXArray, keep: MLXArray?, config c: ACEStepConfig) -> MLXArray {
        let length = x.dim(1)
        let full = AttentionMasks.bidirectional(length: length, window: nil, keep: keep, dtype: x.dtype)
        let sliding = AttentionMasks.bidirectional(length: length, window: c.use_sliding_window ? c.sliding_window : nil, keep: keep, dtype: x.dtype)
        let rope = RotaryTable(length: length, headDim: c.head_dim, theta: Float(c.rope_theta))
        var h = x
        for layer in layers { h = layer(h, full: full, slidingMask: sliding, rope: rope) }
        return h
    }
}
