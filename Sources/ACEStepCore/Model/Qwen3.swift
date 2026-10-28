// Qwen3.swift — mlx-lm's `qwen3.py` (Attention / MLP / TransformerBlock / Qwen3Model / Model) with checkpoint-compatible names,
// used twice: the 5 Hz planner (acestep-5Hz-lm-1.7B, tied embedding → logits) and the Qwen3-Embedding-0.6B text encoder (the
// `Qwen3Model` body: last hidden states after `norm`, causal attention, as `AutoModel(...).last_hidden_state` yields them).
// Facts kept from upstream: q/k RMSNorm per head BEFORE RoPE; RoPE rotate-half (MLXFast traditional: false), θ 1e6; RMSNorm
// `x · rsqrt(mean(x²) + eps) · w`; causal mask for prefills, none for a single cached token (mlx-lm `create_attention_mask`).
import Foundation
import MLX
import MLXFast
import MLXNN

/// Append-only K/V store for one layer, grown in 512-slot steps (the mlx-swift-lm idiom).
public final class KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var offset = 0
    let step = 512
    public init() {}

    /// Appends k, v (B, Hkv, T, D) and returns the used prefix views (B, Hkv, offset+T, D).
    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let t = k.dim(2), prev = offset
        if keys == nil || prev + t > keys!.dim(2) {
            let needed = ((prev + t + step - 1) / step) * step
            let shape = [k.dim(0), k.dim(1), needed, k.dim(3)]
            var nk = MLXArray.zeros(shape, dtype: k.dtype), nv = MLXArray.zeros(shape, dtype: v.dtype)
            if let ok = keys, let ov = values, prev > 0 {
                nk[0..., 0..., ..<prev, 0...] = ok[0..., 0..., ..<prev, 0...]
                nv[0..., 0..., ..<prev, 0...] = ov[0..., 0..., ..<prev, 0...]
            }
            keys = nk; values = nv
        }
        keys![0..., 0..., prev ..< (prev + t), 0...] = k
        values![0..., 0..., prev ..< (prev + t), 0...] = v
        offset = prev + t
        return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...])
    }
}

public final class Qwen3Attention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, theta: Float, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: Qwen3Config) {
        heads = c.num_attention_heads; kvHeads = c.num_key_value_heads; headDim = c.head_dim
        theta = Float(c.rope_theta); scale = 1 / Float(c.head_dim).squareRoot()
        _qProj.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rms_norm_eps))
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(c.rms_norm_eps))
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        var q = qNorm(qProj(x).reshaped(b, t, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: offset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: offset)
        if let cache { (k, v) = cache.update(k, v) }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = t > 1 ? .causal : .none
        precondition(!(t > 1 && offset > 0), "chunked prefill continues a causal prefix: not supported here (prefill the chunks whole)")
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }

    /// Prefill continuation: a causal mask offset by the cache (positions `offset …`), for chunked prefills.
    func callAsFunction(_ x: MLXArray, cache: KVCache, causalOffsetMask: Bool) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let offset = cache.offset
        var q = qNorm(qProj(x).reshaped(b, t, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: offset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: theta, scale: 1, offset: offset)
        (k, v) = cache.update(k, v)
        let total = offset + t
        let rows = MLXArray(Int32(offset) ..< Int32(total)).reshaped(t, 1), cols = MLXArray(Int32(0) ..< Int32(total)).reshaped(1, total)
        let allowed = rows .>= cols
        let mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9))).asType(q.dtype).reshaped(1, 1, t, total)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

public final class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(_ hidden: Int, _ intermediate: Int) {
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

public final class Qwen3Block: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    init(_ c: Qwen3Config) {
        _selfAttn.wrappedValue = Qwen3Attention(c)
        _mlp.wrappedValue = Qwen3MLP(c.hidden_size, c.intermediate_size)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?, chunked: Bool = false) -> MLXArray {
        let a = (chunked && cache != nil) ? selfAttn(inputLayernorm(x), cache: cache!, causalOffsetMask: true) : selfAttn(inputLayernorm(x), cache: cache)
        let h = x + a
        return h + mlp(postAttentionLayernorm(h))
    }
}

/// `Qwen3Model`: embed_tokens → layers → norm. The planner's `Model` wraps it with the tied `as_linear` head.
public final class Qwen3Model: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Qwen3Block]
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    init(_ c: Qwen3Config) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocab_size, dimensions: c.hidden_size)
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { _ in Qwen3Block(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
    }
    /// Hidden states after `norm` (B, L, D). `chunked`: this call continues a cached prefix (causal mask with the offset).
    public func callAsFunction(_ ids: MLXArray, caches: [KVCache]?, chunked: Bool = false) -> MLXArray {
        var h = embedTokens(ids)
        for (i, layer) in layers.enumerated() { h = layer(h, cache: caches?[i], chunked: chunked) }
        return norm(h)
    }
}

/// mlx-lm `Model` (the mlx-community layout: `model.*`, tied embedding, no lm_head).
public final class Qwen3LM: Module {
    public let config: Qwen3Config
    @ModuleInfo(key: "model") public var model: Qwen3Model
    public init(_ c: Qwen3Config) { config = c; _model.wrappedValue = Qwen3Model(c) }
    public var dtype: DType { model.embedTokens.weight.dtype }

    public func newCaches() -> [KVCache] { model.layers.map { _ in KVCache() } }

    /// Logits (B, L, V) for `ids` (B, L): `embed_tokens.as_linear(norm(h))`.
    public func logits(_ ids: MLXArray, caches: [KVCache]?, chunked: Bool = false) -> MLXArray {
        model.embedTokens.asLinear(model(ids, caches: caches, chunked: chunked))
    }
    /// The last position's logits only (V), fp32.
    public func lastLogits(_ ids: MLXArray, caches: [KVCache]?, chunked: Bool = false) -> MLXArray {
        let h = model(ids, caches: caches, chunked: chunked)
        return model.embedTokens.asLinear(h[0..., (h.dim(1) - 1)..., 0...])[0, 0].asType(.float32)
    }
    /// Quantise the layers' Linears (the V2b planner tiers); the embedding / tied head stays as shipped.
    public func quantizeLayers(groupSize: Int, bits: Int) {
        quantize(model: self, groupSize: groupSize, bits: bits) { path, m in m is Linear && path.hasPrefix("model.layers.") }
    }
    /// The bundle shipped pre-quantised (`config.json` carries mlx-lm's `quantization` block): its Linears were built as
    /// `QuantizedLinear` before the weights landed, so a quantise-at-load must not be applied on top.
    public var isPreQuantized: Bool { config.quantization != nil }

    /// Loads an mlx-lm-layout bundle (`model.safetensors`, `config.json`), cast to `dtype` (nil keeps bf16). A pre-quantised bundle
    /// (the `-8bit` planner) has its layers' Linears replaced by `QuantizedLinear`s of the config's group size / bits FIRST, so the
    /// key contract then expects `weight` (packed uint32) + `scales` + `biases` per Linear — the layout mlx-lm writes and the
    /// package's quantise-at-load path produces.
    public static func load(_ dir: URL, dtype: DType?) throws -> Qwen3LM {
        let c = try ConfigIO.load(Qwen3Config.self, from: dir.appendingPathComponent("config.json"))
        guard c.tie_word_embeddings else { throw ACEStepError.badConfig("untied lm_head is not handled by this loader") }
        let m = Qwen3LM(c)
        if let q = c.quantization {
            guard q.mode == nil || q.mode == "affine" else { throw ACEStepError.badConfig("quantization mode \(q.mode!) is not handled (affine only)") }
            m.quantizeLayers(groupSize: q.group_size, bits: q.bits)
        }
        let arrays = try WeightIO.load(dir.appendingPathComponent("model.safetensors"), dtype: dtype)
        try WeightIO.apply(arrays, to: m, component: dir.lastPathComponent)
        return m
    }
}
