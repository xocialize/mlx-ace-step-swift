// ConditionEncoder.swift — modeling_acestep_v15_turbo.py: `AceStepLyricEncoder` (Linear 1024→2048 + 8 bidirectional layers +
// norm), `AceStepTimbreEncoder` (Linear 64→2048 + 4 layers + norm, first position kept, unpacked per batch item; the
// `special_token` parameter is in the checkpoint but unused in the forward, exactly as upstream), `pack_sequences` (kept
// tokens first, stable) and `AceStepConditionEncoder` (text_projector 1024→2048, no bias): lyrics ‖ timbre ‖ text.
import Foundation
import MLX
import MLXFast
import MLXNN

public final class AceStepLyricEncoder: Module {
    let config: ACEStepConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [AceStepEncoderLayer]
    init(_ c: ACEStepConfig) {
        config = c
        _embedTokens.wrappedValue = Linear(c.text_hidden_dim, c.hidden_size, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _layers.wrappedValue = (0 ..< c.num_lyric_encoder_hidden_layers).map { AceStepEncoderLayer(c, index: $0) }
    }
    /// `inputs_embeds` `[B, L, 1024]` + keep mask `[B, L]` → `[B, L, 2048]`.
    func callAsFunction(_ embeds: MLXArray, keep: MLXArray) -> MLXArray {
        norm(EncoderStack.run(layers, embedTokens(embeds), keep: keep, config: config))
    }
}

public final class AceStepTimbreEncoder: Module {
    let config: ACEStepConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_token") var specialToken: MLXArray
    @ModuleInfo(key: "layers") var layers: [AceStepEncoderLayer]
    init(_ c: ACEStepConfig) {
        config = c
        _embedTokens.wrappedValue = Linear(c.timbre_hidden_dim, c.hidden_size, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: Float(c.rms_norm_eps))
        _specialToken.wrappedValue = MLXArray.zeros([1, 1, c.hidden_size])
        _layers.wrappedValue = (0 ..< c.num_timbre_encoder_hidden_layers).map { AceStepEncoderLayer(c, index: $0) }
    }
    /// `refer_audio_acoustic_hidden_states_packed` `[N, T, 64]` with `order` (batch item per reference) →
    /// (`[B, maxRefs, 2048]`, keep `[B, maxRefs]`), `unpack_timbre_embeddings`.
    func callAsFunction(_ packed: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        let h = norm(EncoderStack.run(layers, embedTokens(packed), keep: nil, config: config))
        let first = h[0..., 0, 0...]                                  // [N, D]
        let batch = (order.max() ?? 0) + 1
        var groups = Array(repeating: [Int](), count: batch)
        for (i, b) in order.enumerated() { groups[b].append(i) }
        let width = max(1, groups.map(\.count).max() ?? 1)
        var rows: [MLXArray] = [], keep: [Int32] = []
        for g in groups {
            var row = g.isEmpty ? MLXArray.zeros([0, first.dim(1)], dtype: first.dtype) : first[MLXArray(g.map { Int32($0) })]
            if g.count < width { row = concatenated([row, MLXArray.zeros([width - g.count, first.dim(1)], dtype: first.dtype)], axis: 0) }
            rows.append(row.expandedDimensions(axis: 0))
            keep += Array(repeating: Int32(1), count: g.count) + Array(repeating: Int32(0), count: width - g.count)
        }
        return (concatenated(rows, axis: 0), MLXArray(keep, [batch, width]))
    }
}

public enum SequencePacking {
    /// `pack_sequences(h1, h2, m1, m2)`: concatenate, then a stable sort putting kept tokens first; returns the packed states
    /// `[B, L1+L2, D]` and the new keep mask.
    public static func pack(_ a: MLXArray, _ b: MLXArray, _ keepA: MLXArray, _ keepB: MLXArray) -> (MLXArray, MLXArray) {
        let hidden = concatenated([a, b], axis: 1)
        let keep = concatenated([keepA.asType(.int32), keepB.asType(.int32)], axis: 1)
        let (batch, length) = (keep.dim(0), keep.dim(1))
        let values = keep.asArray(Int32.self)
        var order: [Int32] = [], newKeep: [Int32] = []
        for row in 0 ..< batch {
            let slice = Array(values[(row * length) ..< ((row + 1) * length)])
            let kept = slice.indices.filter { slice[$0] != 0 }, dropped = slice.indices.filter { slice[$0] == 0 }
            order += (kept + dropped).map { Int32($0) }
            newKeep += Array(repeating: Int32(1), count: kept.count) + Array(repeating: Int32(0), count: dropped.count)
        }
        if order == Array(0 ..< Int32(length)).map({ $0 }) && batch == 1 { return (hidden, MLXArray(newKeep, [batch, length])) }
        let index = broadcast(MLXArray(order, [batch, length, 1]), to: [batch, length, hidden.dim(2)])
        return (takeAlong(hidden, index, axis: 1), MLXArray(newKeep, [batch, length]))
    }
}

public final class AceStepConditionEncoder: Module {
    @ModuleInfo(key: "text_projector") var textProjector: Linear
    @ModuleInfo(key: "lyric_encoder") var lyricEncoder: AceStepLyricEncoder
    @ModuleInfo(key: "timbre_encoder") var timbreEncoder: AceStepTimbreEncoder
    init(_ c: ACEStepConfig) {
        _textProjector.wrappedValue = Linear(c.text_hidden_dim, c.hidden_size, bias: false)
        _lyricEncoder.wrappedValue = AceStepLyricEncoder(c)
        _timbreEncoder.wrappedValue = AceStepTimbreEncoder(c)
    }
    /// → (`encoder_hidden_states` `[B, Llyr+Ntimbre+Ltext, 2048]`, `encoder_attention_mask`).
    public func callAsFunction(text: MLXArray, textKeep: MLXArray, lyrics: MLXArray, lyricKeep: MLXArray,
                               references: MLXArray, referenceOrder: [Int]) -> (MLXArray, MLXArray) {
        let projected = textProjector(text)
        let lyric = lyricEncoder(lyrics, keep: lyricKeep)
        let (timbre, timbreKeep) = timbreEncoder(references, order: referenceOrder)
        let (h1, m1) = SequencePacking.pack(lyric, timbre, lyricKeep, timbreKeep)
        return SequencePacking.pack(h1, projected, m1, textKeep)
    }
}
