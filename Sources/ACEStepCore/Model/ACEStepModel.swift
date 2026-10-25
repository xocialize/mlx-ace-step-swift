// ACEStepModel.swift — `AceStepConditionGenerationModel` as one module tree with the checkpoint's names: decoder (the DiT),
// encoder (conditions), tokenizer (FSQ), detokenizer, null_condition_emb (CFG only on the base models; kept for the key
// contract). Loaded from the V1 bundle (Tools/convert/convert.py), cast to the requested dtype, with the silence latent beside.
import Foundation
import MLX
import MLXNN

public final class ACEStepModel: Module {
    public let config: ACEStepConfig
    @ModuleInfo(key: "decoder") public var decoder: AceStepDiTModel
    @ModuleInfo(key: "encoder") public var encoder: AceStepConditionEncoder
    @ModuleInfo(key: "tokenizer") public var tokenizer: AceStepAudioTokenizer
    @ModuleInfo(key: "detokenizer") public var detokenizer: AudioTokenDetokenizer
    @ParameterInfo(key: "null_condition_emb") public var nullConditionEmb: MLXArray

    public init(_ c: ACEStepConfig) {
        config = c
        _decoder.wrappedValue = AceStepDiTModel(c)
        _encoder.wrappedValue = AceStepConditionEncoder(c)
        _tokenizer.wrappedValue = AceStepAudioTokenizer(c)
        _detokenizer.wrappedValue = AudioTokenDetokenizer(c)
        _nullConditionEmb.wrappedValue = MLXArray.zeros([1, 1, c.hidden_size])
    }
    public var dtype: DType { decoder.conditionEmbedder.weight.dtype }

    /// `_decode_audio_codes_to_latents`: codes `[T5]` → 25 Hz hints `[1, T5·5, 64]`.
    public func hints(codes: [Int]) -> MLXArray {
        let ids = MLXArray(codes.map { Int32($0) }).reshaped(1, -1)
        return detokenizer(tokenizer.quantizer.output(fromIndices: ids, dtype: dtype))
    }

    /// `quantizeLayers`: the DiT's Linears (V2b tiers); encoders, FSQ and the detokenizer stay as loaded.
    public func quantizeDiT(groupSize: Int, bits: Int) {
        quantize(model: self, groupSize: groupSize, bits: bits) { path, m in m is Linear && path.hasPrefix("decoder.layers.") }
    }

    /// Loads `ACE-Step-1.5-turbo-<dtype>/` (model.safetensors + config.json + silence_latent.safetensors).
    public static func load(_ dir: URL, dtype: DType?) throws -> (ACEStepModel, MLXArray) {
        let c = try ConfigIO.load(ACEStepConfig.self, from: dir.appendingPathComponent("config.json"))
        let m = ACEStepModel(c)
        let arrays = try WeightIO.load(dir.appendingPathComponent("model.safetensors"), dtype: dtype)
        try WeightIO.apply(arrays, to: m, component: dir.lastPathComponent)
        let sil = try WeightIO.load(dir.appendingPathComponent("silence_latent.safetensors"), dtype: nil)
        guard let silence = sil["silence_latent"] else { throw ACEStepError.badConfig("silence_latent.safetensors has no 'silence_latent'") }
        return (m, silence)       // [15000, 64] fp32
    }
}
