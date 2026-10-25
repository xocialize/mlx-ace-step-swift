// ACEStepConfig.swift — the checkpoints' config.json files, decoded as the converter wrote them (upstream's values are the
// oracle truths: acestep-v15-turbo/config.json, the two Qwen3 configs, vae/config.json).
import Foundation

/// `AceStepConfig` (acestep/models/common/configuration_acestep_v15.py) as the turbo checkpoint resolves it.
public struct ACEStepConfig: Codable, Sendable {
    public var hidden_size: Int
    public var intermediate_size: Int
    public var num_hidden_layers: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var rms_norm_eps: Double
    public var rope_theta: Double
    public var max_position_embeddings: Int
    public var attention_bias: Bool
    public var layer_types: [String]
    public var sliding_window: Int
    public var use_sliding_window: Bool
    public var in_channels: Int
    public var audio_acoustic_hidden_dim: Int
    public var patch_size: Int
    public var text_hidden_dim: Int
    public var timbre_hidden_dim: Int
    public var timbre_fix_frame: Int
    public var pool_window_size: Int
    public var fsq_dim: Int
    public var fsq_input_levels: [Int]
    public var fsq_input_num_quantizers: Int
    public var num_lyric_encoder_hidden_layers: Int
    public var num_timbre_encoder_hidden_layers: Int
    public var num_attention_pooler_hidden_layers: Int
    public var num_audio_decoder_hidden_layers: Int
    public var is_turbo: Bool
    public var model_version: String?

    public func isSliding(layer: Int) -> Bool { use_sliding_window && layer_types[layer] == "sliding_attention" }
}

/// Qwen3 config (the planner and Qwen3-Embedding-0.6B; mlx-lm's ModelArgs fields).
public struct Qwen3Config: Codable, Sendable {
    public var hidden_size: Int
    public var num_hidden_layers: Int
    public var intermediate_size: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var rms_norm_eps: Double
    public var vocab_size: Int
    public var rope_theta: Double
    public var max_position_embeddings: Int
    public var tie_word_embeddings: Bool
}

/// diffusers `AutoencoderOobleck` config.
public struct OobleckConfig: Codable, Sendable {
    public var audio_channels: Int
    public var channel_multiples: [Int]
    public var decoder_channels: Int
    public var decoder_input_channels: Int
    public var downsampling_ratios: [Int]
    public var encoder_hidden_size: Int
    public var sampling_rate: Int
    public var hop: Int { downsampling_ratios.reduce(1, *) }
}
