// Oobleck.swift — diffusers `AutoencoderOobleck` as acestep/models/mlx/vae_model.py writes it (channels-last, weight norm
// folded at conversion, Snake with log-scale alpha/beta): 48 kHz stereo ↔ 64-d latents at 25 Hz (hop 1920). Decoder path
// gated here; the encoder is built and loaded for the key contract and the V3 inputs (reference / cover audio).
// `decodeChunked` = the handler's `_mlx_decode_single` (512-frame windows, 64 frames of overlap discarded, cache cleared
// every 4 windows) — the memory knob V0 measured (18 GB working set at 512 vs 34 GB at 2048).
import Foundation
import MLX
import MLXNN

/// `Snake1d`: `x + 1/(exp(β)+1e-9) · sin(exp(α)·x)²`, parameters `[C]`.
public final class Snake1d: Module, UnaryLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    init(_ channels: Int) { _alpha.wrappedValue = MLXArray.zeros([channels]); _beta.wrappedValue = MLXArray.zeros([channels]) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha), b = exp(beta)
        return x + reciprocal(b + 1e-9) * pow(sin(a * x), 2)
    }
}

public final class OobleckResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "snake1") var snake1: Snake1d
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "snake2") var snake2: Snake1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    init(_ dim: Int, dilation: Int) {
        _snake1.wrappedValue = Snake1d(dim)
        _conv1.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: ((7 - 1) * dilation) / 2, dilation: dilation)
        _snake2.wrappedValue = Snake1d(dim)
        _conv2.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv2(snake2(conv1(snake1(x))))
        let pad = (x.dim(1) - y.dim(1)) / 2
        return (pad > 0 ? x[0..., pad ..< (x.dim(1) - pad), 0...] : x) + y
    }
}

public final class OobleckEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "res_unit1") var resUnit1: OobleckResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: OobleckResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: OobleckResidualUnit
    @ModuleInfo(key: "snake1") var snake1: Snake1d
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    init(input: Int, output: Int, stride: Int) {
        _resUnit1.wrappedValue = OobleckResidualUnit(input, dilation: 1)
        _resUnit2.wrappedValue = OobleckResidualUnit(input, dilation: 3)
        _resUnit3.wrappedValue = OobleckResidualUnit(input, dilation: 9)
        _snake1.wrappedValue = Snake1d(input)
        _conv1.wrappedValue = Conv1d(inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride, padding: (stride + 1) / 2)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { conv1(snake1(resUnit3(resUnit2(resUnit1(x))))) }
}

public final class OobleckDecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "snake1") var snake1: Snake1d
    @ModuleInfo(key: "conv_t1") var convT1: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") var resUnit1: OobleckResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: OobleckResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: OobleckResidualUnit
    init(input: Int, output: Int, stride: Int) {
        _snake1.wrappedValue = Snake1d(input)
        _convT1.wrappedValue = ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride, padding: (stride + 1) / 2)
        _resUnit1.wrappedValue = OobleckResidualUnit(output, dilation: 1)
        _resUnit2.wrappedValue = OobleckResidualUnit(output, dilation: 3)
        _resUnit3.wrappedValue = OobleckResidualUnit(output, dilation: 9)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { resUnit3(resUnit2(resUnit1(convT1(snake1(x))))) }
}

public final class OobleckEncoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "block") var block: [OobleckEncoderBlock]
    @ModuleInfo(key: "snake1") var snake1: Snake1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    init(_ c: OobleckConfig) {
        let cm = [1] + c.channel_multiples
        _conv1.wrappedValue = Conv1d(inputChannels: c.audio_channels, outputChannels: c.encoder_hidden_size, kernelSize: 7, padding: 3)
        _block.wrappedValue = c.downsampling_ratios.enumerated().map { i, s in
            OobleckEncoderBlock(input: c.encoder_hidden_size * cm[i], output: c.encoder_hidden_size * cm[i + 1], stride: s)
        }
        let d = c.encoder_hidden_size * cm.last!
        _snake1.wrappedValue = Snake1d(d)
        _conv2.wrappedValue = Conv1d(inputChannels: d, outputChannels: c.encoder_hidden_size, kernelSize: 3, padding: 1)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block { h = b(h) }
        return conv2(snake1(h))
    }
}

public final class OobleckDecoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "block") var block: [OobleckDecoderBlock]
    @ModuleInfo(key: "snake1") var snake1: Snake1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    init(_ c: OobleckConfig) {
        let cm = [1] + c.channel_multiples
        let strides = Array(c.downsampling_ratios.reversed())
        _conv1.wrappedValue = Conv1d(inputChannels: c.decoder_input_channels, outputChannels: c.decoder_channels * cm.last!, kernelSize: 7, padding: 3)
        _block.wrappedValue = strides.enumerated().map { i, s in
            OobleckDecoderBlock(input: c.decoder_channels * cm[strides.count - i], output: c.decoder_channels * cm[strides.count - i - 1], stride: s)
        }
        _snake1.wrappedValue = Snake1d(c.decoder_channels)
        _conv2.wrappedValue = Conv1d(inputChannels: c.decoder_channels, outputChannels: c.audio_channels, kernelSize: 7, padding: 3, bias: false)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block {
            h = b(h)
            // Bound the live set: evaluating between blocks frees a block's Snake / conv temporaries before the next, wider
            // block allocates its own (the final 128-channel stage on a 640-frame window is 0.6 GB per temporary). Numerically
            // a no-op; the V3 memory trace is the receipt.
            eval(h)
        }
        return conv2(snake1(h))
    }
}

public final class AutoencoderOobleck: Module {
    public let config: OobleckConfig
    @ModuleInfo(key: "encoder") public var encoder: OobleckEncoder
    @ModuleInfo(key: "decoder") public var decoder: OobleckDecoder
    public init(_ c: OobleckConfig) { config = c; _encoder.wrappedValue = OobleckEncoder(c); _decoder.wrappedValue = OobleckDecoder(c) }
    public var dtype: DType { decoder.conv1.weight.dtype }

    /// `encode_mean`: audio `[B, S, 2]` → the posterior mean `[B, T, 64]`.
    public func encodeMean(_ audio: MLXArray) -> MLXArray { split(encoder(audio), parts: 2, axis: -1)[0] }

    /// Full decode: latents `[B, T, 64]` → audio `[B, 1920·T, 2]`.
    public func decode(_ latents: MLXArray) -> MLXArray { decoder(latents.asType(dtype)) }

    /// `_mlx_decode_single`: windows of `chunk` (≥ 192) frames with `overlap` frames each side discarded; cache cleared every 4.
    /// `onWindow(done, total)` runs after every window; whatever it throws (a cancellation) is rethrown unchanged.
    public func decodeChunked(_ latents: MLXArray, chunk: Int = 512, overlap: Int = 64, onWindow: ((Int, Int) throws -> Void)? = nil) throws -> MLXArray {
        let z = latents.asType(dtype)
        let frames = z.dim(1)
        let mlxChunk = max(192, chunk)
        if frames <= mlxChunk { let y = decode(z); eval(y); try onWindow?(1, 1); return y }
        let stride = mlxChunk - 2 * overlap
        let steps = (frames + stride - 1) / stride
        var parts: [MLXArray] = []
        var factor: Double? = nil
        for i in 0 ..< steps {
            let coreStart = i * stride, coreEnd = min(coreStart + stride, frames)
            let winStart = max(0, coreStart - overlap), winEnd = min(frames, coreEnd + overlap)
            let piece = decode(z[0..., winStart ..< winEnd, 0...])
            eval(piece)
            let f = factor ?? Double(piece.dim(1)) / Double(winEnd - winStart)
            factor = f
            let trimStart = Int((Double(coreStart - winStart) * f).rounded(.toNearestOrEven))
            let trimEnd = Int((Double(winEnd - coreEnd) * f).rounded(.toNearestOrEven))
            let endIdx = trimEnd > 0 ? piece.dim(1) - trimEnd : piece.dim(1)
            let trimmed = piece[0..., trimStart ..< endIdx, 0...]
            eval(trimmed)
            parts.append(trimmed)
            if (i + 1) % 4 == 0 { Memory.clearCache() }
            try onWindow?(i + 1, steps)
        }
        return concatenated(parts, axis: 1)
    }

    public static func load(_ dir: URL, dtype: DType?) throws -> AutoencoderOobleck {
        let c = try ConfigIO.load(OobleckConfig.self, from: dir.appendingPathComponent("config.json"))
        let m = AutoencoderOobleck(c)
        let arrays = try WeightIO.load(dir.appendingPathComponent("model.safetensors"), dtype: dtype)
        try WeightIO.apply(arrays, to: m, component: dir.lastPathComponent)
        return m
    }
}
