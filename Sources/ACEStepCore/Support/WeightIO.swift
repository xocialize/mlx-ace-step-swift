// WeightIO.swift — safetensors in, key contract enforced (the mlx-yue2-swift / mlx-auk-swift idiom): load on the CPU stream,
// cast floating tensors to the requested dtype, compare the module's flattened parameter paths with the on-disk keys —
// 0 missing, 0 unused, or throw — then `update(verify:)`. A partial load is the silent-failure class.
//
// Bundle layout = the V1 converter's (Tools/convert/convert.py): DiT checkpoint keys verbatim but for proj_in / proj_out;
// planner and text encoder in mlx-lm's `model.*` layout; VAE with weight-norm folded and convs in MLX (O, K, I).

import Foundation
import MLX
import MLXNN

public enum ACEStepError: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case badConfig(String)
    case invalidInput(String)
    case generation(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file: \(p)"
        case .keyContract(let c, let m, let u):
            return "\(c): key contract violated — missing \(m.count) \(m.prefix(6)), unused \(u.count) \(u.prefix(6))"
        case .badConfig(let s): return "bad config: \(s)"
        case .invalidInput(let s): return "invalid input: \(s)"
        case .generation(let s): return "generation failed: \(s)"
        }
    }
}

public enum WeightIO {
    /// Loads every array of a safetensors file on the CPU stream; floating tensors are cast to `dtype` unless `keepFP32(key)`.
    public static func load(_ url: URL, dtype: DType?, keepFP32: (String) -> Bool = { _ in false }) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw ACEStepError.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                if floating, let d = dtype, !keepFP32(k), v.dtype != d { out[k] = v.asType(d) } else { out[k] = v }
            }
            eval(Array(out.values))
            return out
        }
    }

    /// `rename` maps an on-disk key to the module's parameter path (identity for every ACE-Step bundle).
    public static func apply(_ arrays: [String: MLXArray], to module: Module, component: String,
                             rename: (String) -> String = { $0 }) throws {
        var arrays = Dictionary(uniqueKeysWithValues: arrays.map { (rename($0.key), $0.value) })
        let expected = Set(module.parameters().flattened().map(\.0))
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw ACEStepError.keyContract(component: component, missing: missing, unused: unused)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
        arrays.removeAll()
    }
}

public enum ConfigIO {
    public static func load<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else { throw ACEStepError.missingFile(url.path) }
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }
    public static func json(_ url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw ACEStepError.missingFile(url.path) }
        guard let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw ACEStepError.badConfig("\(url.lastPathComponent) is not a JSON object")
        }
        return o
    }
}
