//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

nonisolated class ReflectionPad1d: Module {
  let padding: IntOrPair

  init(padding: (Int, Int)) {
    self.padding = IntOrPair([padding.0, padding.1])
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    return MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([0, 0]), padding])
  }
}
