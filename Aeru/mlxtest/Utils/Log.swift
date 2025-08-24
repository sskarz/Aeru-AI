//
//  Kokoro-tts-lib
//
import Foundation

#if DEBUG

@inline(__always) nonisolated func logPrint(_ s: String) {
  print(s)
}

#else

@inline(__always) nonisolated func logPrint(_ s: String) {}

#endif
