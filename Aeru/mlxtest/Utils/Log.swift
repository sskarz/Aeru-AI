//
//  Kokoro-tts-lib
//
import Foundation

#if DEBUG

@inline(__always) func logPrint(_ s: String) {
  print(s)
}

#else

@inline(__always) func logPrint(_ s: String) {}

#endif
