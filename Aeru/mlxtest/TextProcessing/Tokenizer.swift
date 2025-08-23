//
//  Kokoro-tts-lib
//
import Foundation

// Utility class for tokenizing the phonemized text
final class Tokenizer {
  private init() {}

  // Use ESpeakNGEngine to phonemize the text first before calling this method
  // Returns tokenized array that can then be passed to TTS system
  static func tokenize(phonemizedText text: String) -> [Int] {
    guard let vocab = KokoroConfig.config?.vocab else { return [] }
    return text
      .map { vocab[String($0)] }
      .filter { $0 != nil }
      .map { $0! }
  }
}
