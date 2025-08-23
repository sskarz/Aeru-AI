import AVFoundation
import MLX
import SwiftUI

class MLXTestModel: ObservableObject {
  let kokoroTTSEngine: KokoroTTS!
  let audioEngine: AVAudioEngine!
  let playerNode: AVAudioPlayerNode!

  init() {
    kokoroTTSEngine = KokoroTTS()
    audioEngine = AVAudioEngine()
    playerNode = AVAudioPlayerNode()
    audioEngine.attach(playerNode)

    #if os(iOS)
      do {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .default)
        try audioSession.setActive(true)
      } catch {
        logPrint("Failed to set up AVAudioSession: \(error.localizedDescription)")
        // Handle the error appropriately
      }
    #endif
  }

  func say(_ text: String) {
    let audioBuffer = try! kokoroTTSEngine.generateAudio(voice: .afHeart, text: text)
    let audio = audioBuffer[0].asArray(Float.self)

    let sampleRate = Double(KokoroTTS.Constants.samplingRate)
    let audioLength = Double(audio.count) / sampleRate
    logPrint("Audio Length: " + String(format: "%.4f", audioLength))
    logPrint("Real Time Factor: " + String(format: "%.2f", audioLength / (BenchmarkTimer.getTimeInSec(KokoroTTS.Constants.bm_TTS) ?? 1.0)))

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
      logPrint("Couldn't create buffer")
      return
    }

    buffer.frameLength = buffer.frameCapacity
    let channels = buffer.floatChannelData!
    for i in 0 ..< audio.count {
      channels[0][i] = audio[i]
    }

    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
    do {
      try audioEngine.start()
    } catch {
      logPrint("Audio engine failed to start: \(error.localizedDescription)")
      return
    }

    playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts, completionHandler: nil)
    playerNode.play()
  }
}
