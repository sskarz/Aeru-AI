//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Available voices
public enum TTSVoice {
  case afHeart
  case bmGeorge
}

// Main class, encapsulates the whole Kokoro text-to-speech pipeline
public class KokoroTTS {
  enum KokoroTTSError: Error {
    case tooManyTokens
  }

  private let bert: CustomAlbert!
  private let bertEncoder: Linear!
  private let durationEncoder: DurationEncoder!
  private let predictorLSTM: LSTM!
  private let durationProj: Linear!
  private let prosodyPredictor: ProsodyPredictor!
  private let textEncoder: TextEncoder!
  private let decoder: Decoder!
  private let eSpeakEngine: ESpeakNGEngine!
  private var chosenVoice: TTSVoice?
  private var voice: MLXArray!

  init() {
    MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    let sanitizedWeights = WeightLoader.loadWeights()
    let config = KokoroConfig.loadConfig()
    
    bert = CustomAlbert(weights: sanitizedWeights,
                        config: AlbertModelArgs(
                          numHiddenLayers: config.plbert.numHiddenLayers,
                          numAttentionHeads: config.plbert.numAttentionHeads,
                          hiddenSize: config.plbert.hiddenSize,
                          intermediateSize: config.plbert.intermediateSize,
                          vocabSize: config.nToken))
    bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
    durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: config.hiddenDim, styDim: config.styleDim, nlayers: config.nLayer)

    predictorLSTM = LSTM(
      inputSize: config.hiddenDim + config.styleDim,
      hiddenSize: config.hiddenDim / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: config.styleDim,
      dHid: config.hiddenDim
    )

    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: config.hiddenDim,
      kernelSize: config.textEncoderKernelSize,
      depth: config.nLayer,
      nSymbols: config.nToken
    )

    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: config.hiddenDim,
      styleDim: config.styleDim,
      dimOut: config.nMels,
      resblockKernelSizes: config.istftNet.resblockKernelSizes,
      upsampleRates: config.istftNet.upsampleRates,
      upsampleInitialChannel: config.istftNet.upsampleInitialChannel,
      resblockDilationSizes: config.istftNet.resblockDilationSizes,
      upsampleKernelSizes: config.istftNet.upsampleKernelSizes,
      genIstftNFft: config.istftNet.genIstftNFFT,
      genIstftHopSize: config.istftNet.genIstftHopSize
    )

    eSpeakEngine = try! ESpeakNGEngine()
  }

  public func generateAudio(voice: TTSVoice, text: String, speed: Float = 1.0) throws -> MLXArray {
    if chosenVoice != voice {
      self.voice = VoiceLoader.loadVoice(voice)
      try eSpeakEngine.setLanguage(for: voice)
      chosenVoice = voice
    }

    BenchmarkTimer.reset()
    BenchmarkTimer.startTimer(Constants.bm_TTS)
    BenchmarkTimer.startTimer(Constants.bm_Phonemize, Constants.bm_TTS)
    let outputStr = try! eSpeakEngine.phonemize(text: text)

    let inputIds = Tokenizer.tokenize(phonemizedText: outputStr)
    guard inputIds.count <= Constants.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    let paddedInputIdsBase = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])

    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    let inputLengthMax: Int = inputLengths.max().item()
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask = textMask + 1 .> inputLengths
    textMask = textMask.expandedDimensions(axes: [0])
    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)
    BenchmarkTimer.stopTimer(Constants.bm_Phonemize, [attentionMask, paddedInputIds])

    BenchmarkTimer.startTimer(Constants.bm_bert, Constants.bm_TTS)
    let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)
    let dEn = bertEncoder(bertDur).transposed(0, 2, 1)
    BenchmarkTimer.stopTimer(Constants.bm_bert, [dEn])

    BenchmarkTimer.startTimer(Constants.bm_duration, Constants.bm_TTS)
    let refS = self.voice[inputIds.count - 1, 0 ... 1, 0...]
    let s = refS[0 ... 1, 128...]
    let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
    let (x, _) = predictorLSTM(d)
    let duration = durationProj(x)
    let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
    let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
    
    let indices = MLX.concatenated(
      predDur.enumerated().map { i, n in
        let nSize: Int = n.item()
        return MLX.repeated(MLXArray([i]), count: nSize)
      }
    )

    var swiftPredAlnTrg = [Float](repeating: 0.0, count: indices.shape[0] * paddedInputIds.shape[1])
    for i in 0 ..< indices.shape[0] {
      let indiceValue: Int = indices[i].item()
      swiftPredAlnTrg[indiceValue * indices.shape[0] + i] = 1.0
    }
    let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([paddedInputIds.shape[1], indices.shape[0]])
    let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
    let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
    BenchmarkTimer.stopTimer(Constants.bm_duration, [en, s])

    BenchmarkTimer.startTimer(Constants.bm_prosody, Constants.bm_TTS)
    let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)
    let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    let asr = MLX.matmul(tEn, predAlnTrg)
    BenchmarkTimer.stopTimer(Constants.bm_prosody, [asr, F0Pred, NPred])

    BenchmarkTimer.startTimer(Constants.bm_decoder, Constants.bm_TTS)
    let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: refS[0 ... 1, 0 ... 127])[0]
    BenchmarkTimer.stopTimer(Constants.bm_decoder, [audio])
    
    BenchmarkTimer.stopTimer(Constants.bm_TTS)
    BenchmarkTimer.print()
    
    return audio
  }

  struct Constants {
    static let maxTokenCount = 510
    static let samplingRate = 24000
    
    static let bm_TTS = "TTSAudio"
    static let bm_Phonemize = "Phonemize"
    static let bm_bert = "BERT"
    static let bm_duration = "Duration"
    static let bm_prosody = "Prosody"
    static let bm_decoder = "Decoder"
  }
}
