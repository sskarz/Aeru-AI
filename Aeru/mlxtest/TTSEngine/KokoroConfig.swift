//
//  Kokoro-tts-lib
//
import Foundation

struct KokoroConfig : Decodable {
  static var config: KokoroConfig?

  struct IstftnetConfig : Decodable {
    let upsampleKernelSizes: [Int]
    let upsampleRates: [Int]
    let genIstftHopSize: Int
    let genIstftNFFT: Int
    let resblockDilationSizes: [[Int]]
    let resblockKernelSizes: [Int]
    let upsampleInitialChannel: Int
    
    enum CodingKeys : String, CodingKey {
      case upsampleKernelSizes = "upsample_kernel_sizes"
      case upsampleRates = "upsample_rates"
      case genIstftHopSize = "gen_istft_hop_size"
      case genIstftNFFT = "gen_istft_n_fft"
      case resblockDilationSizes = "resblock_dilation_sizes"
      case resblockKernelSizes = "resblock_kernel_sizes"
      case upsampleInitialChannel = "upsample_initial_channel"
    }
  }
  
  struct Plbert: Decodable {
    let hiddenSize: Int
    let numAttentionHeads: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let numHiddenLayers: Int
    let dropout: Double
    
    enum CodingKeys : String, CodingKey {
      case hiddenSize = "hidden_size"
      case numAttentionHeads = "num_attention_heads"
      case intermediateSize = "intermediate_size"
      case maxPositionEmbeddings = "max_position_embeddings"
      case numHiddenLayers = "num_hidden_layers"
      case dropout = "dropout"
    }
  }
  
  let istftNet: IstftnetConfig
  let dimIn: Int
  let dropout: Double
  let hiddenDim: Int
  let maxConvDim: Int
  let maxDur: Int
  let multispeaker: Bool
  let nLayer: Int
  let nMels: Int
  let nToken: Int
  let styleDim: Int
  let textEncoderKernelSize: Int
  let plbert: Plbert
  let vocab: [String: Int]
  
  enum CodingKeys : String, CodingKey {
    case istftNet = "istftnet"
    case dimIn = "dim_in"
    case dropout = "dropout"
    case hiddenDim = "hidden_dim"
    case maxConvDim = "max_conv_dim"
    case maxDur = "max_dur"
    case multispeaker = "multispeaker"
    case nLayer = "n_layer"
    case nMels = "n_mels"
    case nToken = "n_token"
    case styleDim = "style_dim"
    case textEncoderKernelSize = "text_encoder_kernel_size"
    case plbert = "plbert"
    case vocab = "vocab"
  }
  
  static func loadConfig() -> KokoroConfig {
    let filePath = Bundle.main.path(forResource: "config", ofType: "json")!
    let configJSON = try! String.init(contentsOf: URL(filePath: filePath), encoding: .utf8)    
    KokoroConfig.config = try! JSONDecoder().decode(KokoroConfig.self, from: configJSON.data(using: .utf8)!)
    return KokoroConfig.config!
  }
}
