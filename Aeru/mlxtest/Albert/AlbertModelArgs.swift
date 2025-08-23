//
//  Kokoro-tts-lib
//
import Foundation

struct AlbertModelArgs {
  let numHiddenLayers: Int
  let numAttentionHeads: Int
  let hiddenSize: Int
  let intermediateSize: Int
  let embeddingSize: Int
  let innerGroupNum: Int
  let numHiddenGroups: Int
  let layerNormEps: Float
  let vocabSize: Int

  init(
    numHiddenLayers: Int,
    numAttentionHeads: Int,
    hiddenSize: Int,
    intermediateSize: Int,
    vocabSize: Int,
    embeddingSize: Int = 128,
    innerGroupNum: Int = 1,
    numHiddenGroups: Int = 1,
    layerNormEps: Float = 1e-12
  ) {
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.hiddenSize = hiddenSize
    self.intermediateSize = intermediateSize
    self.vocabSize = vocabSize
    self.embeddingSize = embeddingSize
    self.innerGroupNum = innerGroupNum
    self.numHiddenGroups = numHiddenGroups
    self.layerNormEps = layerNormEps
  }
}
