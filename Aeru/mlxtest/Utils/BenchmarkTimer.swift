//
//  Kokoro-tts-lib
//
import Foundation
import MLX

class BenchmarkTimer {
  private class Timing {
    let id: String

    private var start: DispatchTime
    private var finish: DispatchTime?
    private var childTasks: [Timing] = []
    internal let parent: Timing?
    private var delta: UInt64 = 0

    init(id: String, parent: Timing?) {
      start = DispatchTime.now()
      self.id = id
      self.parent = parent
      if let parent { parent.childTasks.append(self) }
    }

    func startTimer() {
      start = DispatchTime.now()
    }

    func stop() {
      finish = DispatchTime.now()
      delta += finish!.uptimeNanoseconds - start.uptimeNanoseconds
    }

    func log(spaces: Int = 0) {
      guard let _ = finish else { return }

      let spaceString = String(repeating: " ", count: spaces)
      logPrint(spaceString + id + ": " + deltaInSec + " sec")
      for childTask in childTasks {
        childTask.log(spaces: spaces + 2)
      }
    }

    var deltaTime: Double { Double(delta) / 1_000_000_000 }
    var deltaInSec: String { String(format: "%.4f", Double(delta) / 1_000_000_000) }
  }

  static let shared = BenchmarkTimer()

  private init() {}

  private var timers: [String: Timing] = [:]

  @discardableResult
  private func create(id: String, parent parentId: String? = nil) -> Timing? {
    guard timers[id] == nil else { return timers[id] }

    var parentTiming: Timing?
    if let parentId {
      parentTiming = timers[parentId]
      guard parentTiming != nil else { return nil }
    }

    timers[id] = Timing(id: id, parent: parentTiming)
    return timers[id]
  }

  private func stop(id: String) {
    guard let timing = timers[id] else { return }
    timing.stop()
  }

  private func printLogs(id: String) {
    guard let timing = timers[id] else { return }
    timing.log()
  }

  private func reset() {
    timers = [:]
  }
  
  private func exists(id: String) -> Timing? {
    return timers[id]
  }
  
  #if DEBUG
  
  @inline(__always) static func startTimer(_ id: String, _ parent: String? = nil) {
    if let timer = BenchmarkTimer.shared.create(id: id, parent: parent) {
      timer.startTimer()
    }
  }
  
  @inline(__always) static func stopTimer(_ id: String, _ arrays: [MLXArray] = []) {
    arrays.forEach { $0.eval() }
    
    if let timer = BenchmarkTimer.shared.exists(id: id) {
      timer.stop()
    }
  }
  
  static func print() {
    for (key, timing) in BenchmarkTimer.shared.timers {
      if timing.parent == nil {
        BenchmarkTimer.shared.printLogs(id: key)
      }
    }
  }
  
  static func reset() {
    BenchmarkTimer.shared.reset()
  }
  
  @inline(__always) static func getTimeInSec(_ id: String) -> Double? {
    BenchmarkTimer.shared.timers[id]?.deltaTime
  }
  
  #else
  
  @inline(__always) static func startTimer(_ id: String, _ parent: String? = nil) {}
  @inline(__always) static func stopTimer(_ id: String, _ arrays: [MLXArray] = []) {}
  @inline(__always) static func print() {}
  @inline(__always) static func reset() {}
  @inline(__always) static func getTimeInSec(_ id: String) -> Double? { 0.0 }
  
  #endif
}
