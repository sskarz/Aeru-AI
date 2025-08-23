import SwiftUI

@main
struct mlxtestApp: App {
  let model = MLXTestModel()

  var body: some Scene {
    WindowGroup {
      ContentView(viewModel: model)
    }
  }
}
