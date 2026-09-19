import AppKit
import FlashchatKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: "--show") else { return }
        let name = index + 1 < args.count ? args[index + 1] : "Overview"
        let section = MainSection.allCases.first { $0.rawValue.lowercased() == name.lowercased() } ?? .overview
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            MainActor.assumeIsolated { WindowRouter.shared.open(section) }
        }
    }

    /// Launching the app again (Finder, Spotlight) while it runs opens the window.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        MainActor.assumeIsolated { WindowRouter.shared.bringForward() }
        return false
    }
}

@main
struct FlashchatBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = AppModel()
    @State private var router = WindowRouter.shared

    var body: some Scene {
        MenuBarExtra {
            MenuContent()
                .environment(model)
                .environment(router)
        } label: {
            MenuBarLabel(model: model)
        }
        .menuBarExtraStyle(.window)

        Window("Flashchat", id: "main") {
            MainWindow()
                .environment(model)
                .environment(router)
                .frame(minWidth: 820, minHeight: 540)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 960, height: 640)
    }
}

/// The status item. Deliberately static between polls: no animation, and the
/// speed text updates at most once a second (never in quiet mode) so the app
/// stays out of the way of GPU-bound inference and benchmarks.
struct MenuBarLabel: View {
    let model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let display = model.display
        HStack(spacing: 3) {
            Image(systemName: model.quietMode ? (display.isRunning ? "bolt" : "bolt.slash") : display.symbol)
            if !model.quietMode, model.showSpeedInMenuBar, case .generating = display.activity,
               let tps = model.tokensPerSecond {
                Text(String(format: "%.0f", tps)).monospacedDigit()
            }
        }
        .onAppear {
            WindowRouter.shared.openMainWindow = { openWindow(id: "main") }
        }
    }
}

