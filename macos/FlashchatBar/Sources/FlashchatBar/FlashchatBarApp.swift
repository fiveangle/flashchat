import AppKit
import FlashchatKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()
    private var statusItem: StatusItemController?

    /// Before any window or Dock tile appears, so menu-bar-only mode never
    /// flashes a Dock icon.
    func applicationWillFinishLaunching(_ notification: Notification) {
        MainActor.assumeIsolated { AppPresence.applyActivationPolicy() }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = StatusItemController(model: model, router: .shared)
        // A window at launch is the Dock-app convention, and with no menu bar
        // icon it is the only way in. macOS gives no reliable way to tell a
        // login-item launch from a user launch, so this is a preference
        // instead of a guess.
        if AppPresence.wantsWindowAtLaunch {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                MainActor.assumeIsolated { WindowRouter.shared.bringForward() }
            }
        }
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: "--show") else { return }
        let name = index + 1 < args.count ? args[index + 1] : "Overview"
        let section = MainSection.allCases.first { $0.rawValue.lowercased() == name.lowercased() } ?? .overview
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            MainActor.assumeIsolated { WindowRouter.shared.open(section) }
        }
    }

    /// Flashchat keeps running with no window: the menu bar icon (or the Dock
    /// icon) is the way back in, and the server does not depend on the window.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
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
    @State private var router = WindowRouter.shared

    private var model: AppModel { delegate.model }

    // The menu bar icon and popover are AppKit (StatusItemController), made
    // by the app delegate at launch.
    var body: some Scene {
        // SwiftUI opens this window at launch on its own; WindowRouter closes
        // it again when menu-bar-only mode did not want one.
        Window("Flashchat", id: "main") {
            MainWindow()
                .environment(model)
                .environment(router)
                // Min height keeps the model actions above the fold; SwiftUI
                // sizes this window from its content and ignores defaultSize.
                .frame(minWidth: 860, idealWidth: 1000, minHeight: 700, idealHeight: 720)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1000, height: 740)
        .commands { AppCommands(model: model, router: router) }
    }
}

/// Main menu for Dock mode (menu-bar-only mode never shows it). SwiftUI keeps
/// its standard Edit menu, so cut/copy/paste work in text fields.
struct AppCommands: Commands {
    let model: AppModel
    let router: WindowRouter

    var body: some Commands {
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { router.open(.settings) }
                .keyboardShortcut(",", modifiers: .command)
        }
        CommandGroup(replacing: .newItem) {}
        CommandMenu("Server") {
            let display = model.display
            if display.isRunning {
                Button("Stop Server") { Task { await model.stopServer() } }
                    .disabled(model.transition != nil)
                Button("Restart Server") { Task { await model.restartServer() } }
                    .disabled(model.transition != nil || !display.isOwned)
            } else {
                Button("Start Server") { Task { await model.startServer() } }
                    .disabled(model.transition != nil || model.apiState?.selected == nil)
            }
            Divider()
            Button("Copy API URL") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(model.apiURL, forType: .string)
            }
            Toggle("Quiet Mode (for Benchmarks)", isOn: Binding(
                get: { model.quietMode }, set: { model.setQuietMode($0) }))
        }
        CommandGroup(before: .sidebar) {
            ForEach(Array(MainSection.allCases.enumerated()), id: \.element) { index, section in
                Button(section.rawValue) { router.open(section) }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            }
            Divider()
        }
    }
}
