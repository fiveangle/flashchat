import AppKit
import FlashchatKit
import Observation
import SwiftUI

/// Modal alerts. A menubar app has no key window to hang SwiftUI alerts on, so
/// confirmations use NSAlert after bringing the app forward.
enum Alerts {
    @MainActor
    static func confirm(_ title: String, _ message: String, confirm: String,
                        cancel: String = "Cancel", destructive: Bool = false) -> Bool {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = destructive ? .critical : .informational
        let ok = alert.addButton(withTitle: confirm)
        ok.hasDestructiveAction = destructive
        alert.addButton(withTitle: cancel)
        return alert.runModal() == .alertFirstButtonReturn
    }

    @MainActor
    static func error(_ title: String, _ message: String) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.runModal()
    }

    @MainActor
    static func chooseFolder(_ message: String, start: String? = nil) -> URL? {
        NSApp.activate(ignoringOtherApps: true)
        let panel = NSOpenPanel()
        panel.message = message
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        if let start { panel.directoryURL = URL(fileURLWithPath: (start as NSString).expandingTildeInPath) }
        return panel.runModal() == .OK ? panel.url : nil
    }

    @MainActor
    static func chooseFile(_ message: String) -> URL? {
        NSApp.activate(ignoringOtherApps: true)
        let panel = NSOpenPanel()
        panel.message = message
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        return panel.runModal() == .OK ? panel.url : nil
    }
}

enum MainSection: String, CaseIterable, Identifiable {
    case overview = "Overview"
    case models = "Models"
    case settings = "Settings"
    case logs = "Logs"

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .overview: return "gauge.with.dots.needle.33percent"
        case .models: return "shippingbox"
        case .settings: return "slider.horizontal.3"
        case .logs: return "doc.text.magnifyingglass"
        }
    }
}

/// Navigation shared by the popover and the main window.
@MainActor
@Observable
final class WindowRouter {
    static let shared = WindowRouter()

    var section: MainSection = .overview
    var selectedModel: String?
    var buildRequest: BuildRequest?
    var showingOperation = false
    var openMainWindow: (() -> Void)?
    /// True once a window is open on purpose. Now that the app is a regular
    /// app type, SwiftUI opens the Window scene at launch on its own; in
    /// menu-bar-only mode that window is unwanted and closes itself.
    var windowWanted = AppPresence.wantsWindowAtLaunch
    private var launchWindowSettled = false

    struct BuildRequest: Identifiable, Equatable {
        let id = UUID()
        let model: String
        let variant: String
        let repair: Bool
    }

    func open(_ section: MainSection, model: String? = nil, buildVariant: String? = nil) {
        windowWanted = true
        self.section = section
        if let model { selectedModel = model }
        if let model, let buildVariant {
            buildRequest = BuildRequest(model: model, variant: buildVariant, repair: false)
        }
        bringForward()
    }

    func showOperation() {
        showingOperation = true
        bringForward()
    }

    /// SwiftUI opens the Window scene at launch by itself (the app is a
    /// regular app type) and sizes it from its content, ignoring defaultSize.
    /// Menu-bar-only mode closes that window; otherwise it gets a sane first
    /// size. Both are AppKit calls because the SwiftUI equivalents are
    /// unreliable here.
    func settleLaunchWindow() {
        guard !launchWindowSettled else { return }
        launchWindowSettled = true
        // macOS 14 has no way to suppress SwiftUI's launch window, and the
        // window is not in NSApp.windows immediately, so retry briefly.
        closeUnwantedWindow(attempt: 0)
        DispatchQueue.main.async { [self] in
            guard windowWanted, let window = mainWindow() else { return }
            if UserDefaults.standard.object(forKey: "NSWindow Frame main") == nil,
               let screen = window.screen ?? NSScreen.main {
                let size = NSSize(width: min(1000, screen.visibleFrame.width - 80),
                                  height: min(740, screen.visibleFrame.height - 80))
                window.setContentSize(size)
                window.center()
            }
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    private func closeUnwantedWindow(attempt: Int) {
        guard !windowWanted, attempt < 10 else { return }
        if let window = mainWindow() {
            window.close()
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [self] in
            closeUnwantedWindow(attempt: attempt + 1)
        }
    }

    private func mainWindow() -> NSWindow? {
        NSApp.windows.first { $0.canBecomeMain && $0.contentView != nil && $0.frame.width > 400 }
    }

    func bringForward() {
        windowWanted = true
        if let openMainWindow {
            openMainWindow()
        } else {
            // No view has handed us SwiftUI's window opener yet (menu bar icon
            // hidden, window never shown). The Window scene's own item in the
            // Window menu opens it.
            openViaWindowMenu()
        }
        NSApp.activate(ignoringOtherApps: true)
        // Opening a window makes SwiftUI promote an accessory app to a Dock
        // app; menu-bar-only mode should stay out of the Dock.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            AppPresence.applyActivationPolicy()
        }
    }

    private func openViaWindowMenu() {
        guard let windowMenu = NSApp.mainMenu?.items.first(where: { $0.title == "Window" })?.submenu,
              let item = windowMenu.items.first(where: { $0.title == "Flashchat" }) else { return }
        windowMenu.performActionForItem(at: windowMenu.index(of: item))
    }
}

extension ModelInfo {
    var archiveTitle: String {
        switch archive {
        case "full": return "Full copy in offload storage"
        case "originals": return "Original files in offload storage"
        default: return "No offload copy"
        }
    }
}

struct StatusDot: View {
    let display: ServerDisplay

    var color: Color {
        switch display.activity {
        case .stopped: return .secondary
        case .starting, .stopping, .preparing, .prefill, .generating: return .blue
        case .idle, .external: return .green
        case .restartNeeded, .unreachable: return .orange
        }
    }

    var body: some View {
        Circle().fill(color).frame(width: 9, height: 9)
    }
}

struct Pill: View {
    let text: String
    var color: Color = .secondary

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

/// Dock vs. menu-bar-only presence. Read before the app finishes launching, so
/// it lives in UserDefaults rather than on the model.
enum AppPresence {
    private static let dockKey = "showDockIcon"
    private static let menuBarKey = "showMenuBarIcon"
    private static let launchWindowKey = "openWindowAtLaunch"
    private static let pollKey = "statusPollSeconds"

    static var showDockIcon: Bool {
        get { UserDefaults.standard.object(forKey: dockKey) as? Bool ?? false }
        set { UserDefaults.standard.set(newValue, forKey: dockKey) }
    }

    static var showMenuBarIcon: Bool {
        get { UserDefaults.standard.object(forKey: menuBarKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: menuBarKey) }
    }

    /// Menu-bar-only mode keeps quiet at launch; a Dock app opens its window
    /// unless the user turned that off; with no menu bar icon it must open.
    static var wantsWindowAtLaunch: Bool {
        !showMenuBarIcon || (showDockIcon && openWindowAtLaunch)
    }

    /// Seconds between /health polls while the server is running and idle.
    /// Busy phases poll twice as often so progress stays live.
    static var statusPollSeconds: Double {
        get {
            let stored = UserDefaults.standard.double(forKey: pollKey)
            return stored > 0 ? stored : 5
        }
        set { UserDefaults.standard.set(newValue, forKey: pollKey) }
    }

    static let pollChoices: [Double] = [1, 2, 3, 5, 10, 30]

    static var openWindowAtLaunch: Bool {
        get { UserDefaults.standard.object(forKey: launchWindowKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: launchWindowKey) }
    }

    @MainActor
    static func applyActivationPolicy() {
        let policy: NSApplication.ActivationPolicy = showDockIcon ? .regular : .accessory
        guard NSApp.activationPolicy() != policy else { return }
        let hadVisibleWindow = NSApp.windows.contains { $0.isVisible && $0.canBecomeMain }
        NSApp.setActivationPolicy(policy)
        // Switching to .accessory deactivates the app; keep an open window in front.
        if hadVisibleWindow {
            DispatchQueue.main.async {
                NSApp.activate(ignoringOtherApps: true)
                NSApp.windows.first { $0.canBecomeMain && $0.isVisible }?.makeKeyAndOrderFront(nil)
            }
        }
    }
}



