import AppKit
import Observation
import SwiftUI

/// The menu bar icon and its popover, in AppKit. SwiftUI's MenuBarExtra cannot
/// be closed from code without falling out of step with its own open/closed
/// state, which cost an extra click to reopen it and sent the app's windows
/// behind other apps.
@MainActor
final class StatusItemController: NSObject, NSWindowDelegate {
    private let model: AppModel
    private let router: WindowRouter
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let panel = PopoverPanel()
    private let menuPanel = MenuPanel()
    private var outsideClickMonitor: Any?
    /// While Flashchat takes focus back, the popover keeps it too.
    private var reclaimingFocusUntil: Date?

    init(model: AppModel, router: WindowRouter) {
        self.model = model
        self.router = router
        super.init()
        statusItem.autosaveName = "Flashchat"
        if let button = statusItem.button {
            button.target = self
            button.action = #selector(toggle)
            button.sendAction(on: [.leftMouseDown, .rightMouseDown])
            button.imagePosition = .imageOnly
        }
        panel.delegate = self
        panel.onCancel = { [weak self] in self?.hide() }
        menuPanel.window = panel
        menuPanel.hide = { [weak self] in self?.hide() }
        trackLabel()
        let center = NotificationCenter.default
        center.addObserver(forName: NSApplication.didResignActiveNotification, object: nil, queue: .main) {
            [weak self] _ in
            MainActor.assumeIsolated { self?.reclaimFocusLostToIconPress() }
        }
        center.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) {
            [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.panel.isVisible else { return }
                self.panel.makeKey()
            }
        }
    }

    /// In menu-bar-only mode Flashchat does not own the menu bar, so pressing
    /// its icon hands focus to the app that does, sending an open Flashchat
    /// window behind that app before the click even reaches us. Take focus
    /// back as soon as it goes, while the button is still down over the icon.
    /// With the Dock icon shown Flashchat owns the menu bar and never gets here.
    private func reclaimFocusLostToIconPress() {
        guard NSEvent.pressedMouseButtons != 0,
              let iconFrame = statusItem.button?.window?.frame,
              iconFrame.contains(NSEvent.mouseLocation),
              NSApp.windows.contains(where: { $0.isVisible && $0.canBecomeMain }) else { return }
        reclaimingFocusUntil = Date().addingTimeInterval(0.5)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Redraws the icon whenever a value it shows changes, and only then.
    private func trackLabel() {
        withObservationTracking {
            updateLabel()
        } onChange: { [weak self] in
            Task { @MainActor in self?.trackLabel() }
        }
    }

    private func updateLabel() {
        statusItem.isVisible = model.showMenuBarIcon
        if !model.showMenuBarIcon { hide() }
        let display = model.display
        let symbol = model.quietMode ? (display.isRunning ? "bolt" : "bolt.slash") : display.symbol
        var speed: String?
        if !model.quietMode, model.showSpeedInMenuBar, case .generating = display.activity,
           let tps = model.tokensPerSecond {
            speed = String(format: tps < 100 ? "%.1f" : "%.0f", tps)
        }
        guard let button = statusItem.button else { return }
        button.image = StatusItemImage.image(symbol: symbol, speed: speed)
        button.setAccessibilityLabel(speed.map { "Flashchat, \($0) tokens per second" } ?? "Flashchat")
    }

    @objc private func toggle() {
        panel.isVisible ? hide() : show()
    }

    /// A fresh SwiftUI view per opening, so its appear/disappear hooks (status
    /// refresh, live server memory) run each time, as they did in MenuBarExtra.
    private func show() {
        guard let button = statusItem.button, let buttonWindow = button.window else { return }
        let content = NSHostingView(rootView: MenuContent()
            .environment(model)
            .environment(router)
            .environment(\.menuPanel, menuPanel))
        panel.setContent(content)
        let size = content.fittingSize
        let anchor = buttonWindow.convertToScreen(button.convert(button.bounds, to: nil))
        let visible = (buttonWindow.screen ?? NSScreen.main)?.visibleFrame ?? anchor
        let x = min(max(anchor.minX, visible.minX + 8), visible.maxX - size.width - 8)
        panel.setFrame(NSRect(x: x, y: anchor.minY - size.height - 3, width: size.width, height: size.height),
                       display: true)
        panel.orderFrontRegardless()
        panel.makeKey()
        button.highlight(true)
        outsideClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) {
            [weak self] _ in
            Task { @MainActor in self?.hide() }
        }
    }

    private func hide() {
        guard panel.isVisible else { return }
        panel.orderOut(nil)
        // Often called from inside one of the popover's own button actions, so
        // the SwiftUI view is torn down after that action returns.
        DispatchQueue.main.async { [panel] in
            if !panel.isVisible { panel.setContent(nil) }
        }
        statusItem.button?.highlight(false)
        if let outsideClickMonitor { NSEvent.removeMonitor(outsideClickMonitor) }
        outsideClickMonitor = nil
    }

    /// Another window of this app (the main window, an alert) took focus.
    func windowDidResignKey(_ notification: Notification) {
        if let reclaimingFocusUntil, Date() < reclaimingFocusUntil { return }
        hide()
    }
}

/// A borderless, rounded panel that takes keyboard focus without activating
/// the app, so opening it leaves the app's other windows where they are.
final class PopoverPanel: NSPanel {
    var onCancel: () -> Void = {}
    private let background = NSVisualEffectView()

    init() {
        super.init(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel],
                   backing: .buffered, defer: true)
        isReleasedWhenClosed = false
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        level = .popUpMenu
        collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
        background.material = .popover
        background.state = .active
        background.wantsLayer = true
        background.layer?.cornerRadius = 10
        background.layer?.masksToBounds = true
        contentView = background
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    override func cancelOperation(_ sender: Any?) {
        onCancel()
    }

    func setContent(_ view: NSView?) {
        background.subviews.forEach { $0.removeFromSuperview() }
        guard let view else { return }
        view.frame = background.bounds
        view.autoresizingMask = [.width, .height]
        background.addSubview(view)
    }
}

/// The popover's window. Actions that open a window or another app dismiss the
/// popover first, then act on the next turn of the run loop.
final class MenuPanel {
    static let detached = MenuPanel()

    weak var window: NSWindow?
    var hide: @MainActor () -> Void = {}

    @MainActor func dismiss(then action: @escaping @MainActor () -> Void) {
        hide()
        DispatchQueue.main.async { MainActor.assumeIsolated { action() } }
    }
}

extension EnvironmentValues {
    @Entry var menuPanel = MenuPanel.detached
}

/// The icon and the small two-line speed, drawn into one template image so
/// the menu bar tints it. The last image is reused until its inputs change.
@MainActor
enum StatusItemImage {
    private static var cached: (key: String, image: NSImage)?

    static func image(symbol: String, speed: String?) -> NSImage {
        let key = "\(symbol)|\(speed ?? "")"
        if let cached, cached.key == key { return cached.image }
        let renderer = ImageRenderer(content: content(symbol: symbol, speed: speed))
        renderer.scale = NSScreen.main?.backingScaleFactor ?? 2
        let image = renderer.nsImage ?? NSImage(systemSymbolName: symbol, accessibilityDescription: nil) ?? NSImage()
        image.isTemplate = true
        cached = (key, image)
        return image
    }

    private static func content(symbol: String, speed: String?) -> some View {
        HStack(spacing: 3) {
            if let bolt = NSImage(systemSymbolName: symbol, accessibilityDescription: nil) {
                Image(nsImage: bolt)
            }
            if let speed {
                VStack(spacing: -4) {
                    ZStack {
                        Text("99.9").hidden()
                        Text(speed)
                    }
                    .font(.system(size: 9, weight: .semibold).monospacedDigit())
                    Text("t/s").font(.system(size: 11, weight: .medium))
                }
                .fixedSize()
                .offset(y: 0.5)
            }
        }
        .foregroundStyle(.black)
        .frame(height: NSStatusBar.system.thickness)
    }
}
