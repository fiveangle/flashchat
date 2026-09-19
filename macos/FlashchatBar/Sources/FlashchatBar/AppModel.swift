import AppKit
import FlashchatKit
import Observation
import ServiceManagement
@preconcurrency import UserNotifications

struct ActivityEntry: Identifiable {
    let id = UUID()
    let date = Date()
    let text: String
    let isError: Bool
}

@MainActor
@Observable
final class OperationState: Identifiable {
    let id = UUID()
    let title: String
    let arguments: [String]
    var phase = ""
    var message = "Starting…"
    var fraction: Double?
    var log: [String] = []
    var artifacts: [ArtifactRow] = []
    var done: OperationEvent?
    var handle: RunningProcess?
    var cancelRequested = false

    init(title: String, arguments: [String]) {
        self.title = title
        self.arguments = arguments
    }

    var isRunning: Bool { done == nil }
    var succeeded: Bool { done?.ok == true }

    func apply(_ event: OperationEvent) {
        switch event.event {
        case "progress":
            phase = event.phase ?? phase
            if let msg = event.message, !msg.isEmpty { message = msg }
            fraction = event.fraction
        case "log":
            if let msg = event.message {
                log.append(msg)
                if log.count > 2000 { log.removeFirst(log.count - 2000) }
            }
        case "artifact":
            if let scope = event.scope, let rel = event.relpath {
                artifacts.append(ArtifactRow(scope: scope, relpath: rel, state: event.state ?? "",
                                             detail: event.detail ?? "", required: true,
                                             satisfied: ["ok", "unhashed", "skipped"].contains(event.state ?? "")))
            }
        default:
            break
        }
    }
}

@MainActor
@Observable
final class AppModel {
    private(set) var env: FlashchatEnvironment?
    var apiState: ApiState?
    var stateError: String?
    var launcherStatus: LauncherStatus?
    var health: Health?
    var transition: ServerTransition?
    var memory = SystemMemorySnapshot.current()
    var operation: OperationState?
    var activity: [ActivityEntry] = []
    var isLoadingState = false
    var quietMode = false
    var showSpeedInMenuBar: Bool {
        didSet { UserDefaults.standard.set(showSpeedInMenuBar, forKey: "showSpeedInMenuBar") }
    }

    /// Dock icon and menu bar icon; at least one is always on, or the app
    /// would have no way back in.
    var showDockIcon: Bool = AppPresence.showDockIcon {
        didSet {
            guard showDockIcon != oldValue else { return }
            AppPresence.showDockIcon = showDockIcon
            if !showDockIcon && !showMenuBarIcon { showMenuBarIcon = true }
            AppPresence.applyActivationPolicy()
            updateDockBadge()
        }
    }

    var openWindowAtLaunch: Bool = AppPresence.openWindowAtLaunch {
        didSet { AppPresence.openWindowAtLaunch = openWindowAtLaunch }
    }

    var showMenuBarIcon: Bool = AppPresence.showMenuBarIcon {
        didSet {
            guard showMenuBarIcon != oldValue else { return }
            AppPresence.showMenuBarIcon = showMenuBarIcon
            if !showMenuBarIcon && !showDockIcon { showDockIcon = true }
        }
    }

    private var meter = ThroughputMeter()
    private var pollTask: Task<Void, Never>?
    private var lastStatusRefresh = Date.distantPast
    private var fileStamps: [String: Date] = [:]
    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 2
        config.timeoutIntervalForResource = 3
        return URLSession(configuration: config)
    }()

    init() {
        showSpeedInMenuBar = UserDefaults.standard.object(forKey: "showSpeedInMenuBar") as? Bool ?? true
        env = Self.discoverRepo().map(FlashchatEnvironment.init(repoRoot:))
        startPolling()
        Task { await reloadAll() }
    }

    var diagnostics: String {
        var lines = ["repo: \(env?.repoRoot.path ?? "nil")",
                     "pythonReady: \(env?.pythonReady ?? false)",
                     "stateError: \(stateError ?? "nil")",
                     "models: \(apiState?.models.map { "\($0.id)=\($0.variants.map { "\($0.name):\($0.ready)" })" } ?? [])",
                     "selected: \(apiState?.selected.map { "\($0.model)/\($0.variant) ready=\($0.ready) mem=\($0.memory.totalBytes)" } ?? "nil")",
                     "settings: \(apiState?.settings.count ?? 0)",
                     "launcher: \(launcherStatus.map { "\($0.server.state) \($0.server.url)" } ?? "nil")",
                     "display: \(display.title) / \(display.symbol)",
                     "memory: \(Format.bytes(memory.availableBytes)) \(memory.pressure.title)"]
        lines += activity.map { "activity: \($0.text)" }
        return lines.joined(separator: "\n")
    }

    var backend: Backend? { env.map(Backend.init(env:)) }

    var display: ServerDisplay {
        ServerDisplay.derive(status: launcherStatus, health: health, transition: transition,
                             tokensPerSecond: meter.tokensPerSecond)
    }

    var tokensPerSecond: Double? { meter.tokensPerSecond }

    var apiURL: String {
        if let url = launcherStatus?.server.url { return url }
        let host = apiState?.config["SERVER_HOST"] ?? "127.0.0.1"
        let port = apiState?.config["SERVER_PORT"] ?? "8000"
        return "http://\(host):\(port)/v1"
    }

    var serverLogURL: URL? {
        resolveServerLog(launcherStatus?.server.log ?? apiState?.config["SERVER_LOG_PATH"] ?? "")
    }

    var quietFlagURL: URL? {
        guard let dir = apiState?.configDir else { return nil }
        return URL(fileURLWithPath: dir).appendingPathComponent("menubar-quiet")
    }

    // MARK: Repo discovery

    static func discoverRepo() -> URL? {
        var candidates: [URL] = []
        if let saved = UserDefaults.standard.string(forKey: "repoRoot") {
            candidates.append(URL(fileURLWithPath: saved))
        }
        if let embedded = Bundle.main.object(forInfoDictionaryKey: "FlashchatRepoRoot") as? String {
            candidates.append(URL(fileURLWithPath: embedded))
        }
        var dir = Bundle.main.bundleURL
        for _ in 0..<6 {
            dir.deleteLastPathComponent()
            candidates.append(dir)
        }
        return candidates.first { FlashchatEnvironment(repoRoot: $0).isValidRepo }
    }

    func setRepo(_ url: URL) {
        let candidate = FlashchatEnvironment(repoRoot: url)
        guard candidate.isValidRepo else {
            Alerts.error("That folder is not a Flashchat checkout",
                         "Choose the folder that contains the ‘flashchat’ launcher and ‘modelmgr’.")
            return
        }
        UserDefaults.standard.set(url.path, forKey: "repoRoot")
        env = candidate
        Task { await reloadAll() }
    }

    // MARK: Refresh

    func reloadAll() async {
        await refreshStatus()
        await refreshState(fast: true)
        await refreshState(fast: false)
    }

    func refreshState(fast: Bool = false) async {
        guard let backend else { return }
        guard backend.env.pythonReady else {
            stateError = nil
            apiState = nil
            return
        }
        isLoadingState = true
        defer { isLoadingState = false }
        do {
            apiState = try await backend.api(fast ? ["state", "--no-offload"] : ["state"],
                                             as: ApiState.self)
            stateError = nil
            updateQuietMode()
        } catch {
            stateError = error.localizedDescription
        }
    }

    func refreshStatus() async {
        guard let backend else { return }
        lastStatusRefresh = Date()
        if let status = try? await backend.status() {
            launcherStatus = status
        }
    }

    private func updateQuietMode() {
        guard let flag = quietFlagURL else { return }
        quietMode = FileManager.default.fileExists(atPath: flag.path)
    }

    func setQuietMode(_ enabled: Bool) {
        guard let flag = quietFlagURL else { return }
        if enabled {
            FileManager.default.createFile(atPath: flag.path, contents: Data())
        } else {
            try? FileManager.default.removeItem(at: flag)
        }
        quietMode = enabled
        log(enabled ? "Quiet mode on — polling slowed for benchmarks." : "Quiet mode off.")
    }

    // MARK: Polling

    private func startPolling() {
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.pollOnce()
                try? await Task.sleep(for: .seconds(self.pollInterval))
            }
        }
    }

    var statusPollSeconds: Double = AppPresence.statusPollSeconds {
        didSet { AppPresence.statusPollSeconds = statusPollSeconds }
    }

    private var pollInterval: Double {
        if quietMode { return 30 }
        if transition != nil { return 1 }
        // Prompt reading and generation move fast; halve the interval so
        // progress and tok/s stay readable.
        if display.isBusy { return max(1, statusPollSeconds / 2) }
        return statusPollSeconds
    }

    /// The launcher publishes exactly which files feed its "restart needed"
    /// signature; watching their timestamps costs microseconds, so
    /// `status --json` only runs when one of them changed. The pid file covers
    /// the server being started or stopped from a terminal.
    private func watchedFiles() -> [String] {
        guard let status = launcherStatus else { return [] }
        return (status.watch ?? [status.configFile]) + [status.server.pidFile]
    }

    private func watchedFilesChanged() -> Bool {
        var stamps: [String: Date] = [:]
        for path in watchedFiles() {
            let attributes = try? FileManager.default.attributesOfItem(atPath: path)
            stamps[path] = (attributes?[.modificationDate] as? Date) ?? .distantPast
        }
        defer { fileStamps = stamps }
        guard !fileStamps.isEmpty else { return false }
        return stamps != fileStamps
    }

    private func pollOnce() async {
        updateQuietMode()
        // Assign only on change: @Observable notifies (and SwiftUI redraws)
        // even when the new value is equal.
        let snapshot = SystemMemorySnapshot.current()
        if snapshot != memory { memory = snapshot }
        let hadHealth = health != nil
        let fresh = await fetchHealth()
        if fresh != health { health = fresh }
        meter.record(health: health, at: Date().timeIntervalSinceReferenceDate)
        updateDockBadge()
        // `status --json` costs ~0.28 CPU-seconds (bash, cksums for the
        // restart-needed signature, lsof), so it is never on a plain timer.
        // /health covers moment-to-moment state; this runs when the server
        // appears or disappears, when a watched file changes (a config edit or
        // an engine rebuild from a terminal), and on demand — when the popover
        // opens, the window appears, or an action finishes.
        if hadHealth != (health != nil) || watchedFilesChanged() {
            await refreshStatus()
        }
    }

    /// Called when the user actually looks: the popover opening or the window
    /// appearing. Cheap enough at human speed, and keeps "restart needed"
    /// honest exactly when it is read.
    func refreshStatusOnDemand() {
        guard Date().timeIntervalSince(lastStatusRefresh) > 2 else { return }
        Task { await refreshStatus() }
    }

    private func fetchHealth() async -> Health? {
        let host: String
        let port: Int
        if let server = launcherStatus?.server {
            host = server.host
            port = server.port
        } else if let config = apiState?.config {
            host = ["", "0.0.0.0", "::"].contains(config["SERVER_HOST"] ?? "")
                ? "127.0.0.1" : (config["SERVER_HOST"] ?? "127.0.0.1")
            port = Int(config["SERVER_PORT"] ?? "") ?? 8000
        } else {
            return nil
        }
        guard port > 0, let url = URL(string: "http://\(host):\(port)/health") else { return nil }
        guard let (data, response) = try? await session.data(from: url),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return try? JSONDecoder().decode(Health.self, from: data)
    }

    // MARK: Server control

    func startServer() async {
        guard let backend, transition == nil else { return }
        if let selected = apiState?.selected, !selected.ready {
            let open = Alerts.confirm(
                "\(apiState?.selectedModel?.name ?? selected.model) [\(selected.variant)] is not ready",
                "Its runtime files need to be built or restored before the server can start.",
                confirm: "Open Models")
            if open { WindowRouter.shared.open(.models) }
            return
        }
        if let estimate = apiState?.selected?.memory.totalBytes {
            switch MemoryPreflight.evaluate(estimateBytes: estimate, memory: SystemMemorySnapshot.current()) {
            case .ok:
                break
            case .tight(let message):
                guard Alerts.confirm("Memory is tight", message + "\n\nStart the server anyway?",
                                     confirm: "Start Anyway") else { return }
            case .insufficient(let message):
                guard Alerts.confirm("Not enough free memory", message
                                     + "\n\nStarting anyway can make this Mac swap heavily or freeze.",
                                     confirm: "Start Anyway", destructive: true) else { return }
            }
        }
        transition = .starting
        log("Starting server…")
        defer { transition = nil }
        do {
            let result = try await backend.launcher(["serve", "--non-interactive"])
            await refreshStatus()
            health = await fetchHealth()
            if result.succeeded {
                log("Server started at \(apiURL).")
            } else {
                log("Server failed to start:\n\(result.tail())", error: true)
                Alerts.error("The server did not start", result.tail())
            }
        } catch {
            Alerts.error("The server did not start", error.localizedDescription)
        }
    }

    func stopServer(confirmExternal: Bool = true) async {
        guard let backend, transition == nil else { return }
        let external = launcherStatus?.server.state == "external"
        if external && confirmExternal {
            guard Alerts.confirm("Stop the external server?",
                                 "A Flashchat server is listening on port \(launcherStatus?.server.port ?? 0), "
                                 + "but it was not started from here.", confirm: "Stop It") else { return }
        }
        if display.isBusy {
            guard Alerts.confirm("The server is busy",
                                 "Stopping now cancels the request it is processing.",
                                 confirm: "Stop Anyway", destructive: true) else { return }
        }
        transition = .stopping
        log("Stopping server…")
        defer { transition = nil }
        let base = external ? ["serve", "--stop", "--external"] : ["serve", "--stop"]
        do {
            var result = try await backend.launcher(base)
            if !result.succeeded {
                if Alerts.confirm("The server did not stop cleanly",
                                  result.tail() + "\n\nForce it to stop?",
                                  confirm: "Force Stop", destructive: true) {
                    result = try await backend.launcher(base + ["--force"])
                }
            }
            if result.succeeded {
                log("Server stopped.")
            } else {
                log("Stop failed:\n\(result.tail())", error: true)
            }
        } catch {
            Alerts.error("Could not stop the server", error.localizedDescription)
        }
        health = nil
        await refreshStatus()
    }

    func restartServer() async {
        guard transition == nil else { return }
        await stopServer(confirmExternal: false)
        guard launcherStatus?.server.state == "stopped" else { return }
        await startServer()
    }

    // MARK: Model and settings

    func selectModel(_ modelId: String, variant: String) async {
        guard let backend else { return }
        do {
            let result = try await backend.api(["select", "--model", modelId, "--variant", variant],
                                               as: SelectResult.self)
            log("Selected \(result.resolvedId).")
            result.warnings.forEach { log($0) }
            await refreshState()
            await refreshStatus()
            if !result.ready {
                if Alerts.confirm("\(modelName(modelId)) [\(variant)] is not ready",
                                  "Prepare it now? You can review what will be downloaded or built first.",
                                  confirm: "Prepare…") {
                    WindowRouter.shared.open(.models, model: modelId, buildVariant: variant)
                }
            } else if result.serverRunning {
                offerRestart(reason: "The server is still running the previous model.")
            }
        } catch {
            Alerts.error("Could not select the model", error.localizedDescription)
        }
    }

    func setEnabled(_ modelId: String, _ enabled: Bool) async {
        guard let backend else { return }
        do {
            _ = try await backend.api(["enable", "--model", modelId, "--enabled", enabled ? "1" : "0"],
                                      as: OkResult.self)
            await refreshState()
        } catch {
            Alerts.error("Could not update the model", error.localizedDescription)
        }
    }

    func saveSettings(_ changes: [String: String]) async -> SetResult? {
        guard let backend, !changes.isEmpty else { return nil }
        do {
            let data = try JSONSerialization.data(withJSONObject: changes)
            let result = try await backend.api(["set", "--values", String(decoding: data, as: UTF8.self)],
                                               as: SetResult.self)
            log("Saved settings: \(changes.keys.sorted().joined(separator: ", ")).")
            await refreshState()
            await refreshStatus()
            if result.serverRunning && result.changed {
                offerRestart(reason: "The running server still uses the previous settings.")
            }
            return result
        } catch {
            Alerts.error("Settings were not saved", error.localizedDescription)
            return nil
        }
    }

    func offerRestart(reason: String) {
        guard launcherStatus?.server.state != "external" else { return }
        if Alerts.confirm("Restart the server?", reason, confirm: "Restart Now", cancel: "Later") {
            Task { await restartServer() }
        }
    }

    func modelName(_ id: String) -> String {
        apiState?.models.first { $0.id == id }?.name ?? id
    }

    // MARK: Operations

    @discardableResult
    func runOperation(_ title: String, _ arguments: [String], presentSheet: Bool = true,
                      onDone: (@MainActor (OperationEvent) -> Void)? = nil) -> OperationState? {
        guard let backend else { return nil }
        if let current = operation, current.isRunning {
            Alerts.error("Another operation is running", "Wait for “\(current.title)” to finish or cancel it.")
            return nil
        }
        let op = OperationState(title: title, arguments: arguments)
        operation = op
        log("\(title)…")
        do {
            op.handle = try backend.operation(arguments, onEvent: { event in
                Task { @MainActor in op.apply(event) }
            }, completion: { done, stderr in
                Task { @MainActor in
                    op.done = done
                    op.fraction = done.ok == true ? 1 : op.fraction
                    op.message = done.message ?? (done.ok == true ? "Done." : "Failed.")
                    if done.ok != true, !stderr.isEmpty {
                        op.log.append(contentsOf: stderr.split(separator: "\n").suffix(40).map(String.init))
                    }
                    self.log("\(title): \(op.message)", error: done.ok != true)
                    self.notify(title: title, body: op.message)
                    await self.refreshState()
                    onDone?(done)
                }
            })
        } catch {
            op.done = OperationEvent(event: "done", message: error.localizedDescription, ok: false,
                                     code: "launch")
            op.message = error.localizedDescription
        }
        if presentSheet {
            // Let a dismissing sheet finish before presenting the progress sheet.
            Task {
                try? await Task.sleep(for: .milliseconds(350))
                WindowRouter.shared.showOperation()
            }
        }
        return op
    }

    func cancelOperation() {
        guard let op = operation, op.isRunning else { return }
        op.cancelRequested = true
        op.message = "Cancelling…"
        op.handle?.cancel()
    }

    // MARK: Setup helpers

    func setUpPython() async {
        guard let backend else { return }
        guard Alerts.confirm("Set up Flashchat's Python environment?",
                             "This creates a virtual environment inside the Flashchat folder "
                             + "(metal_infer/.venv) and installs numpy and huggingface_hub into it. "
                             + "Nothing is installed system-wide.", confirm: "Set Up") else { return }
        isLoadingState = true
        log("Setting up the Python environment…")
        let result = try? await backend.launcher(["doctor"])
        isLoadingState = false
        if result?.succeeded != true, !(backend.env.pythonReady) {
            Alerts.error("Setup did not finish", result?.tail() ?? "")
        }
        await reloadAll()
    }

    func openInTerminal(_ arguments: [String] = []) {
        guard let env else { return }
        let script = ([env.launcher.path] + arguments).map { "'\($0.replacingOccurrences(of: "'", with: "'\\''"))'" }
            .joined(separator: " ")
        let escaped = script.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        let source = "tell application \"Terminal\"\nactivate\ndo script \"\(escaped)\"\nend tell"
        var error: NSDictionary?
        NSAppleScript(source: source)?.executeAndReturnError(&error)
        if let error { Alerts.error("Could not open Terminal", "\(error)") }
    }

    // MARK: Launch at login

    var launchAtLogin: Bool {
        get { SMAppService.mainApp.status == .enabled }
        set {
            do {
                if newValue { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            } catch {
                Alerts.error("Could not change the login item", error.localizedDescription)
            }
        }
    }

    // MARK: Dock

    /// Mirrors the menu bar icon: "!" when attention is needed, the decode
    /// speed while generating (unless quiet), nothing otherwise.
    func updateDockBadge() {
        guard showDockIcon else {
            NSApp.dockTile.badgeLabel = nil
            return
        }
        let display = self.display
        var badge: String?
        switch display.activity {
        case .restartNeeded, .unreachable:
            badge = "!"
        case .generating:
            if !quietMode, showSpeedInMenuBar, let tps = tokensPerSecond { badge = String(format: "%.0f", tps) }
        default:
            badge = nil
        }
        if NSApp.dockTile.badgeLabel != badge { NSApp.dockTile.badgeLabel = badge }
    }

    // MARK: Logging and notifications

    func log(_ text: String, error: Bool = false) {
        activity.append(ActivityEntry(text: text, isError: error))
        if activity.count > 500 { activity.removeFirst(activity.count - 500) }
    }

    private func notify(title: String, body: String) {
        guard Bundle.main.bundleIdentifier != nil, !NSApp.isActive else { return }
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .sound]) { granted, _ in
            guard granted else { return }
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
        }
    }
}
