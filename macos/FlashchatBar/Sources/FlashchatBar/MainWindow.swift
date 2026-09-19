import AppKit
import FlashchatKit
import SwiftUI

struct MainWindow: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router

    var body: some View {
        @Bindable var router = router
        NavigationSplitView {
            List(MainSection.allCases, selection: Binding(
                get: { router.section }, set: { if let s = $0 { router.section = s } })) { section in
                Label(section.rawValue, systemImage: section.symbol).tag(section)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
        } detail: {
            VStack(spacing: 0) {
                Banners()
                switch router.section {
                case .overview: OverviewView()
                case .models: ModelsView()
                case .settings: SettingsView()
                case .logs: LogsView()
                }
            }
        }
        .sheet(isPresented: $router.showingOperation) {
            if let op = model.operation {
                OperationSheet(operation: op)
            }
        }
        .onAppear { NSApp.activate(ignoringOtherApps: true) }
    }
}

private struct Banners: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router

    var body: some View {
        VStack(spacing: 0) {
            if model.env == nil {
                Banner(text: "Flashchat folder not found. Choose the folder that contains the ‘flashchat’ launcher.",
                       action: "Choose Folder…") {
                    if let url = Alerts.chooseFolder("Choose your Flashchat folder") { model.setRepo(url) }
                }
            } else if model.env?.pythonReady == false {
                Banner(text: "Flashchat's Python environment isn't set up yet.", action: "Set Up…") {
                    Task { await model.setUpPython() }
                }
            } else if let state = model.apiState {
                if !state.configExists {
                    Banner(text: "Welcome! Pick a model to get started — select a variant and choose Use.",
                           action: router.section == .models ? nil : "Choose Model") {
                        router.section = .models
                    }
                } else if state.migrationNeeded {
                    Banner(text: "Flashchat's model storage layout needs a one-time update "
                           + "(in place; originals untouched).", action: "Update Now") {
                        if Alerts.confirm("Update the model storage layout?",
                                          "Variant-independent files move into a shared folder and integrity "
                                          + "manifests are added. No re-extraction is needed.",
                                          confirm: "Update") {
                            model.runOperation("Updating storage layout", ["migrate"])
                        }
                    }
                }
            } else if let error = model.stateError {
                Banner(text: error, action: "Retry") { Task { await model.reloadAll() } }
            }
        }
    }
}

private struct Banner: View {
    let text: String
    let action: String?
    let perform: () -> Void

    var body: some View {
        HStack {
            Image(systemName: "info.circle.fill").foregroundStyle(.blue)
            Text(text).fixedSize(horizontal: false, vertical: true)
            Spacer()
            if let action { Button(action, action: perform) }
        }
        .padding(10)
        .background(Color.blue.opacity(0.08))
    }
}

// MARK: - Overview

struct OverviewView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Form {
            ServerSection()
            MemorySection()
            AppSection()
        }
        .formStyle(.grouped)
        .navigationTitle("Overview")
    }
}

private struct ServerSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let display = model.display
        Section("Server") {
            LabeledContent("Status") {
                HStack(spacing: 6) {
                    StatusDot(display: display)
                    Text(display.title)
                }
            }
            if let detail = display.detail { LabeledContent("Activity", value: detail) }
            LabeledContent("API") {
                HStack {
                    Text(model.apiURL).textSelection(.enabled).font(.body.monospaced())
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.apiURL, forType: .string)
                    } label: { Image(systemName: "doc.on.doc") }
                        .buttonStyle(.borderless)
                }
            }
            if let server = model.launcherStatus?.server {
                if let pid = server.pid { LabeledContent("Process", value: "PID \(pid)") }
                LabeledContent("Listening on", value: "\(server.bind):\(server.port)")
            }
            if let health = model.health {
                LabeledContent("Serving", value: health.model)
                LabeledContent("Context used",
                               value: "\(Format.count(health.contextUsed)) of \(Format.count(health.maxContext)) tokens")
            }
            if model.launcherStatus?.binariesCurrent == false {
                LabeledContent("Engine") {
                    Text("Will be rebuilt on the next start (sources changed)").foregroundStyle(.secondary)
                }
            }
            HStack {
                if display.isRunning {
                    Button("Stop") { Task { await model.stopServer() } }
                    if display.isOwned { Button("Restart") { Task { await model.restartServer() } } }
                } else {
                    Button("Start Server") { Task { await model.startServer() } }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.apiState?.selected == nil)
                }
            }
            .disabled(model.transition != nil)
        }
    }
}

private struct MemorySection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Section("Memory") {
            let mem = model.memory
            LabeledContent("Available now", value: "\(Format.bytes(mem.availableBytes)) of \(Format.bytes(mem.totalBytes))")
            LabeledContent("Pressure", value: mem.pressure.title)
            LabeledContent("Swap used", value: Format.bytes(mem.swapUsedBytes))
            if let estimate = model.apiState?.selected?.memory {
                LabeledContent("Server needs (est.)", value: Format.bytes(estimate.totalBytes))
                LabeledContent("  Weights in RAM", value: Format.bytes(estimate.weightsBytes))
                LabeledContent("  Context cache",
                               value: "\(Format.bytes(estimate.kvCacheBytes)) (\(Format.tokensShort(estimate.contextWindow)) tokens, \(estimate.kvQuant))")
                if estimate.expertCacheMaxBytes > 0 {
                    LabeledContent("  Expert cache (up to)", value: Format.bytes(estimate.expertCacheMaxBytes))
                }
                Text("Experts stream from the SSD and are not counted. The expert cache only grows into free RAM.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

private struct AppSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Section("App") {
            LabeledContent("Flashchat folder") {
                HStack {
                    Text(model.env?.repoRoot.path ?? "Not set").lineLimit(1).truncationMode(.middle)
                    Button("Change…") {
                        if let url = Alerts.chooseFolder("Choose your Flashchat folder",
                                                         start: model.env?.repoRoot.path) {
                            model.setRepo(url)
                        }
                    }
                }
            }
            Toggle("Launch at login", isOn: Binding(get: { model.launchAtLogin },
                                                     set: { model.launchAtLogin = $0 }))
            Toggle("Show generation speed in the menu bar", isOn: $model.showSpeedInMenuBar)
            Toggle(isOn: Binding(get: { model.quietMode }, set: { model.setQuietMode($0) })) {
                Text("Quiet mode")
                Text("Polls every 30 s and keeps the menu bar icon static, so benchmarks see no GPU or CPU from this app. Scripts can toggle it with the menubar-quiet file in the Flashchat config folder.")
            }
            HStack {
                Button("New Chat in Terminal") { model.openInTerminal(["chat"]) }
                Button("Open Terminal Menu") { model.openInTerminal() }
                Button("Copy Diagnostics") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(model.diagnostics, forType: .string)
                }
                if let file = model.apiState?.configFile {
                    Button("Reveal Config File") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: file)])
                    }
                }
            }
        }
    }
}

// MARK: - Logs

struct LogsView: View {
    @Environment(AppModel.self) private var model
    @State private var source = 0
    @State private var serverLog = ""
    @State private var follow = true

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Picker("", selection: $source) {
                    Text("Server log").tag(0)
                    Text("App activity").tag(1)
                }
                .pickerStyle(.segmented)
                .fixedSize()
                Spacer()
                Toggle("Follow", isOn: $follow).toggleStyle(.checkbox)
                if source == 0, let url = model.serverLogURL {
                    Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                }
            }
            .padding(10)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    Group {
                        if source == 0 {
                            Text(serverLog.isEmpty ? "No server log yet." : serverLog)
                        } else {
                            Text(model.activity.map { entry in
                                "\(entry.date.formatted(date: .omitted, time: .standard))  \(entry.isError ? "⚠︎ " : "")\(entry.text)"
                            }.joined(separator: "\n"))
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    Color.clear.frame(height: 1).id("bottom")
                }
                .onChange(of: serverLog) { if follow { proxy.scrollTo("bottom") } }
                .onChange(of: model.activity.count) { if follow { proxy.scrollTo("bottom") } }
            }
        }
        .navigationTitle("Logs")
        .task(id: source) {
            while !Task.isCancelled && source == 0 {
                serverLog = Self.tail(model.serverLogURL)
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    static func tail(_ url: URL?, maxBytes: UInt64 = 256 * 1024) -> String {
        guard let url, let handle = try? FileHandle(forReadingFrom: url) else { return "" }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        try? handle.seek(toOffset: size > maxBytes ? size - maxBytes : 0)
        let data = (try? handle.readToEnd()) ?? Data()
        var text = String(decoding: data, as: UTF8.self)
        if size > maxBytes, let firstNewline = text.firstIndex(of: "\n") {
            text = String(text[text.index(after: firstNewline)...])
        }
        return text
    }
}
