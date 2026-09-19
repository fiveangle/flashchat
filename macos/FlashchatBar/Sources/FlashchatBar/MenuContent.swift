import AppKit
import FlashchatKit
import SwiftUI

struct MenuContent: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if model.env == nil {
                SetupNotice(text: "Flashchat folder not found.", action: "Choose Folder…") {
                    if let url = Alerts.chooseFolder("Choose your Flashchat folder") { model.setRepo(url) }
                }
            } else if model.env?.pythonReady == false {
                SetupNotice(text: "Flashchat's Python environment isn't set up.", action: "Set Up…") {
                    Task { await model.setUpPython() }
                }
            }
            ServerHeader()
            Divider()
            ModelLine()
            if model.health != nil { ContextLine() }
            MemoryLine()
            if let op = model.operation {
                OperationRow(operation: op)
            }
            Divider()
            ServerButtons()
            Divider()
            Footer()
        }
        .padding(14)
        .frame(width: 330)
        .onAppear {
            clearInitialFocus()
            model.refreshStatusOnDemand()
        }
    }

    /// SwiftUI focuses the first control when the popover opens, which draws a
    /// focus ring for anyone using keyboard navigation. Start with nothing
    /// focused; Tab still enters the key-view loop at the first control.
    private func clearInitialFocus() {
        DispatchQueue.main.async {
            guard let panel = NSApp.windows.first(where: { $0.isKeyWindow && $0.level != .normal })
                    ?? NSApp.keyWindow else { return }
            panel.makeFirstResponder(nil)
        }
    }
}

private struct SetupNotice: View {
    let text: String
    let action: String
    let perform: () -> Void

    var body: some View {
        HStack {
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .font(.callout)
            Spacer()
            Button(action, action: perform)
        }
    }
}

private struct ServerHeader: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let display = model.display
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                StatusDot(display: display)
                Text(display.title).font(.headline)
                Spacer()
                if model.quietMode { Pill(text: "QUIET", color: .purple) }
                if display.activity == .starting || display.activity == .stopping {
                    ProgressView().controlSize(.small)
                }
            }
            if let detail = display.detail {
                Text(detail).font(.callout).foregroundStyle(.secondary)
            }
            if let progress = display.progress {
                ProgressView(value: progress)
            }
            if display.isRunning {
                HStack {
                    Text(model.apiURL).font(.caption.monospaced()).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    Spacer()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.apiURL, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .help("Copy API URL")
                }
            }
        }
    }
}

private struct ModelLine: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack {
            Image(systemName: "shippingbox").foregroundStyle(.secondary)
            if let state = model.apiState, let selected = state.selected {
                VStack(alignment: .leading, spacing: 1) {
                    Text(state.selectedModel?.name ?? selected.model).lineLimit(1)
                    Text("\(selected.variant) · \(selected.ready ? "ready" : "not ready")")
                        .font(.caption)
                        .foregroundStyle(selected.ready ? Color.secondary : Color.orange)
                }
            } else if model.apiState?.configExists == false {
                Text("No model selected yet").foregroundStyle(.secondary)
            } else {
                Text(model.stateError ?? "Loading…").foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer()
            ModelSwitcher()
        }
    }
}

private struct ModelSwitcher: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router

    var body: some View {
        Menu {
            if let state = model.apiState {
                ForEach(state.models.filter { $0.enabled || $0.selected }) { info in
                    Section(info.name) {
                        ForEach(info.variants) { variant in
                            Button {
                                Task { await model.selectModel(info.id, variant: variant.name) }
                            } label: {
                                let current = info.selected && info.selectedVariant == variant.name
                                Text("\(current ? "✓ " : "")\(variant.name) — \(variant.ready ? "ready" : variant.summary)")
                            }
                        }
                    }
                }
            }
            Divider()
            Button("Manage Models…") { router.open(.models) }
        } label: {
            Text("Switch")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .disabled(model.apiState == nil)
    }
}

private struct ContextLine: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let health = model.health {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: "text.alignleft").foregroundStyle(.secondary)
                    Text("Context")
                    Spacer()
                    Text("\(Format.tokensShort(health.contextUsed)) / \(Format.tokensShort(health.maxContext))")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                ProgressView(value: Double(health.contextUsed),
                             total: Double(max(health.maxContext, 1)))
                Text("Context cache \(Format.bytes(health.kvTotalBytes)) (\(health.kvQuantization))")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

private struct MemoryLine: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let mem = model.memory
        VStack(alignment: .leading, spacing: 4) {
            if let server = model.serverMemory {
                HStack {
                    Image(systemName: "server.rack").foregroundStyle(.secondary)
                    Text("Server using \(Format.bytes(server.footprintBytes))")
                    Spacer()
                }
                .help("The server process's memory, as Activity Monitor reports it. "
                      + "The memory-mapped weights (\(Format.bytes(server.residentBytes)) resident "
                      + "in total) are excluded because macOS can reclaim them.")
            }
            HStack {
                Image(systemName: "memorychip").foregroundStyle(.secondary)
                Text("\(Format.bytes(mem.availableBytes)) available")
                Spacer()
                Text(mem.pressure.title)
                    .foregroundStyle(mem.pressure == .normal ? Color.secondary
                                     : mem.pressure == .warning ? Color.orange : Color.red)
            }
            .help("Free + reclaimable RAM. Swap used: \(Format.bytes(mem.swapUsedBytes))")
        }
        .font(.callout)
        .onAppear { model.startShowingServerMemory() }
        .onDisappear { model.stopShowingServerMemory() }
    }
}

struct OperationRow: View {
    let operation: OperationState
    @Environment(WindowRouter.self) private var router

    var body: some View {
        Button {
            router.section = .models
            router.showOperation()
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: operation.isRunning ? "gearshape.2"
                          : operation.succeeded ? "checkmark.circle.fill" : "xmark.octagon.fill")
                        .foregroundStyle(operation.isRunning ? Color.blue
                                         : operation.succeeded ? Color.green : Color.red)
                    Text(operation.title).lineLimit(1)
                    Spacer()
                }
                Text(operation.message).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                if operation.isRunning {
                    if let fraction = operation.fraction {
                        ProgressView(value: fraction)
                    } else {
                        ProgressView().progressViewStyle(.linear)
                    }
                }
            }
        }
        .buttonStyle(.plain)
    }
}

private struct ServerButtons: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let display = model.display
        let working = model.transition != nil
        HStack {
            if display.isRunning {
                Button {
                    Task { await model.stopServer() }
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                if display.isOwned {
                    Button {
                        Task { await model.restartServer() }
                    } label: {
                        Label("Restart", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(display.activity == .restartNeeded ? .orange : .accentColor)
                    .opacity(display.activity == .restartNeeded ? 1 : 0.9)
                }
            } else {
                Button {
                    Task { await model.startServer() }
                } label: {
                    Label("Start Server", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.apiState?.selected == nil)
            }
            Spacer()
        }
        .disabled(working || model.env == nil)
    }
}

private struct Footer: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router

    var body: some View {
        HStack {
            Button("Open Flashchat…") { router.open(router.section) }
            Spacer()
            Menu {
                Toggle("Quiet mode (for benchmarks)", isOn: Binding(
                    get: { model.quietMode }, set: { model.setQuietMode($0) }))
                Toggle("Show speed in menu bar", isOn: Binding(
                    get: { model.showSpeedInMenuBar }, set: { model.showSpeedInMenuBar = $0 }))
                Toggle("Show Dock icon", isOn: Binding(
                    get: { model.showDockIcon }, set: { model.showDockIcon = $0 }))
                Divider()
                Button("New Chat in Terminal") { model.openInTerminal(["chat"]) }
                Button("Open Terminal Menu") { model.openInTerminal() }
                Button("Refresh") { Task { await model.reloadAll() } }
                Divider()
                Button("Quit Flashchat Menu") { NSApp.terminate(nil) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
        }
    }
}
