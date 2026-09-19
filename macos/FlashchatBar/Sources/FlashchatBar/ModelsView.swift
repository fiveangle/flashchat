import AppKit
import FlashchatKit
import SwiftUI

struct ModelsView: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router
    @State private var addingModel = false

    var body: some View {
        @Bindable var router = router
        HSplitView {
            VStack(spacing: 0) {
                List(selection: $router.selectedModel) {
                    ForEach(model.apiState?.models ?? []) { info in
                        ModelListRow(info: info).tag(info.id)
                    }
                }
                Divider()
                HStack {
                    Button {
                        addingModel = true
                    } label: {
                        Label("Add Model…", systemImage: "plus")
                    }
                    Spacer()
                    if model.isLoadingState { ProgressView().controlSize(.small) }
                    Button {
                        Task { await model.refreshState() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .help("Re-check models")
                }
                .buttonStyle(.borderless)
                .padding(8)
            }
            .frame(minWidth: 240, idealWidth: 270, maxWidth: 340)

            Group {
                if let id = router.selectedModel,
                   let info = model.apiState?.models.first(where: { $0.id == id }) {
                    ModelDetailView(info: info)
                } else {
                    ContentUnavailableView("Select a model", systemImage: "shippingbox",
                                           description: Text("Choose a model to see its variants and storage."))
                }
            }
            .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle("Models")
        .sheet(isPresented: $addingModel) { AddModelSheet() }
        .sheet(item: $router.buildRequest) { request in
            BuildSheet(modelId: request.model, variant: request.variant, repair: request.repair)
        }
        .onAppear {
            if router.selectedModel == nil {
                router.selectedModel = model.apiState?.models.first(where: \.selected)?.id
                    ?? model.apiState?.models.first(where: \.suggestedDefault)?.id
            }
        }
    }
}

private struct ModelListRow: View {
    let info: ModelInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(info.name).lineLimit(1)
                if info.selected { Pill(text: "IN USE", color: .accentColor) }
                if info.suggestedDefault && !info.selected { Pill(text: "RECOMMENDED", color: .green) }
            }
            Text(info.hfRepo).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            HStack(spacing: 4) {
                ForEach(info.variants) { variant in
                    Pill(text: variant.name, color: variant.ready ? .green : variant.offloaded ? .blue : .secondary)
                }
                if !info.enabled { Pill(text: "disabled") }
            }
        }
        .padding(.vertical, 3)
        .opacity(info.enabled || info.selected ? 1 : 0.6)
    }
}

private enum ModelSheet: Identifiable {
    case restore, offload, delete
    var id: Int { hashValue }
}

struct ModelDetailView: View {
    let info: ModelInfo
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router
    @State private var detail: ModelDetail?
    @State private var sheet: ModelSheet?

    var body: some View {
        Form {
            Section {
                LabeledContent("Repository", value: info.hfRepo)
                if info.maxContext > 0 {
                    LabeledContent("Max context", value: "\(Format.count(info.maxContext)) tokens")
                }
                if info.numExpertsPerTok > 0 {
                    LabeledContent("Experts per token", value: "\(info.numExpertsPerTok)")
                }
                Toggle("Show in model menus", isOn: Binding(
                    get: { info.enabled }, set: { value in Task { await model.setEnabled(info.id, value) } }))
            } header: {
                Text(info.name).font(.title2.bold())
            }

            Section("Variants") {
                ForEach(info.variants) { variant in
                    VariantRow(info: info, variant: variant)
                }
            }

            Section("Storage") {
                LabeledContent("On this Mac") {
                    if let snapshot = info.snapshot {
                        HStack {
                            Text(snapshot).lineLimit(1).truncationMode(.middle)
                            Button("Reveal") {
                                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: snapshot)])
                            }
                        }
                    } else {
                        Text("Not stored locally").foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Original files",
                               value: info.originalsLocal ? Format.bytes(info.originalsBytes) : "Not on this Mac")
                LabeledContent("Offload storage", value: info.archiveTitle)
                if !info.pendingOffloadSync.isEmpty {
                    LabeledContent("Offload copy") {
                        HStack {
                            Text("Out of date (\(info.pendingOffloadSync.joined(separator: ", ")))")
                                .foregroundStyle(.orange)
                            Button("Update") {
                                model.runOperation("Updating offload copy of \(info.name)",
                                                   ["sync-offload", "--model", info.id])
                            }
                        }
                    }
                }
                HStack {
                    Button("Verify…") { verify() }.disabled(!info.isLocal)
                        .help("Hash every artifact against its recorded baseline")
                    Button("Restore…") { sheet = .restore }.disabled(info.archive == "none")
                    Button("Offload…") { sheet = .offload }.disabled(!info.originalsLocal)
                    Spacer()
                    Button("Delete…", role: .destructive) { sheet = .delete }.disabled(!info.isLocal)
                }
            }

            if let detail, !detail.artifacts.isEmpty {
                Section("Artifacts") {
                    ForEach(detail.artifacts) { row in
                        ArtifactLine(row: row)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .task(id: info) { await loadDetail() }
        .sheet(item: $sheet) { which in
            switch which {
            case .restore: RestoreSheet(info: info)
            case .offload: OffloadSheet(info: info)
            case .delete: DeleteSheet(info: info)
            }
        }
    }

    private func loadDetail() async {
        guard info.isLocal, let backend = model.backend else {
            detail = nil
            return
        }
        detail = try? await backend.api(["model", "--model", info.id], as: ModelDetail.self)
    }

    private func verify() {
        guard Alerts.confirm("Verify \(info.name)?",
                             "Every artifact is re-hashed and compared with its recorded baseline. "
                             + "This reads the whole model from disk and can take several minutes.",
                             confirm: "Verify") else { return }
        model.runOperation("Verifying \(info.name)", ["verify", "--model", info.id])
    }
}

private struct VariantRow: View {
    let info: ModelInfo
    let variant: VariantInfo
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router

    var body: some View {
        let inUse = info.selected && info.selectedVariant == variant.name
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(variant.name).font(.body.weight(.medium))
                    Text("\(variant.bits)-bit").foregroundStyle(.secondary)
                    if inUse { Pill(text: "IN USE", color: .accentColor) }
                }
                Text(variant.ready ? "Ready · \(Format.bytes(variant.localBytes))" : variant.summary)
                    .font(.caption)
                    .foregroundStyle(variant.ready ? Color.secondary : Color.orange)
            }
            Spacer()
            if variant.ready {
                Button("Repair…") {
                    router.buildRequest = .init(model: info.id, variant: variant.name, repair: true)
                }
                .help("Rebuild anything missing or failing verification")
            } else {
                Button("Prepare…") {
                    router.buildRequest = .init(model: info.id, variant: variant.name, repair: false)
                }
            }
            Button("Use") {
                Task { await model.selectModel(info.id, variant: variant.name) }
            }
            .disabled(inUse)
        }
    }
}

private struct ArtifactLine: View {
    let row: ArtifactRow

    var body: some View {
        HStack {
            Image(systemName: row.satisfied ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                .foregroundStyle(row.satisfied ? .green : (row.required ? .orange : .secondary))
            Text("\(row.scope)/\(row.relpath)").font(.callout.monospaced())
            Spacer()
            Text(row.state + (row.detail.isEmpty ? "" : " — \(row.detail)"))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Sheets

private struct SheetFrame<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.title3.bold())
            content
        }
        .padding(20)
        .frame(width: 540)
    }
}

struct BuildSheet: View {
    let modelId: String
    let variant: String
    let repair: Bool
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var plan: BuildPlan?
    @State private var error: String?
    /// "" until the user picks, when a choice is needed; "build" = build from what's available.
    @State private var source = ""

    var body: some View {
        SheetFrame(title: "\(repair ? "Repair" : "Prepare") \(model.modelName(modelId)) [\(variant)]") {
            if let plan {
                planBody(plan)
            } else if let error {
                Text(error).foregroundStyle(.red)
            } else {
                ProgressView("Checking what's needed…")
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(actionTitle) { run() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canRun)
            }
        }
        .task { await load() }
    }

    private var needsChoice: Bool {
        guard let plan else { return false }
        return plan.needsSource || plan.runtimeRestore != nil
    }

    private var canRun: Bool {
        guard let plan, plan.servingConflict == nil else { return false }
        if plan.ready && plan.repair == nil { return false }
        if needsChoice { return !source.isEmpty }
        return true
    }

    private var actionTitle: String {
        switch source {
        case "restore-runtime": return "Restore"
        case "download-local", "download-offload": return "Download & Build"
        default: return repair ? "Repair" : "Build"
        }
    }

    @ViewBuilder
    private func planBody(_ plan: BuildPlan) -> some View {
        if let conflict = plan.servingConflict {
            Label(conflict, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        }
        if plan.ready && plan.repair == nil {
            Label("Everything is already in place.", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        } else {
            if let repair = plan.repair {
                if !repair.rebuild.isEmpty {
                    Text("These will be deleted, then rebuilt: " + repair.rebuild.joined(separator: ", "))
                        .foregroundStyle(.orange)
                }
                if !repair.create.isEmpty {
                    Text("Missing, will be created: " + repair.create.joined(separator: ", "))
                }
                if repair.rebuild.isEmpty && repair.create.isEmpty {
                    Text("Nothing buildable is missing or broken.").foregroundStyle(.secondary)
                }
            }
            if !plan.steps.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Steps").font(.headline)
                    ForEach(plan.steps) { step in
                        HStack(alignment: .top) {
                            Image(systemName: "circle").font(.caption2).foregroundStyle(.secondary)
                            VStack(alignment: .leading) {
                                Text(step.description)
                                Text("\(step.scope)/\(step.artifacts.joined(separator: ", ")) · \(step.reason)")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            if plan.totalBytes > 0 {
                Text("Disk needed: ~\(Format.bytes(plan.totalBytes)) (free: \(Format.bytes(plan.freeBytes)))")
                    .foregroundStyle(plan.totalBytes > plan.freeBytes ? .red : .secondary)
            }
            if needsChoice {
                Picker("Get the files from", selection: $source) {
                    if let restore = plan.runtimeRestore {
                        Text("Restore ready runtime files from offload (\(Format.bytes(restore.neededBytes)))"
                             + (restore.fits ? "" : " — won't fit"))
                            .tag("restore-runtime")
                    }
                    if plan.needsSource {
                        ForEach(plan.sources) { src in
                            Text(src.title).tag(src.id)
                        }
                    } else {
                        Text("Build from the original files already available").tag("build")
                    }
                }
                .pickerStyle(.radioGroup)
                if source.hasPrefix("download") {
                    Text("Downloads the original model from HuggingFace. This can be tens of gigabytes.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func load() async {
        guard let backend = model.backend else { return }
        var args = ["plan", "build", "--model", modelId, "--variant", variant]
        if repair { args.append("--repair") }
        do {
            let loaded = try await backend.api(args, as: BuildPlan.self)
            plan = loaded
            if loaded.runtimeRestore?.fits == true { source = "restore-runtime" }
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func run() {
        let chosen = source.isEmpty || source == "build" ? "auto" : source
        var args = ["build", "--model", modelId, "--variant", variant, "--source", chosen]
        if repair { args.append("--repair") }
        let name = model.modelName(modelId)
        dismiss()
        model.runOperation("\(repair ? "Repairing" : "Preparing") \(name) [\(variant)]", args) { done in
            guard done.ok == true, done.offloadSuggested == true else { return }
            if Alerts.confirm("Offload the original files?",
                              "The runtime files are self-contained, so the original model files are no "
                              + "longer needed for inference. Flashchat can copy the model to your offload "
                              + "folder and remove the local originals.", confirm: "Offload", cancel: "Not Now") {
                model.runOperation("Offloading \(name)", ["offload", "--model", modelId])
            }
        }
    }
}

private struct RestoreSheet: View {
    let info: ModelInfo
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var plan: RestorePlan?
    @State private var choice = "runtime"

    var body: some View {
        SheetFrame(title: "Restore \(info.name) from offload storage") {
            if let plan {
                if plan.available {
                    Picker("Restore", selection: $choice) {
                        ForEach(plan.options) { option in
                            Text(label(option)).tag(option.id)
                        }
                    }
                    .pickerStyle(.radioGroup)
                    if let free = plan.options.first?.freeBytes {
                        Text("Free space on this Mac: \(Format.bytes(free)). Files already restored are skipped.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                } else {
                    Text("There is no archive for this model.").foregroundStyle(.secondary)
                }
            } else {
                ProgressView()
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Restore") {
                    dismiss()
                    model.runOperation("Restoring \(info.name)", ["restore", "--model", info.id, "--what", choice])
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!(selected?.fits ?? false) || !(selected?.available ?? false))
            }
        }
        .task {
            plan = try? await model.backend?.api(["plan", "restore", "--model", info.id], as: RestorePlan.self)
        }
    }

    private var selected: RestoreOption? { plan?.options.first { $0.id == choice } }

    private func label(_ option: RestoreOption) -> String {
        if !option.available { return "\(option.title) — nothing archived" }
        if option.neededBytes == 0 { return "\(option.title) — already local" }
        return "\(option.title) — needs \(Format.bytes(option.neededBytes))"
            + (option.fits ? "" : " (won't fit: free \(Format.bytes(option.shortfallBytes)) more)")
    }
}

private struct OffloadSheet: View {
    let info: ModelInfo
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var plan: OffloadPlan?

    var body: some View {
        SheetFrame(title: "Offload \(info.name)") {
            if let plan {
                Text("Copies the full model folder to \(plan.offloadDir.isEmpty ? "the offload folder" : plan.offloadDir), records what was copied, then removes only the local original files (\(Format.bytes(plan.originalsBytes))). Runtime files stay on this Mac and keep working.")
                    .fixedSize(horizontal: false, vertical: true)
                Text("Saved system prompt caches are never copied — they can contain prompt-derived data.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(plan.heldBack.sorted(by: { $0.key < $1.key }), id: \.key) { scope, reason in
                    Label("flashchat/\(scope)/ is not copied (\(reason))", systemImage: "info.circle")
                        .foregroundStyle(.orange)
                }
                if !plan.anyReady {
                    Label("No variant is built yet — you'll need to restore the originals before building.",
                          systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                }
                ForEach(plan.errors, id: \.self) { Label($0, systemImage: "xmark.octagon").foregroundStyle(.red) }
                ForEach(plan.warnings, id: \.self) { Label($0, systemImage: "info.circle").foregroundStyle(.secondary) }
            } else {
                ProgressView()
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Offload") {
                    dismiss()
                    model.runOperation("Offloading \(info.name)", ["offload", "--model", info.id])
                }
                .keyboardShortcut(.defaultAction)
                .disabled(plan?.available != true)
            }
        }
        .task {
            plan = try? await model.backend?.api(["plan", "offload", "--model", info.id], as: OffloadPlan.self)
        }
    }
}

private struct DeleteSheet: View {
    let info: ModelInfo
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var plan: DeletePlan?
    @State private var choice = ""
    @State private var typed = ""

    var body: some View {
        SheetFrame(title: "Delete files for \(info.name)") {
            if let plan {
                Text("Offload copies are not changed.").foregroundStyle(.secondary)
                Picker("Delete", selection: $choice) {
                    ForEach(plan.components) { item in
                        VStack(alignment: .leading) {
                            Text("\(item.title) — \(Format.bytes(item.bytes))")
                            Text(item.detail).font(.caption).foregroundStyle(.secondary)
                        }
                        .tag(item.id)
                    }
                }
                .pickerStyle(.radioGroup)
                if let conflict = selected?.servingConflict {
                    Label(conflict, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Type **\(info.id)** to confirm:")
                    TextField(info.id, text: $typed).textFieldStyle(.roundedBorder)
                }
            } else {
                ProgressView()
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Delete", role: .destructive) {
                    let title = selected?.title ?? choice
                    dismiss()
                    model.runOperation("Deleting \(title) of \(info.name)",
                                       ["delete", "--model", info.id, "--component", choice, "--confirm", typed])
                }
                .disabled(typed != info.id || selected == nil || selected?.servingConflict != nil)
            }
        }
        .task {
            plan = try? await model.backend?.api(["plan", "delete", "--model", info.id], as: DeletePlan.self)
            choice = plan?.components.first?.id ?? ""
        }
    }

    private var selected: DeleteComponent? { plan?.components.first { $0.id == choice } }
}

private struct AddModelSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(WindowRouter.self) private var router
    @Environment(\.dismiss) private var dismiss
    @State private var repo = ""
    @State private var generationConfig: URL?
    @State private var needsGenerationConfig: String?
    @State private var failure: String?
    @State private var running: OperationState?

    var body: some View {
        SheetFrame(title: "Add a model from HuggingFace") {
            TextField("Model ID, e.g. Qwen/Qwen3.6-35B-A3B", text: $repo)
                .textFieldStyle(.roundedBorder)
                .disabled(running?.isRunning == true)
            Text("Only the model's config files are downloaded now. You choose when to download the weights.")
                .font(.caption).foregroundStyle(.secondary)
            if let needsGenerationConfig {
                Text(needsGenerationConfig).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Text(generationConfig?.lastPathComponent ?? "No file chosen").foregroundStyle(.secondary)
                    Button("Choose Generation Settings…") {
                        generationConfig = Alerts.chooseFile("Choose a JSON file with the model's generation settings")
                    }
                }
            }
            if let failure {
                Text(failure).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            if let running, running.isRunning {
                ProgressView(running.message).progressViewStyle(.linear)
            }
            HStack {
                Spacer()
                Button("Cancel") {
                    model.cancelOperation()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Add") { add() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!repo.contains("/") || running?.isRunning == true
                              || (needsGenerationConfig != nil && generationConfig == nil))
            }
        }
    }

    private func add() {
        failure = nil
        var args = ["add-model", "--repo", repo.trimmingCharacters(in: .whitespaces)]
        if let generationConfig { args += ["--generation-config", generationConfig.path] }
        running = model.runOperation("Adding \(repo)", args, presentSheet: false) { done in
            if done.ok == true {
                router.selectedModel = done.model
                dismiss()
            } else if done.code == "needs_generation_config" {
                needsGenerationConfig = done.message
            } else {
                failure = done.message
            }
        }
    }
}

struct OperationSheet: View {
    let operation: OperationState
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(operation.title).font(.title3.bold())
                Spacer()
                if !operation.isRunning {
                    Image(systemName: operation.succeeded ? "checkmark.circle.fill" : "xmark.octagon.fill")
                        .foregroundStyle(operation.succeeded ? .green : .red)
                        .font(.title2)
                }
            }
            Text(operation.message).fixedSize(horizontal: false, vertical: true)
            if operation.isRunning {
                if let fraction = operation.fraction {
                    ProgressView(value: fraction) { Text(operation.phase) }
                } else {
                    ProgressView { Text(operation.phase) }.progressViewStyle(.linear)
                }
            }
            if !operation.artifacts.isEmpty {
                List(operation.artifacts) { row in
                    HStack {
                        Image(systemName: row.satisfied ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(row.satisfied ? .green : .red)
                        Text("\(row.scope)/\(row.relpath)").font(.callout.monospaced())
                        Spacer()
                        Text(row.state).font(.caption).foregroundStyle(.secondary)
                    }
                }
                .frame(height: 160)
            }
            if !operation.log.isEmpty {
                ScrollView {
                    Text(operation.log.suffix(400).joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 150)
                .padding(6)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
            }
            HStack {
                Spacer()
                if operation.isRunning {
                    Button("Cancel") { model.cancelOperation() }.disabled(operation.cancelRequested)
                    Button("Hide") { dismiss() }.keyboardShortcut(.defaultAction)
                } else {
                    Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(20)
        .frame(width: 560)
    }
}
