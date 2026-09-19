import AppKit
import FlashchatKit
import SwiftUI

/// Rendered entirely from the settings schema `modelmgr api state` returns, so
/// a setting added to modelmgr/settings.py appears here without app changes.
struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var draft: [String: String] = [:]
    @State private var baseline: [String: String] = [:]
    @State private var warnings: [String] = []
    @State private var saving = false
    @State private var showAdvanced = false

    private static let sections: [(id: String, title: String)] = [
        ("generation", "Generation"), ("sampling", "Sampling"), ("server", "Server"),
        ("storage", "Storage"), ("advanced", "Advanced"),
    ]

    var body: some View {
        Group {
            if let state = model.apiState, state.configExists {
                form(state)
            } else if model.apiState != nil {
                ContentUnavailableView("No configuration yet", systemImage: "slider.horizontal.3",
                                       description: Text("Choose a model under Models to create your configuration."))
            } else {
                ProgressView()
            }
        }
        .navigationTitle("Settings")
        .onAppear { resetDraft() }
        .onChange(of: model.apiState?.config) { resetDraft(keepEdits: true) }
    }

    private var changes: [String: String] {
        draft.filter { baseline[$0.key] != $0.value }
    }

    @ViewBuilder
    private func form(_ state: ApiState) -> some View {
        VStack(spacing: 0) {
            Form {
                if let selected = state.selected {
                    Section {
                        LabeledContent("Model") {
                            Text("\(state.selectedModel?.name ?? selected.model) [\(selected.variant)]")
                        }
                        if let memory = model.apiState?.selected?.memory {
                            LabeledContent("Context cache RAM",
                                           value: "\(Format.bytes(memory.kvCacheBytes)) at \(Format.tokensShort(memory.contextWindow)) tokens (\(memory.kvQuant))")
                        }
                    }
                }
                ForEach(Self.sections, id: \.id) { section in
                    let defs = visibleSettings(state, section: section.id)
                    if section.id == "advanced" {
                        // A plain header button, not DisclosureGroup: inside a
                        // grouped Form the disclosure never expanded.
                        Section {
                            if showAdvanced {
                                ForEach(defs) { def in row(def, state: state) }
                            }
                        } header: {
                            Button {
                                withAnimation { showAdvanced.toggle() }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "chevron.right")
                                        .rotationEffect(.degrees(showAdvanced ? 90 : 0))
                                        .font(.caption.weight(.semibold))
                                    Text("Advanced options (performance, debugging, caches)")
                                    Spacer()
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    } else if !defs.isEmpty {
                        Section(section.title) {
                            ForEach(defs) { def in row(def, state: state) }
                        }
                    }
                }
                if !warnings.isEmpty {
                    Section("Notes") {
                        ForEach(warnings, id: \.self) { Label($0, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.orange) }
                    }
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack {
                if !changes.isEmpty {
                    Text("\(changes.count) unsaved change\(changes.count == 1 ? "" : "s")")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Revert") { resetDraft() }.disabled(changes.isEmpty || saving)
                Button("Save") { save() }
                    .keyboardShortcut("s", modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .disabled(changes.isEmpty || saving)
            }
            .padding(10)
        }
    }

    private func visibleSettings(_ state: ApiState, section: String) -> [SettingDef] {
        state.settings.filter { def in
            guard def.section == section else { return false }
            if def.key == "ACTIVE_EXPERTS" { return (state.selectedModel?.numExpertsPerTok ?? 0) > 0 }
            return true
        }
    }

    private var profileLocked: Bool {
        let profile = draft["SAMPLING_PROFILE"] ?? ""
        return !profile.isEmpty && profile != "custom"
    }

    private func parentEnabled(_ def: SettingDef, state: ApiState) -> Bool {
        guard let parent = def.parent,
              let parentDef = state.settings.first(where: { $0.key == parent }),
              parentDef.kind == "bool" else { return true }
        return (draft[parent] ?? "") == "1"
    }

    @ViewBuilder
    private func row(_ def: SettingDef, state: ApiState) -> some View {
        let locked = def.section == "sampling" && def.key != "ACTIVE_EXPERTS" && profileLocked
        SettingRow(def: def, value: binding(def.key), profiles: state.selectedModel?.samplingProfiles ?? [],
                   modelDefaultK: state.selectedModel?.numExpertsPerTok ?? 0,
                   maxK: state.selected?.maxActiveExperts ?? 16,
                   onProfile: applyProfile)
            .disabled(locked || !parentEnabled(def, state: state))
            .padding(.leading, def.parent == nil ? 0 : 16)
            .help(locked ? "Set by the sampling profile. Choose “custom” to edit." : (def.help ?? ""))
    }

    private func binding(_ key: String) -> Binding<String> {
        Binding(get: { draft[key] ?? "" }, set: { draft[key] = $0 })
    }

    private func applyProfile(_ name: String) {
        guard let profile = model.apiState?.selectedModel?.samplingProfiles.first(where: { $0.name == name })
        else { return }
        for (key, value) in profile.values { draft[key] = value }
    }

    private func resetDraft(keepEdits: Bool = false) {
        let config = model.apiState?.config ?? [:]
        let edits = keepEdits ? changes : [:]
        baseline = config
        draft = config.merging(edits) { _, edit in edit }
    }

    private func save() {
        saving = true
        let pending = changes
        Task {
            let result = await model.saveSettings(pending)
            saving = false
            if let result {
                warnings = result.warnings
                resetDraft()
            }
        }
    }
}

private struct SettingRow: View {
    let def: SettingDef
    @Binding var value: String
    let profiles: [SamplingProfile]
    let modelDefaultK: Int
    let maxK: Int
    let onProfile: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            control
            if let help = def.help {
                Text(help).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let problem = validation {
                Text(problem).font(.caption).foregroundStyle(.red)
            }
        }
    }

    @ViewBuilder
    private var control: some View {
        switch def.kind {
        case "bool":
            Toggle(def.title, isOn: Binding(get: { value == "1" }, set: { value = $0 ? "1" : "0" }))
        case "choice" where def.key == "SAMPLING_PROFILE":
            Picker(def.title, selection: Binding(get: { value }, set: { value = $0; onProfile($0) })) {
                ForEach(profiles) { Text($0.label).tag($0.name) }
                Text("Custom — set each value yourself").tag("custom")
                if !value.isEmpty && value != "custom" && !profiles.contains(where: { $0.name == value }) {
                    Text(value).tag(value)
                }
            }
        case "choice":
            Picker(def.title, selection: $value) {
                ForEach(def.choices, id: \.self) { Text(choiceTitle($0)).tag($0) }
                if !def.choices.contains(value) { Text(value.isEmpty ? "Default" : value).tag(value) }
            }
        case "mtp":
            MTPControl(title: def.title, value: $value)
        case "path":
            HStack {
                TextField(def.title, text: $value, prompt: Text(def.emptyTitle ?? ""))
                Button("Choose…") {
                    if def.key == "PREAD_PROFILE" || def.key == "SERVER_LOG_PATH" {
                        let panel = NSSavePanel()
                        panel.nameFieldStringValue = (value as NSString).lastPathComponent
                        if panel.runModal() == .OK, let url = panel.url { value = url.path }
                    } else if let url = Alerts.chooseFolder("Choose a folder for \(def.title.lowercased())",
                                                            start: value.isEmpty ? nil : value) {
                        value = url.path
                    }
                }
            }
        default:
            TextField(def.title, text: $value, prompt: Text(placeholder))
        }
    }

    private var placeholder: String {
        if def.key == "ACTIVE_EXPERTS", modelDefaultK > 0 { return "Model default (\(modelDefaultK)), max \(maxK)" }
        if let empty = def.emptyTitle { return empty }
        if let min = def.minimum, let max = def.maximum { return "\(min.clean)–\(max.clean)" }
        return ""
    }

    private func choiceTitle(_ choice: String) -> String {
        switch (def.key, choice) {
        case ("KV_QUANT", "off"): return "Off (fp32, lossless)"
        case ("KV_QUANT", "q8"): return "q8 (~lossless, best for large windows)"
        case ("KV_QUANT", "q4"): return "q4 (lossy, smallest)"
        case ("PREFILL_DEBUG", "0"): return "Off"
        case ("PREFILL_DEBUG", "1"): return "Chunk timings"
        case ("PREFILL_DEBUG", "2"): return "Timings + state dump (slow)"
        default: return choice
        }
    }

    /// Client-side hint only; modelmgr validates authoritatively on save.
    private var validation: String? {
        let text = value.trimmingCharacters(in: .whitespaces)
        if text.isEmpty {
            return def.allowsEmpty ? nil : "\(def.title) needs a value"
        }
        if let clear = def.clearWord, text.lowercased() == clear.lowercased() { return nil }
        guard def.kind == "int" || def.kind == "float" else { return nil }
        guard let number = Double(text), def.kind == "float" || Int(text) != nil else {
            return def.kind == "int" ? "Enter a whole number" : "Enter a number"
        }
        if let min = def.minimum, number < min { return "Must be at least \(min.clean)" }
        if let max = def.maximum, number > max { return "Must be at most \(max.clean)" }
        return nil
    }
}

private struct MTPControl: View {
    let title: String
    @Binding var value: String

    private var mode: String {
        switch value {
        case "": return "default"
        case "0": return "off"
        default: return "custom"
        }
    }

    var body: some View {
        HStack {
            Picker(title, selection: Binding(get: { mode }, set: { newMode in
                switch newMode {
                case "default": value = ""
                case "off": value = "0"
                default: value = (Int(value) ?? 0) >= 2 ? value : "2"
                }
            })) {
                Text("Model default").tag("default")
                Text("Off").tag("off")
                Text("Batch size…").tag("custom")
            }
            if mode == "custom" {
                Stepper(value: Binding(get: { Int(value) ?? 2 }, set: { value = String($0) }), in: 1...8) {
                    Text(value).monospacedDigit()
                }
                .fixedSize()
            }
        }
    }
}

private extension Double {
    var clean: String {
        truncatingRemainder(dividingBy: 1) == 0 ? String(Int(self)) : String(self)
    }
}
