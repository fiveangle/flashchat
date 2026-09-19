import Foundation

// Mirrors of the JSON produced by `modelmgr api` and `flashchat status --json`.
// Coding keys are explicit: config dictionaries carry keys like MAX_TOKENS that
// automatic snake-case conversion would mangle.

public struct ApiState: Decodable, Sendable {
    public var schema: Int
    public var repoRoot: String
    public var configDir: String
    public var configFile: String
    public var configExists: Bool
    public var migrationNeeded: Bool
    public var hfCacheDir: String
    public var offloadDir: String
    public var serverRunning: Bool
    public var config: [String: String]
    public var selected: SelectedModel?
    public var models: [ModelInfo]
    public var settings: [SettingDef]

    enum CodingKeys: String, CodingKey {
        case schema, config, selected, models, settings
        case repoRoot = "repo_root"
        case configDir = "config_dir"
        case configFile = "config_file"
        case configExists = "config_exists"
        case migrationNeeded = "migration_needed"
        case hfCacheDir = "hf_cache_dir"
        case offloadDir = "offload_dir"
        case serverRunning = "server_running"
    }

    public var selectedModel: ModelInfo? {
        guard let selected else { return nil }
        return models.first { $0.id == selected.model }
    }
}

public struct SelectedModel: Decodable, Sendable {
    public var model: String
    public var variant: String
    public var resolvedId: String
    public var ready: Bool
    public var memory: MemoryEstimate
    public var maxActiveExperts: Int
    public var supportsThinking: Bool?

    enum CodingKeys: String, CodingKey {
        case model, variant, ready, memory
        case resolvedId = "resolved_id"
        case maxActiveExperts = "max_active_experts"
        case supportsThinking = "supports_thinking"
    }
}

public struct MemoryEstimate: Decodable, Sendable {
    public var weightsBytes: Int64
    public var kvCacheBytes: Int64
    public var contextWindow: Int
    public var kvQuant: String
    public var expertCacheMaxBytes: Int64
    public var overheadBytes: Int64
    public var totalBytes: Int64

    enum CodingKeys: String, CodingKey {
        case weightsBytes = "weights_bytes"
        case kvCacheBytes = "kv_cache_bytes"
        case contextWindow = "context_window"
        case kvQuant = "kv_quant"
        case expertCacheMaxBytes = "expert_cache_max_bytes"
        case overheadBytes = "overhead_bytes"
        case totalBytes = "total_bytes"
    }
}

public struct SamplingProfile: Decodable, Sendable, Identifiable, Hashable {
    public var name: String
    public var label: String
    public var description: String
    public var values: [String: String]
    public var id: String { name }
}

public struct MissingArtifact: Decodable, Sendable, Hashable {
    public var relpath: String
    public var state: String
    public var detail: String
}

public struct VariantInfo: Decodable, Sendable, Identifiable, Hashable {
    public var name: String
    public var bits: Int
    public var resolvedId: String
    public var ready: Bool
    public var offloaded: Bool
    public var localBytes: Int64
    public var offloadBytes: Int64
    public var summary: String
    public var missing: [MissingArtifact]
    public var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, bits, ready, offloaded, summary, missing
        case resolvedId = "resolved_id"
        case localBytes = "local_bytes"
        case offloadBytes = "offload_bytes"
    }
}

public struct ModelInfo: Decodable, Sendable, Identifiable, Hashable {
    public var id: String
    public var name: String
    public var hfRepo: String
    public var enabled: Bool
    public var userDefined: Bool
    public var suggestedDefault: Bool
    public var selected: Bool
    public var selectedVariant: String?
    public var defaultVariant: String
    public var snapshot: String?
    public var originalsLocal: Bool
    public var originalsBytes: Int64
    public var originalsOffloaded: Bool
    public var archive: String
    public var offloadSnapshot: String?
    public var pendingOffloadSync: [String]
    public var maxContext: Int
    public var numExpertsPerTok: Int
    public var thinkingCapable: Bool
    public var mtpCapable: Bool
    public var defaultSamplingProfile: String
    public var samplingProfiles: [SamplingProfile]
    public var variants: [VariantInfo]

    enum CodingKeys: String, CodingKey {
        case id, name, enabled, selected, snapshot, archive, variants
        case hfRepo = "hf_repo"
        case userDefined = "user_defined"
        case suggestedDefault = "suggested_default"
        case selectedVariant = "selected_variant"
        case defaultVariant = "default_variant"
        case originalsLocal = "originals_local"
        case originalsBytes = "originals_bytes"
        case originalsOffloaded = "originals_offloaded"
        case offloadSnapshot = "offload_snapshot"
        case pendingOffloadSync = "pending_offload_sync"
        case maxContext = "max_context"
        case numExpertsPerTok = "num_experts_per_tok"
        case thinkingCapable = "thinking_capable"
        case mtpCapable = "mtp_capable"
        case defaultSamplingProfile = "default_sampling_profile"
        case samplingProfiles = "sampling_profiles"
    }

    public var isLocal: Bool { snapshot != nil }
    public var anyReady: Bool { variants.contains { $0.ready } }
}

public struct SettingDef: Decodable, Sendable, Identifiable, Hashable {
    public var key: String
    public var section: String
    public var title: String
    public var label: String
    public var help: String?
    public var kind: String
    public var choices: [String]
    public var minimum: Double?
    public var maximum: Double?
    public var clearWord: String?
    public var emptyTitle: String?
    public var parent: String?
    public var id: String { key }

    enum CodingKeys: String, CodingKey {
        case key, section, title, label, help, kind, choices, minimum, maximum, parent
        case clearWord = "clear_word"
        case emptyTitle = "empty_title"
    }

    public var allowsEmpty: Bool {
        clearWord != nil || emptyTitle != nil || ["text", "path", "mtp"].contains(kind)
    }
}

public struct ArtifactRow: Decodable, Sendable, Identifiable, Hashable {
    public var scope: String
    public var relpath: String
    public var state: String
    public var detail: String
    public var required: Bool
    public var satisfied: Bool
    public var id: String { "\(scope)/\(relpath)" }

    public init(scope: String, relpath: String, state: String, detail: String, required: Bool,
                satisfied: Bool) {
        self.scope = scope
        self.relpath = relpath
        self.state = state
        self.detail = detail
        self.required = required
        self.satisfied = satisfied
    }
}

public struct ModelDetail: Decodable, Sendable {
    public var model: ModelInfo
    public var artifacts: [ArtifactRow]
}

public struct RestoreInfo: Decodable, Sendable, Hashable {
    public var available: Bool
    public var neededBytes: Int64
    public var freeBytes: Int64
    public var fits: Bool
    public var shortfallBytes: Int64

    enum CodingKeys: String, CodingKey {
        case available, fits
        case neededBytes = "needed_bytes"
        case freeBytes = "free_bytes"
        case shortfallBytes = "shortfall_bytes"
    }
}

public struct BuildSource: Decodable, Sendable, Identifiable, Hashable {
    public var id: String
    public var title: String
    /// Where the files come from or land; shown under the picker, not in the row.
    public var detail: String?
    public var restore: RestoreInfo?
}

public struct PlanStep: Decodable, Sendable, Identifiable, Hashable {
    public var step: String
    public var scope: String
    public var artifacts: [String]
    public var reason: String
    public var detail: String
    public var description: String
    public var bytes: Int64
    public var id: String { "\(scope):\(step)" }
}

public struct RepairInfo: Decodable, Sendable, Hashable {
    public var create: [String]
    public var rebuild: [String]
}

public struct BuildPlan: Decodable, Sendable {
    public var model: String
    public var variant: String
    public var ready: Bool
    public var localSnapshot: String?
    public var needsSource: Bool
    public var sources: [BuildSource]
    public var runtimeRestore: RestoreInfo?
    public var steps: [PlanStep]
    public var totalBytes: Int64
    public var freeBytes: Int64
    public var repair: RepairInfo?
    public var servingConflict: String?

    enum CodingKeys: String, CodingKey {
        case model, variant, ready, sources, steps, repair
        case localSnapshot = "local_snapshot"
        case needsSource = "needs_source"
        case runtimeRestore = "runtime_restore"
        case totalBytes = "total_bytes"
        case freeBytes = "free_bytes"
        case servingConflict = "serving_conflict"
    }
}

public struct RestoreOption: Decodable, Sendable, Identifiable, Hashable {
    public var id: String
    public var title: String
    public var available: Bool
    public var neededBytes: Int64
    public var freeBytes: Int64
    public var fits: Bool
    public var shortfallBytes: Int64

    enum CodingKeys: String, CodingKey {
        case id, title, available, fits
        case neededBytes = "needed_bytes"
        case freeBytes = "free_bytes"
        case shortfallBytes = "shortfall_bytes"
    }
}

public struct RestorePlan: Decodable, Sendable {
    public var model: String
    public var available: Bool
    public var offloadDir: String?
    public var options: [RestoreOption]

    enum CodingKeys: String, CodingKey {
        case model, available, options
        case offloadDir = "offload_dir"
    }
}

public struct OffloadPlan: Decodable, Sendable {
    public var model: String
    public var offloadDir: String
    public var originalsBytes: Int64
    public var anyReady: Bool
    public var heldBack: [String: String]
    public var errors: [String]
    public var warnings: [String]
    public var available: Bool

    enum CodingKeys: String, CodingKey {
        case model, errors, warnings, available
        case offloadDir = "offload_dir"
        case originalsBytes = "originals_bytes"
        case anyReady = "any_ready"
        case heldBack = "held_back"
    }
}

public struct DeleteComponent: Decodable, Sendable, Identifiable, Hashable {
    public var id: String
    public var title: String
    public var detail: String
    public var bytes: Int64
    public var servingConflict: String?

    enum CodingKeys: String, CodingKey {
        case id, title, detail, bytes
        case servingConflict = "serving_conflict"
    }
}

public struct DeletePlan: Decodable, Sendable {
    public var model: String
    public var archive: String
    public var components: [DeleteComponent]
}

public struct SetResult: Decodable, Sendable {
    public var ok: Bool
    public var changed: Bool
    public var values: [String: String]
    public var warnings: [String]
    public var serverRunning: Bool

    enum CodingKeys: String, CodingKey {
        case ok, changed, values, warnings
        case serverRunning = "server_running"
    }
}

public struct SelectResult: Decodable, Sendable {
    public var ok: Bool
    public var ready: Bool
    public var resolvedId: String
    public var warnings: [String]
    public var serverRunning: Bool

    enum CodingKeys: String, CodingKey {
        case ok, ready, warnings
        case resolvedId = "resolved_id"
        case serverRunning = "server_running"
    }
}

public struct OkResult: Decodable, Sendable {
    public var ok: Bool
}

public struct PreflightResult: Decodable, Sendable {
    public var ok: Bool
    public var freeBytes: Int64
    public var symlinks: Bool
    public var errors: [String]
    public var warnings: [String]

    enum CodingKeys: String, CodingKey {
        case ok, symlinks, errors, warnings
        case freeBytes = "free_bytes"
    }
}

public struct ApiErrorBody: Decodable, Sendable {
    public var ok: Bool
    public var error: String
    public var code: String
}

/// One NDJSON line from `modelmgr api run`.
public struct OperationEvent: Decodable, Sendable {
    public var event: String
    public var phase: String?
    public var current: Int64?
    public var total: Int64?
    public var message: String?
    public var ok: Bool?
    public var code: String?
    public var scope: String?
    public var relpath: String?
    public var state: String?
    public var detail: String?
    public var corrupt: Int?
    public var unhashed: Int?
    public var offloadSuggested: Bool?
    public var model: String?

    enum CodingKeys: String, CodingKey {
        case event, phase, current, total, message, ok, code, scope, relpath, state
        case detail, corrupt, unhashed, model
        case offloadSuggested = "offload_suggested"
    }

    public var fraction: Double? {
        guard let current, let total, total > 0 else { return nil }
        return min(1, max(0, Double(current) / Double(total)))
    }
}

// MARK: - Launcher and engine

public struct LauncherStatus: Decodable, Sendable, Equatable {
    public struct Server: Decodable, Sendable, Equatable {
        public var state: String
        public var pid: Int?
        public var port: Int
        public var host: String
        public var bind: String
        public var url: String
        public var log: String
        public var pidFile: String

        enum CodingKeys: String, CodingKey {
            case state, pid, port, host, bind, url, log
            case pidFile = "pid_file"
        }
    }

    public var schema: Int
    public var server: Server
    public var model: String
    public var configFile: String
    public var binariesCurrent: Bool
    /// Files that feed the launcher's restart-needed signature. Watching their
    /// timestamps tells a client when `status --json` is worth re-running.
    public var watch: [String]?

    enum CodingKeys: String, CodingKey {
        case schema, server, model, watch
        case configFile = "config_file"
        case binariesCurrent = "binaries_current"
    }
}

/// `GET /health` from the inference server.
public struct Health: Decodable, Sendable, Equatable {
    public var status: String
    public var ready: Bool
    public var model: String
    public var contextUsed: Int
    public var maxContext: Int
    public var kvQuantization: String
    public var kvTotalBytes: Int64
    public var phase: String
    public var cachedTokens: Int
    public var promptTokens: Int
    public var prefillDone: Int
    public var generatedTokens: Int
    public var chunk: Int
    public var chunks: Int
    public var layer: Int
    public var layers: Int

    enum CodingKeys: String, CodingKey {
        case status, ready, model, phase, chunk, chunks, layer, layers
        case contextUsed = "context_used"
        case maxContext = "max_context"
        case kvQuantization = "kv_quantization"
        case kvTotalBytes = "kv_total_bytes"
        case cachedTokens = "cached_tokens"
        case promptTokens = "prompt_tokens"
        case prefillDone = "prefill_done"
        case generatedTokens = "generated_tokens"
    }
}
