import Foundation

/// What the app is doing to the server right now, on top of what it observes.
public enum ServerTransition: Equatable, Sendable {
    case starting
    case stopping
    case restarting
}

public enum ServerActivity: Equatable, Sendable {
    case stopped
    case starting
    case stopping
    case restartNeeded
    case external
    case unreachable
    case idle
    case preparing
    case prefill(done: Int, total: Int, layer: Int, layers: Int)
    case generating(tokens: Int)
}

/// Everything the menubar icon and popover header need, derived in one place.
public struct ServerDisplay: Equatable, Sendable {
    public var activity: ServerActivity
    public var title: String
    public var detail: String?
    public var symbol: String
    public var progress: Double?

    public var isRunning: Bool {
        switch activity {
        case .stopped, .starting: return false
        default: return true
        }
    }

    public var isBusy: Bool {
        switch activity {
        case .preparing, .prefill, .generating: return true
        default: return false
        }
    }

    public var isOwned: Bool { activity != .external && activity != .stopped }

    public static func derive(status: LauncherStatus?, health: Health?,
                              transition: ServerTransition?, tokensPerSecond: Double?) -> ServerDisplay {
        switch transition {
        case .starting:
            return ServerDisplay(activity: .starting, title: "Starting…",
                                 detail: "Loading the model", symbol: "bolt.badge.clock", progress: nil)
        case .stopping:
            return ServerDisplay(activity: .stopping, title: "Stopping…", detail: nil,
                                 symbol: "bolt.badge.clock", progress: nil)
        case .restarting:
            return ServerDisplay(activity: .starting, title: "Restarting…",
                                 detail: "Applying the new configuration",
                                 symbol: "bolt.badge.clock", progress: nil)
        case nil:
            break
        }

        let state = status?.server.state ?? (health != nil ? "running" : "stopped")
        if state == "stopped" && health == nil {
            return ServerDisplay(activity: .stopped, title: "Server stopped", detail: nil,
                                 symbol: "bolt.slash", progress: nil)
        }
        guard let health else {
            return ServerDisplay(activity: .unreachable, title: "Server not responding",
                                 detail: "The process is running but /health did not answer",
                                 symbol: "exclamationmark.triangle", progress: nil)
        }

        var display: ServerDisplay
        switch health.phase {
        case "preparing":
            display = ServerDisplay(activity: .preparing, title: "Preparing prompt", detail: nil,
                                    symbol: "bolt.fill", progress: nil)
        case "prefill":
            let fraction = health.promptTokens > 0
                ? Double(health.prefillDone) / Double(health.promptTokens) : nil
            var detail = health.promptTokens > 0
                ? "\(Format.count(health.prefillDone)) / \(Format.count(health.promptTokens)) tokens"
                : "Reading prompt"
            if health.layers > 0 { detail += " · layer \(health.layer)/\(health.layers)" }
            let percent = fraction.map { " \(Int(($0 * 100).rounded()))%" } ?? ""
            display = ServerDisplay(
                activity: .prefill(done: health.prefillDone, total: health.promptTokens,
                                   layer: health.layer, layers: health.layers),
                title: "Reading prompt\(percent)", detail: detail, symbol: "bolt.fill",
                progress: fraction)
        case "generating":
            var detail = "\(Format.count(health.generatedTokens)) tokens"
            if let tps = tokensPerSecond { detail = "\(Format.rate(tps)) · " + detail }
            display = ServerDisplay(activity: .generating(tokens: health.generatedTokens),
                                    title: "Generating", detail: detail, symbol: "bolt.fill",
                                    progress: nil)
        default:
            display = ServerDisplay(activity: .idle, title: "Ready", detail: "Idle — waiting for a request",
                                    symbol: "bolt", progress: nil)
        }

        if state == "stale" && !display.isBusy {
            display.activity = .restartNeeded
            display.title = "Restart needed"
            display.detail = "Settings changed since the server started"
            display.symbol = "bolt.trianglebadge.exclamationmark"
        } else if state == "external" && !display.isBusy {
            display.activity = .external
            display.title = "Ready (external)"
            display.detail = "Started outside Flashchat's control"
            display.symbol = "bolt.horizontal"
        }
        return display
    }
}

/// Decode speed from successive /health samples while generating.
public struct ThroughputMeter: Sendable {
    private var samples: [(time: TimeInterval, tokens: Int)] = []
    private let window: TimeInterval

    public init(window: TimeInterval = 4) {
        self.window = window
    }

    public mutating func record(health: Health?, at time: TimeInterval) {
        guard let health, health.phase == "generating" else {
            samples.removeAll()
            return
        }
        if let last = samples.last, health.generatedTokens < last.tokens {
            samples.removeAll()
        }
        samples.append((time, health.generatedTokens))
        samples.removeAll { time - $0.time > window }
    }

    public var tokensPerSecond: Double? {
        guard let first = samples.first, let last = samples.last,
              last.time - first.time >= 0.5, last.tokens > first.tokens else { return nil }
        return Double(last.tokens - first.tokens) / (last.time - first.time)
    }
}

public enum Format {
    public static func bytes(_ value: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        formatter.allowedUnits = [.useMB, .useGB, .useTB]
        return formatter.string(fromByteCount: value)
    }

    public static func count(_ value: Int) -> String {
        value.formatted(.number)
    }

    public static func rate(_ tps: Double) -> String {
        String(format: tps >= 10 ? "%.0f tok/s" : "%.1f tok/s", tps)
    }

    public static func tokensShort(_ value: Int) -> String {
        value >= 1024 ? "\((value + 512) / 1024)K" : "\(value)"
    }
}

/// The engine's log-path rule: a directory (existing, trailing slash, or no
/// extension) holds server.log; anything else is the log file itself.
public func resolveServerLog(_ configured: String) -> URL? {
    guard !configured.isEmpty else { return nil }
    let expanded = (configured as NSString).expandingTildeInPath
    var isDir: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir)
    let treatAsDir = exists
        ? isDir.boolValue
        : (expanded.hasSuffix("/") || (expanded as NSString).pathExtension.isEmpty)
    let url = URL(fileURLWithPath: expanded)
    return treatAsDir ? url.appendingPathComponent("server.log") : url
}
