import Foundation

public struct CommandResult: Sendable {
    public var status: Int32
    public var stdout: String
    public var stderr: String

    public var succeeded: Bool { status == 0 }

    /// The last lines of combined output, for error alerts.
    public func tail(_ lines: Int = 12) -> String {
        let text = [stdout, stderr].filter { !$0.isEmpty }.joined(separator: "\n")
        return text.split(separator: "\n", omittingEmptySubsequences: true)
            .suffix(lines).joined(separator: "\n")
    }
}

public enum BackendError: LocalizedError, Sendable {
    case api(message: String, code: String)
    case command(String)
    case decoding(String)
    case setup(String)

    public var errorDescription: String? {
        switch self {
        case .api(let message, _): return message
        case .command(let output): return output.isEmpty ? "The command failed." : output
        case .decoding(let detail): return "Unexpected response from Flashchat: \(detail)"
        case .setup(let detail): return detail
        }
    }

    public var code: String? {
        if case .api(_, let code) = self { return code }
        return nil
    }
}

/// Where Flashchat lives and how to run its tools from a GUI process.
public struct FlashchatEnvironment: Sendable {
    public var repoRoot: URL

    public init(repoRoot: URL) {
        self.repoRoot = repoRoot
    }

    public var launcher: URL { repoRoot.appendingPathComponent("flashchat") }
    public var python: URL { repoRoot.appendingPathComponent("metal_infer/.venv/bin/python") }

    public var isValidRepo: Bool {
        FileManager.default.isExecutableFile(atPath: launcher.path)
            && FileManager.default.fileExists(
                atPath: repoRoot.appendingPathComponent("modelmgr/api.py").path)
    }

    public var pythonReady: Bool {
        FileManager.default.isExecutableFile(atPath: python.path)
    }

    /// GUI apps inherit a minimal PATH; the launcher needs Homebrew tools, make,
    /// nc, lsof and curl. FLASHCHAT_* overrides are dropped so the user's config
    /// file is the single source of truth, exactly as in a fresh terminal.
    public func processEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment.filter { !$0.key.hasPrefix("FLASHCHAT_") }
        let base = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        let existing = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        env["PATH"] = (base + existing.filter { !base.contains($0) }).joined(separator: ":")
        env["PYTHONPATH"] = repoRoot.path
        env["PYTHONUNBUFFERED"] = "1"
        env["TERM"] = "dumb"
        return env
    }
}

public final class RunningProcess: @unchecked Sendable {
    private let process: Process

    init(process: Process) {
        self.process = process
    }

    public var isRunning: Bool { process.isRunning }

    /// SIGINT: operations treat it like Ctrl-C in the terminal.
    public func cancel() {
        guard process.isRunning else { return }
        kill(process.processIdentifier, SIGINT)
    }
}

/// Collects a pipe's bytes off the main thread; splits complete lines on demand.
private final class PipeCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()
    private var lineBuffer = Data()
    private let onLine: (@Sendable (String) -> Void)?

    init(onLine: (@Sendable (String) -> Void)? = nil) {
        self.onLine = onLine
    }

    func append(_ chunk: Data) {
        var lines: [String] = []
        lock.lock()
        data.append(chunk)
        if onLine != nil {
            lineBuffer.append(chunk)
            while let newline = lineBuffer.firstIndex(of: 0x0A) {
                let line = lineBuffer[lineBuffer.startIndex..<newline]
                lines.append(String(decoding: line, as: UTF8.self))
                lineBuffer.removeSubrange(lineBuffer.startIndex...newline)
            }
        }
        lock.unlock()
        lines.forEach { onLine?($0) }
    }

    func finish() {
        lock.lock()
        let rest = lineBuffer
        lineBuffer = Data()
        lock.unlock()
        if !rest.isEmpty { onLine?(String(decoding: rest, as: UTF8.self)) }
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: data, as: UTF8.self)
    }
}

public enum CommandRunner {
    @discardableResult
    static func launch(executable: URL, arguments: [String], environment: [String: String],
                       directory: URL, stdoutLine: (@Sendable (String) -> Void)? = nil,
                       stderrLine: (@Sendable (String) -> Void)? = nil,
                       completion: @escaping @Sendable (CommandResult) -> Void) throws -> RunningProcess {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.currentDirectoryURL = directory
        process.standardInput = FileHandle.nullDevice
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        let outCollector = PipeCollector(onLine: stdoutLine)
        let errCollector = PipeCollector(onLine: stderrLine)
        let group = DispatchGroup()
        for (pipe, collector) in [(out, outCollector), (err, errCollector)] {
            group.enter()
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                if chunk.isEmpty {
                    handle.readabilityHandler = nil
                    collector.finish()
                    group.leave()
                } else {
                    collector.append(chunk)
                }
            }
        }
        process.terminationHandler = { proc in
            group.notify(queue: .global()) {
                completion(CommandResult(status: proc.terminationStatus,
                                         stdout: outCollector.text, stderr: errCollector.text))
            }
        }
        try process.run()
        return RunningProcess(process: process)
    }

    public static func run(executable: URL, arguments: [String], environment: [String: String],
                           directory: URL) async throws -> CommandResult {
        try await withCheckedThrowingContinuation { continuation in
            do {
                try launch(executable: executable, arguments: arguments,
                           environment: environment, directory: directory) { result in
                    continuation.resume(returning: result)
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

/// Typed access to the launcher and `modelmgr api`.
public struct Backend: Sendable {
    public var env: FlashchatEnvironment

    public init(env: FlashchatEnvironment) {
        self.env = env
    }

    public func launcher(_ arguments: [String]) async throws -> CommandResult {
        try await CommandRunner.run(executable: URL(fileURLWithPath: "/bin/bash"),
                                    arguments: [env.launcher.path] + arguments,
                                    environment: env.processEnvironment(),
                                    directory: env.repoRoot)
    }

    public func status() async throws -> LauncherStatus {
        let result = try await launcher(["status", "--json"])
        guard result.succeeded else { throw BackendError.command(result.tail()) }
        return try Self.decode(LauncherStatus.self, from: result.stdout)
    }

    public func api<T: Decodable>(_ arguments: [String], as type: T.Type) async throws -> T {
        guard env.pythonReady else {
            throw BackendError.setup("Flashchat's Python environment is not set up yet.")
        }
        let result = try await CommandRunner.run(
            executable: env.python, arguments: ["-m", "modelmgr", "api"] + arguments,
            environment: env.processEnvironment(), directory: env.repoRoot)
        let line = result.stdout.split(separator: "\n").last.map(String.init) ?? ""
        if !result.succeeded {
            if let body = try? Self.decode(ApiErrorBody.self, from: line) {
                throw BackendError.api(message: body.error, code: body.code)
            }
            throw BackendError.command(result.tail())
        }
        return try Self.decode(T.self, from: line)
    }

    /// Streams NDJSON events; `completion` receives the final `done` event (or a
    /// synthesized failure if the process died without one).
    public func operation(_ arguments: [String],
                          onEvent: @escaping @Sendable (OperationEvent) -> Void,
                          completion: @escaping @Sendable (OperationEvent, String) -> Void) throws -> RunningProcess {
        guard env.pythonReady else {
            throw BackendError.setup("Flashchat's Python environment is not set up yet.")
        }
        let doneBox = DoneBox()
        return try CommandRunner.launch(
            executable: env.python, arguments: ["-m", "modelmgr", "api", "run"] + arguments,
            environment: env.processEnvironment(), directory: env.repoRoot,
            stdoutLine: { line in
                guard let event = try? Self.decode(OperationEvent.self, from: line) else { return }
                if event.event == "done" { doneBox.set(event) } else { onEvent(event) }
            },
            completion: { result in
                let done = doneBox.get() ?? OperationEvent(
                    event: "done", message: result.tail().isEmpty
                        ? "The operation stopped unexpectedly." : result.tail(),
                    ok: false, code: "crashed")
                completion(done, result.stderr)
            })
    }

    static func decode<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
        do {
            return try JSONDecoder().decode(T.self, from: Data(text.utf8))
        } catch {
            throw BackendError.decoding(String(describing: error))
        }
    }
}

private final class DoneBox: @unchecked Sendable {
    private let lock = NSLock()
    private var event: OperationEvent?

    func set(_ value: OperationEvent) {
        lock.lock(); event = value; lock.unlock()
    }

    func get() -> OperationEvent? {
        lock.lock(); defer { lock.unlock() }
        return event
    }
}

extension OperationEvent {
    public init(event: String, message: String?, ok: Bool?, code: String?) {
        self.event = event
        self.message = message
        self.ok = ok
        self.code = code
    }
}
