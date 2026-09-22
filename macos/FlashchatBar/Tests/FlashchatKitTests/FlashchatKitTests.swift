import XCTest
@testable import FlashchatKit

final class DecodingTests: XCTestCase {
    private func fixture(_ name: String) throws -> String {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil))
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testStateDecodesWithUnmangledConfigKeys() throws {
        let state = try Backend.decode(ApiState.self, from: fixture("state.json"))
        XCTAssertEqual(state.schema, 1)
        XCTAssertNotNil(state.config["MAX_TOKENS"])
        XCTAssertNotNil(state.config["SERVER_PORT"])
        XCTAssertEqual(state.selected?.variant, "q4")
        XCTAssertEqual(state.selectedModel?.id, state.selected?.model)
        XCTAssertTrue(state.settings.contains { $0.key == "KV_QUANT" && $0.choices == ["off", "q8", "q4"] })
        XCTAssertGreaterThan(state.selected?.memory.totalBytes ?? 0, 0)
    }

    func testPlansDecode() throws {
        let build = try Backend.decode(BuildPlan.self, from: fixture("plan_build.json"))
        XCTAssertFalse(build.ready)
        XCTAssertFalse(build.steps.isEmpty)
        let delete = try Backend.decode(DeletePlan.self, from: fixture("plan_delete.json"))
        XCTAssertTrue(delete.components.contains { $0.id == "model" })
        _ = try Backend.decode(RestorePlan.self, from: fixture("plan_restore.json"))
        let offload = try Backend.decode(OffloadPlan.self, from: fixture("plan_offload.json"))
        XCTAssertFalse(offload.available)
        let detail = try Backend.decode(ModelDetail.self, from: fixture("model.json"))
        XCTAssertFalse(detail.artifacts.isEmpty)
    }

    func testPredictorToggleMapsToExistingBF16Preference() throws {
        let json = #"{"key":"MTP_BF16","section":"advanced","title":"Use reduced-precision predictor weights","label":"","kind":"bool","choices":[],"inverted":true}"#
        let def = try Backend.decode(SettingDef.self, from: json)
        XCTAssertTrue(def.toggleIsOn("0"))
        XCTAssertFalse(def.toggleIsOn("1"))
        XCTAssertEqual(def.toggleValue(isOn: true), "0")
        XCTAssertEqual(def.toggleValue(isOn: false), "1")

        let state = try Backend.decode(ApiState.self, from: fixture("state.json"))
        let ordinary = try XCTUnwrap(state.settings.first { $0.key == "SERVER_DEBUG" })
        XCTAssertNil(ordinary.inverted)
        XCTAssertTrue(ordinary.toggleIsOn("1"))
        XCTAssertFalse(ordinary.toggleIsOn("0"))
        XCTAssertEqual(ordinary.toggleValue(isOn: true), "1")
        XCTAssertEqual(ordinary.toggleValue(isOn: false), "0")
    }

    func testLauncherStatusAndHealthDecode() throws {
        let status = try Backend.decode(LauncherStatus.self, from: fixture("status.json"))
        XCTAssertEqual(status.server.state, "stopped")
        XCTAssertNil(status.server.pid)
        let watch = try XCTUnwrap(status.watch)
        XCTAssertTrue(watch.contains { $0.hasSuffix("/config") })
        XCTAssertTrue(watch.contains { $0.hasSuffix("shaders.metal") })
        XCTAssertTrue(watch.contains { $0.hasSuffix("lib/config.sh") })
        let health = try Backend.decode(Health.self, from: fixture("health_prefill.json"))
        XCTAssertEqual(health.phase, "prefill")
        XCTAssertEqual(health.layers, 40)
    }

    func testOperationEventsDecode() throws {
        let events = try fixture("verify_events.ndjson").split(separator: "\n").map {
            try Backend.decode(OperationEvent.self, from: String($0))
        }
        XCTAssertEqual(events.last?.event, "done")
        XCTAssertEqual(events.last?.ok, true)
        XCTAssertTrue(events.contains { $0.event == "artifact" && $0.relpath != nil })
    }
}

final class ServerDisplayTests: XCTestCase {
    private func status(_ state: String) -> LauncherStatus {
        LauncherStatus(schema: 1, server: .init(state: state, pid: 42, port: 8000, host: "127.0.0.1",
                                                bind: "127.0.0.1", url: "", log: "", pidFile: ""),
                       model: "m", configFile: "", binariesCurrent: true)
    }

    private func health(_ phase: String, prefill: Int = 0, prompt: Int = 0, generated: Int = 0) -> Health {
        Health(status: "ok", ready: true, model: "m", contextUsed: 0, maxContext: 65536,
               kvQuantization: "q8", kvTotalBytes: 0, phase: phase, cachedTokens: 0,
               promptTokens: prompt, prefillDone: prefill, generatedTokens: generated,
               chunk: 0, chunks: 0, layer: 3, layers: 40)
    }

    func testStoppedWhenNoServer() {
        let d = ServerDisplay.derive(status: status("stopped"), health: nil, transition: nil,
                                     tokensPerSecond: nil)
        XCTAssertEqual(d.activity, .stopped)
        XCTAssertFalse(d.isRunning)
    }

    func testTransitionWins() {
        let d = ServerDisplay.derive(status: status("running"), health: health("idle"),
                                     transition: .stopping, tokensPerSecond: nil)
        XCTAssertEqual(d.activity, .stopping)
    }

    func testPrefillProgress() {
        let d = ServerDisplay.derive(status: status("running"),
                                     health: health("prefill", prefill: 1000, prompt: 4000),
                                     transition: nil, tokensPerSecond: nil)
        XCTAssertEqual(d.progress ?? 0, 0.25, accuracy: 0.001)
        XCTAssertEqual(d.title, "Reading prompt 25%")
        XCTAssertTrue(d.isBusy)
    }

    func testStaleOnlyShownWhenIdle() {
        let idle = ServerDisplay.derive(status: status("stale"), health: health("idle"),
                                        transition: nil, tokensPerSecond: nil)
        XCTAssertEqual(idle.activity, .restartNeeded)
        let busy = ServerDisplay.derive(status: status("stale"), health: health("generating", generated: 5),
                                        transition: nil, tokensPerSecond: 12)
        XCTAssertEqual(busy.activity, .generating(tokens: 5))
        XCTAssertTrue(busy.detail?.contains("12 tok/s") ?? false)
    }

    func testRunningWithoutHealthIsUnreachable() {
        let d = ServerDisplay.derive(status: status("running"), health: nil, transition: nil,
                                     tokensPerSecond: nil)
        XCTAssertEqual(d.activity, .unreachable)
    }
}

final class ThroughputTests: XCTestCase {
    private func generating(_ tokens: Int) -> Health {
        Health(status: "ok", ready: true, model: "m", contextUsed: 0, maxContext: 1, kvQuantization: "",
               kvTotalBytes: 0, phase: "generating", cachedTokens: 0, promptTokens: 0, prefillDone: 0,
               generatedTokens: tokens, chunk: 0, chunks: 0, layer: 0, layers: 0)
    }

    func testRateFromSamples() {
        var meter = ThroughputMeter()
        meter.record(health: generating(0), at: 0)
        meter.record(health: generating(10), at: 1)
        meter.record(health: generating(20), at: 2)
        XCTAssertEqual(meter.tokensPerSecond ?? 0, 10, accuracy: 0.01)
    }

    func testResetsWhenGenerationRestarts() {
        var meter = ThroughputMeter()
        meter.record(health: generating(100), at: 0)
        meter.record(health: generating(120), at: 1)
        meter.record(health: generating(2), at: 2)
        XCTAssertNil(meter.tokensPerSecond)
        meter.record(health: nil, at: 3)
        XCTAssertNil(meter.tokensPerSecond)
    }
}

final class MemoryPreflightTests: XCTestCase {
    private let gib: Int64 = 1 << 30

    func testComfortableIsOk() {
        let mem = SystemMemorySnapshot(totalBytes: 32 * gib, availableBytes: 20 * gib,
                                       swapUsedBytes: 0, pressure: .normal)
        XCTAssertEqual(MemoryPreflight.evaluate(estimateBytes: 4 * gib, memory: mem), .ok)
    }

    func testTightAndInsufficient() {
        let mem = SystemMemorySnapshot(totalBytes: 16 * gib, availableBytes: 7 * gib,
                                       swapUsedBytes: 0, pressure: .normal)
        if case .tight = MemoryPreflight.evaluate(estimateBytes: 4 * gib, memory: mem) {} else {
            XCTFail("expected tight")
        }
        if case .insufficient = MemoryPreflight.evaluate(estimateBytes: 6 * gib, memory: mem) {} else {
            XCTFail("expected insufficient")
        }
    }

    func testCriticalPressureBlocks() {
        let mem = SystemMemorySnapshot(totalBytes: 32 * gib, availableBytes: 20 * gib,
                                       swapUsedBytes: 0, pressure: .critical)
        if case .insufficient = MemoryPreflight.evaluate(estimateBytes: gib, memory: mem) {} else {
            XCTFail("expected insufficient")
        }
    }

    func testProcessMemoryReadsOwnProcess() throws {
        let own = try XCTUnwrap(ProcessMemory.read(pid: getpid()))
        XCTAssertGreaterThan(own.footprintBytes, 0)
        XCTAssertGreaterThan(own.residentBytes, 0)
        XCTAssertNil(ProcessMemory.read(pid: 0))
        XCTAssertNil(ProcessMemory.read(pid: 999_999))
    }

    func testLiveSnapshotIsPlausible() {
        let mem = SystemMemorySnapshot.current()
        XCTAssertGreaterThan(mem.totalBytes, 0)
        XCTAssertGreaterThan(mem.availableBytes, 0)
        XCTAssertLessThanOrEqual(mem.availableBytes, mem.totalBytes)
    }
}

final class LogPathTests: XCTestCase {
    func testDirectoryGetsServerLog() {
        XCTAssertEqual(resolveServerLog("/tmp/fc-logs/")?.lastPathComponent, "server.log")
        XCTAssertEqual(resolveServerLog("/tmp/nonexistent-dir-xyz")?.lastPathComponent, "server.log")
        XCTAssertEqual(resolveServerLog("/tmp/custom.log")?.lastPathComponent, "custom.log")
        XCTAssertNil(resolveServerLog(""))
    }
}

final class BackendIntegrationTests: XCTestCase {
    /// Runs the real launcher from this checkout (read-only command).
    func testLauncherStatusJSON() async throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let env = FlashchatEnvironment(repoRoot: repo)
        try XCTSkipUnless(env.isValidRepo)
        let status = try await Backend(env: env).status()
        XCTAssertEqual(status.schema, 1)
        XCTAssertTrue(["running", "stale", "external", "stopped"].contains(status.server.state))
    }
}
