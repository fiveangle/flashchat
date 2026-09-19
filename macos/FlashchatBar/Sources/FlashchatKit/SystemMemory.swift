import Darwin
import Foundation

public enum MemoryPressure: Int, Sendable {
    case normal = 1
    case warning = 2
    case critical = 4

    public var title: String {
        switch self {
        case .normal: return "Normal"
        case .warning: return "Elevated"
        case .critical: return "Critical"
        }
    }
}

public struct SystemMemorySnapshot: Sendable, Equatable {
    public var totalBytes: Int64
    /// Free + inactive + purgeable + speculative: what can be handed to a new
    /// process without swapping.
    public var availableBytes: Int64
    public var swapUsedBytes: Int64
    public var pressure: MemoryPressure

    public init(totalBytes: Int64, availableBytes: Int64, swapUsedBytes: Int64,
                pressure: MemoryPressure) {
        self.totalBytes = totalBytes
        self.availableBytes = availableBytes
        self.swapUsedBytes = swapUsedBytes
        self.pressure = pressure
    }

    public static func current() -> SystemMemorySnapshot {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        var pageSize: vm_size_t = 0
        host_page_size(mach_host_self(), &pageSize)
        let page = Int64(pageSize)
        let available = result == KERN_SUCCESS
            ? (Int64(stats.free_count) + Int64(stats.inactive_count)
               + Int64(stats.purgeable_count) + Int64(stats.speculative_count)) * page
            : 0

        var swap = xsw_usage()
        var swapSize = MemoryLayout<xsw_usage>.size
        sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0)

        var level: Int32 = 1
        var levelSize = MemoryLayout<Int32>.size
        sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &levelSize, nil, 0)

        return SystemMemorySnapshot(
            totalBytes: Int64(ProcessInfo.processInfo.physicalMemory),
            availableBytes: available,
            swapUsedBytes: Int64(swap.xsu_used),
            pressure: MemoryPressure(rawValue: Int(level)) ?? .normal)
    }
}

/// Decides whether starting the server is safe. Past incidents: launching a
/// model without checking headroom swapped the machine into a hard restart.
public enum MemoryPreflight {
    public enum Verdict: Equatable, Sendable {
        case ok
        case tight(String)
        case insufficient(String)
    }

    public static let safetyMargin: Int64 = 2 << 30
    public static let comfortMargin: Int64 = 4 << 30

    public static func evaluate(estimateBytes: Int64, memory: SystemMemorySnapshot) -> Verdict {
        let need = Format.bytes(estimateBytes)
        let have = Format.bytes(memory.availableBytes)
        if memory.pressure == .critical {
            return .insufficient("macOS reports critical memory pressure. Starting the server now "
                                 + "is likely to swap heavily. Close other apps first.")
        }
        if estimateBytes + safetyMargin > memory.availableBytes {
            return .insufficient("The server needs about \(need) of RAM, but only \(have) is "
                                 + "available. Close other apps or lower the context window.")
        }
        if memory.pressure == .warning {
            return .tight("macOS reports elevated memory pressure (\(have) available, "
                          + "about \(need) needed).")
        }
        if estimateBytes + comfortMargin > memory.availableBytes {
            return .tight("Memory is tight: about \(need) needed, \(have) available.")
        }
        return .ok
    }
}
