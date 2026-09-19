// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "FlashchatBar",
    platforms: [.macOS(.v14)],
    targets: [
        .target(name: "FlashchatKit", path: "Sources/FlashchatKit"),
        .executableTarget(
            name: "FlashchatBar",
            dependencies: ["FlashchatKit"],
            path: "Sources/FlashchatBar"
        ),
        .testTarget(
            name: "FlashchatKitTests",
            dependencies: ["FlashchatKit"],
            path: "Tests/FlashchatKitTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
