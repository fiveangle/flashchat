#!/bin/bash
# Build macos/build/Flashchat.app from the Swift package.
#
#   macos/build-app.sh            release build, ad-hoc signed
#   CODESIGN_IDENTITY="Developer ID Application: …" macos/build-app.sh
#
# The bundle records this checkout's path (FlashchatRepoRoot) so it finds the
# launcher and modelmgr; the app also lets the user pick another folder.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/macos/FlashchatBar"
OUT="$ROOT/macos/build"
APP="$OUT/Flashchat.app"
CONFIG="${CONFIG:-release}"

swift build -c "$CONFIG" --package-path "$PKG" --product FlashchatBar
BIN_DIR="$(swift build -c "$CONFIG" --package-path "$PKG" --show-bin-path)"

VERSION="$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)"
BUILD="$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 1)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/FlashchatBar" "$APP/Contents/MacOS/Flashchat"

ICONSET="$OUT/AppIcon.iconset"
rm -rf "$ICONSET"
swift "$ROOT/macos/tools/make-icon.swift" "$ICONSET"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$ICONSET"

plist_escape() {
    local s="$1"
    s="${s//&/&amp;}"; s="${s//</&lt;}"; s="${s//>/&gt;}"
    printf '%s' "$s"
}

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleDisplayName</key><string>Flashchat</string>
    <key>CFBundleExecutable</key><string>Flashchat</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleIdentifier</key><string>dev.flashchat.menubar</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>Flashchat</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$(plist_escape "$VERSION")</string>
    <key>CFBundleVersion</key><string>$BUILD</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
    <key>NSAppleEventsUsageDescription</key><string>Flashchat opens chats and the terminal menu in Terminal.</string>
    <key>FlashchatRepoRoot</key><string>$(plist_escape "$ROOT")</string>
</dict>
</plist>
PLIST

codesign --force --options runtime --sign "${CODESIGN_IDENTITY:--}" "$APP"
echo "Built $APP ($VERSION)"
