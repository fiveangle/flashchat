#!/bin/bash
# Notarize and staple a Developer ID–signed Flashchat.app.
#
#   NOTARY_PROFILE=<name> macos/notarize-app.sh [APP]
#
# One-time setup — store App Store Connect credentials in your keychain:
#   xcrun notarytool store-credentials <name> \
#       --apple-id <you@example.com> --team-id <TEAMID>
# (it prompts for an app-specific password from appleid.apple.com)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$ROOT/macos/build/Flashchat.app}"
PROFILE="${NOTARY_PROFILE:-}"

[ -d "$APP" ] || { echo "No app at $APP — run 'make menubar' first." >&2; exit 1; }
if [ -z "$PROFILE" ]; then
    echo "Set NOTARY_PROFILE to a notarytool keychain profile. One-time setup:" >&2
    echo "  xcrun notarytool store-credentials <name> --apple-id <you@example.com> --team-id <TEAMID>" >&2
    exit 1
fi
if ! codesign -dv "$APP" 2>&1 | grep -q '^Authority=Developer ID Application'; then
    echo "$APP is not signed with a Developer ID Application certificate." >&2
    echo "Rebuild with: make menubar SIGN_IDENTITY=developer-id" >&2
    exit 1
fi

ZIP="$(mktemp -d)/Flashchat.zip"
trap 'rm -rf "$(dirname "$ZIP")"' EXIT
ditto -c -k --keepParent "$APP" "$ZIP"
xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$APP"
spctl --assess --type execute --verbose "$APP"
echo "Notarized and stapled $APP"
