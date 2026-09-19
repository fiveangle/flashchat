#!/bin/bash
# Sign macos/build/Flashchat.app (hardened runtime + entitlements).
#
#   macos/sign-app.sh [APP]
#
# SIGN_IDENTITY (set by the Makefile; see macos/local.mk.example) selects the certificate:
#   -  or adhoc          ad-hoc (default): runs on this Mac only
#   development          your "Apple Development" certificate
#   developer-id         your "Developer ID Application" certificate
#                        (required for notarization and distribution)
#   anything else        passed to codesign as-is: a full certificate name
#                        or its SHA-1 hash from `security find-identity`
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$ROOT/macos/build/Flashchat.app}"
ENTITLEMENTS="$ROOT/macos/Flashchat.entitlements"
REQUESTED="${SIGN_IDENTITY:--}"

[ -d "$APP" ] || { echo "No app at $APP — run 'make menubar' first." >&2; exit 1; }

find_identity() {
    local prefix="$1"
    security find-identity -v -p codesigning 2>/dev/null \
        | sed -n "s/^ *[0-9]*) \([0-9A-F]\{40\}\) \"\($prefix[^\"]*\)\"$/\1 \2/p"
}

pick_identity() {
    local prefix="$1" matches count
    matches="$(find_identity "$prefix")"
    count=$(printf '%s' "$matches" | grep -c . || true)
    if [ "$count" -eq 0 ]; then
        echo "No \"$prefix\" certificate found in your keychain." >&2
        echo "Available signing identities:" >&2
        security find-identity -v -p codesigning >&2 || true
        if [ "$prefix" = "Developer ID Application" ]; then
            echo "Create one in Xcode → Settings → Accounts → Manage Certificates → + → Developer ID Application" >&2
            echo "(requires a paid Apple Developer Program membership)." >&2
        fi
        exit 1
    fi
    if [ "$count" -gt 1 ]; then
        echo "Several \"$prefix\" certificates found; pick one with SIGN_IDENTITY=<SHA-1 or full name>:" >&2
        printf '%s\n' "$matches" >&2
        exit 1
    fi
    printf '%s' "$matches" | cut -d' ' -f1
}

case "$REQUESTED" in
    -|adhoc|ad-hoc) IDENTITY="-" ;;
    development)    IDENTITY="$(pick_identity "Apple Development")" ;;
    developer-id)   IDENTITY="$(pick_identity "Developer ID Application")" ;;
    *)              IDENTITY="$REQUESTED" ;;
esac

if [ "$IDENTITY" = "-" ]; then
    TIMESTAMP="--timestamp=none"
    LABEL="ad-hoc"
else
    # Secure timestamps are required for notarization (needs network access).
    TIMESTAMP="--timestamp"
    LABEL="$(security find-identity -v -p codesigning | grep -F "$IDENTITY" | sed 's/.*"\(.*\)"/\1/' | head -1)"
    LABEL="${LABEL:-$IDENTITY}"
fi

codesign --force --options runtime $TIMESTAMP \
    --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"
codesign --verify --strict "$APP"
echo "Signed $APP ($LABEL)"
