# Flashchat menubar app

A native macOS menubar app for running and managing Flashchat without a
terminal. It shows the inference server's state at a glance and replaces the
TUI's configuration and model-management menus with native windows. The
terminal menu (`./flashchat`) and every CLI command keep working, and both use
the same backend, so you can switch between them freely.

## Build, install, and sign

```bash
make menubar             # build macos/build/Flashchat.app (ad-hoc signed)
make menubar-run         # build and launch it
make menubar-install     # build and install to /Applications (INSTALL_DIR=… to change)
make menubar-uninstall   # remove the installed app; settings are kept
make menubar-test        # Swift unit tests for the app's core library
make menubar-clean       # remove build output
```

Requires macOS 14 or later and the Xcode command line tools. Installing quits
a running copy of the app and relaunches the new one. The inference server
runs as a separate process and keeps running. The app remembers which
Flashchat folder it was built from; you can point it at another checkout from
**Overview → Flashchat folder**.

### Signing

`SIGN_IDENTITY` picks the certificate for `make menubar`,
`make menubar-install` and `make menubar-sign`. The make targets never read
environment variables. Settings come from:

1. Defaults in the Makefile (ad-hoc signing, install to `/Applications`).
2. `macos/local.mk`, an optional git-ignored file with your per-machine
   defaults. Copy `macos/local.mk.example` to start; for example, set
   `SIGN_IDENTITY := development` to always sign with your Apple Development
   certificate.
3. The make command line, for a one-off override, e.g.
   `make menubar SIGN_IDENTITY=-`.

| `SIGN_IDENTITY` | Certificate | Use |
|---|---|---|
| `-` (default) | ad-hoc | Runs on this Mac |
| `development` | your "Apple Development" certificate | Your own Macs |
| `developer-id` | your "Developer ID Application" certificate | Distribution; required for notarization |
| a full name or SHA-1 | exactly that certificate | When you have more than one of a kind |

```bash
make menubar SIGN_IDENTITY=developer-id       # build + sign with Developer ID
make menubar-sign SIGN_IDENTITY=development   # re-sign the existing build, no rebuild
make menubar-verify                        # show the signature and Gatekeeper's verdict
```

`security find-identity -v -p codesigning` lists your certificates. A
Developer ID Application certificate needs a paid Apple Developer Program
membership; create it in Xcode → Settings → Accounts → Manage Certificates.
Every build uses the hardened runtime, plus the Apple Events entitlement the
app needs to open chats in Terminal (`macos/Flashchat.entitlements`).

### Notarization (distribution)

One-time setup, which stores your credentials in the keychain:

```bash
xcrun notarytool store-credentials flashchat-notary --apple-id <you@example.com> --team-id <TEAMID>
```

Then set `SIGN_IDENTITY := developer-id` and
`NOTARY_PROFILE := flashchat-notary` in `macos/local.mk`, and run:

```bash
make menubar
make menubar-notarize   # submit, wait, staple, check
```

Opening the app again while it is running brings up its window.
`open Flashchat.app --args --show Models` opens a specific section.

## What it does

**Menu bar icon.** The icon shows whether the server is stopped, starting,
ready, busy, or needs a restart. While a response is being generated it also
shows tokens per second (you can turn this off).

**Dock icon or menu bar icon, your choice.** Flashchat ships as a menubar-only
app, and **Overview → Show Dock icon** turns it into an ordinary Mac app: a
Dock icon, ⌘-Tab, a full menu bar (including Edit, so cut/copy/paste work in
text fields, and a Server menu), and a Dock badge that warns when the server
needs a restart or shows the decode speed. The switch takes effect
immediately, no relaunch. You can also hide the menu bar icon and keep only
the Dock icon; at least one of the two always stays on. In Dock mode,
**Open the window when Flashchat starts** controls whether launching opens the
window — turn it off if Flashchat launches at login and you only want the
Dock icon.

**Popover.** Shows:
- server status, including prompt-reading progress with the current layer
- the API URL, with a copy button
- the active model, with a switcher listing every enabled model's variants
- how much of the context window is in use, and the context cache size
- available RAM and memory pressure
- progress of any running operation
- Start, Stop and Restart buttons

**Overview.** Server details, the memory estimate broken down (weights in
RAM, context cache, expert cache limit), launch at login, Dock/menu bar
presence, how often to check server status, quiet mode, and shortcuts to a
terminal chat or the terminal menu.

**Models.** Every registry model with per-variant status. Available actions:
- **Use** a variant.
- **Prepare** a variant that isn't built. You see a plan first: the steps,
  disk space needed, and where the original files will come from (this Mac,
  offload storage, restoring from offload storage, or a HuggingFace
  download).
- **Repair** broken or missing artifacts.
- **Verify** with a full hash check.
- **Restore** from offload storage, or **Offload** to it.
- **Delete** parts of a model. You must type the model id to confirm, just
  as in the TUI.
- **Add** a model from HuggingFace.

Long operations show live progress and a log, can be cancelled (like Ctrl-C),
and post a notification when they finish.

**Settings.** Generation, sampling, server, storage and advanced options, with
validation and the model's sampling profiles. Saving tells you when the
running server needs a restart and offers to do it.
Expand Advanced options below Storage, then hover over a setting for its
explanation. “Use reduced-precision predictor weights” is on for the
quantized predictor and off to request BF16 weights. Existing preferences are
preserved; the saved `MTP_BF16` key retains its original meaning.

**Logs.** A live view of the server log, plus the app's own activity log.

## Safety

- **Memory check before start.** The app compares the server's estimated RAM
  (non-expert weights + context cache + overhead) with available memory and
  macOS memory pressure. If memory is tight it asks before starting. If there
  isn't enough, it warns that the Mac may swap heavily and makes you confirm.
- **Server lifetime.** The server runs as its own process. Quitting or
  crashing the app never stops inference.
- **Confirmation.** Every destructive or long action shows what it will do
  and asks first. The app never starts downloads or builds on its own.

## Privacy

The app talks to nothing but your own machine: `GET /health` on the local
server, the `flashchat` launcher, and `modelmgr`. It has no analytics and no
network access of its own. Downloads happen only when you ask for them, and
they go to HuggingFace from `modelmgr`, as they do in the TUI.

What a **signed app bundle** tells anyone who receives it:

- **The signing certificate's name**, readable with `codesign -dvv`. An
  *Apple Development* certificate carries the Apple ID it was issued to — your
  email address. Keep those builds on your own Macs; sign anything you share
  with a Developer ID certificate, whose name is the one on your developer
  account. `make menubar-sign` prints this warning when it signs with a
  development certificate.
- **Your team identifier**, in any signed build. That is not sensitive.
- **This checkout's path**, which contains your username. Local builds embed
  it so the app can find the launcher; builds signed with a Developer ID leave
  it out, as does `EMBED_REPO_ROOT=0 macos/build-app.sh`, and the app then
  asks the person for a folder on first run. The compiled binary carries no
  build paths.

**Notarization uploads the app bundle to Apple**, so do it on a build that
left the checkout path out. That is automatic for Developer ID builds.

Things that stay local but are worth knowing about: **Copy Diagnostics** in
Overview puts your model list and folder paths on the clipboard, for pasting
into a bug report; the **Logs** tab reads the server log, which records prompt
text when you turn on `SERVER_HTTP_LOG`; and the app runs unsandboxed with the
Apple Events entitlement, which it uses only to open Terminal for
**New Chat in Terminal**.

## Resource use

Measured on an M-series Mac with a server running and the app idle:

| | Memory | CPU while idle |
|---|---|---|
| Menubar app | 77 MB | ~0.4 CPU-seconds a minute (~0.7% of one core) |
| Menubar app, quiet mode | 77 MB | ~0.1 CPU-seconds a minute |
| Terminal.app, one window + shell | ~140 MB | ~0 at a prompt |

The app's memory is flat; a terminal's grows with scrollback and with every
window it restores, which is why the TUI's real cost depends on how you use
your terminal. The app's CPU is the price of live status: it polls
`GET /health` on an interval you choose (**Overview → Check server status**,
5 seconds by default, halved while a prompt is being read or a response
generated), and `flashchat status --json` once a minute for the
restart-needed check. The TUI does the equivalent work on every menu redraw.

## Quiet mode (benchmarks)

A menubar app can't be hidden during benchmarks the way windows can. In quiet
mode the app polls every 30 seconds and keeps its icon static, so it adds no
measurable CPU or GPU load. Turn it on from the popover's ⋯ menu or
**Overview**, or create the file `~/.config/flashchat/menubar-quiet`.
`tests/bench_api.sh` (`make bench-api`) creates this file for the length of
the run and removes it afterwards. It leaves the file alone if quiet mode was
already on.

## Architecture

```
Flashchat.app (SwiftUI)
 ├─ ./flashchat status --json          server state, incl. "restart needed"
 ├─ ./flashchat serve --non-interactive / serve --stop [--force|--external]
 ├─ GET /health                        live phase, progress, context use
 └─ python -m modelmgr api …           models, settings, plans, operations
```

The app contains no configuration or model-management logic of its own:

- **Server control** goes through the launcher, the same code path as
  `./flashchat serve`. It covers the runtime signature ("restart needed"),
  stale-binary rebuilds, and verified stop.
- **Everything about models and settings** goes through `modelmgr api`
  (`modelmgr/api.py`):
  - Query commands (`state`, `model`, `plan …`) print one JSON object.
  - Change commands (`select`, `set`, `enable`) validate first, then write
    through `configfile`, like the wizard does.
  - `run …` operations stream NDJSON progress events ending with a `done`
    event. They never prompt; every choice the TUI asks for is an explicit
    argument, and the app asks the user beforehand.
- **Settings are schema-driven.** `modelmgr/settings.py` defines every
  user-facing setting once. The TUI wizard prompts from it and the app
  builds its forms from it, so a new setting added there appears in both.
  Adding a debug flag still follows the full config chain in AGENTS.md.

The Swift package has two targets:
- **`FlashchatKit`:** JSON models, the process runner, server-state
  derivation, the throughput meter and the memory check. Unit-tested against
  fixtures generated from the real API.
- **`FlashchatBar`:** the SwiftUI app.

## Chat

The app doesn't include a chat window. **New Chat in Terminal** opens the
existing chat TUI. Any OpenAI-compatible client can use the API URL shown in
the popover.
