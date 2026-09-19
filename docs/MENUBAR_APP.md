# Flashchat menubar app

A native macOS menubar app for running and managing Flashchat without a
terminal. It shows the inference server's state at a glance and replaces the
TUI's configuration and model-management menus with native windows. The
terminal menu (`./flashchat`) and every CLI command keep working, and both use
the same backend, so you can switch between them freely.

## Build and run

```bash
make menubar        # builds macos/build/Flashchat.app (ad-hoc signed)
make menubar-run    # builds, then launches it
make menubar-test   # Swift unit tests for the app's core library
```

Requires macOS 14 or later and the Xcode command line tools. Drag
`macos/build/Flashchat.app` to `/Applications` if you like. It remembers the
Flashchat folder it was built from, and you can point it at another checkout
from **Overview → Flashchat folder**. To sign for distribution, set
`CODESIGN_IDENTITY` when running `macos/build-app.sh`.

Opening the app again while it is running brings up its window.
`open Flashchat.app --args --show Models` opens a specific section.

## What it does

**Menu bar icon.** The icon shows whether the server is stopped, starting,
ready, busy, or needs a restart. While a response is being generated it also
shows tokens per second (you can turn this off).

**Popover.** Shows:
- server status, including prompt-reading progress with the current layer
- the API URL, with a copy button
- the active model, with a switcher listing every enabled model's variants
- how much of the context window is in use, and the context cache size
- available RAM and memory pressure
- progress of any running operation
- Start, Stop and Restart buttons

**Overview.** Server details, the memory estimate broken down (weights in
RAM, context cache, expert cache limit), launch at login, quiet mode, and
shortcuts to a terminal chat or the terminal menu.

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
