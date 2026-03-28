# Flash-MoE Implementation Plan

> Last Updated: 2026-03-27
> Status: Phase 1-6 Complete (Future Enhancements Pending)

---

## Overview

This document captures the implementation plan for the Flash-MoE unified management framework. As implementation progresses, changes will be documented here with reasons for any scope modifications.

---

## Original Plan (2026-03-27)

### Core Philosophy
- **Single entry point**: `flashchat` handles everything
- **Auto-setup**: Missing components detected and prompted on first use
- **No separate setup command**: Setup is implicit in any command that needs it

### Configuration System

| Item | Value |
|------|-------|
| Config location | `~/.config/flash-moe/config` |
| Config created | Automatically on first run |
| PID file | `~/.config/flash-moe/server.pid` |
| Sessions | `~/.flash-moe/sessions/` (for compatibility with chat.m) |
| Reset | `flashchat config` → re-run setup wizard (retains sessions) |

### Command Interface

```bash
flashchat [command] [options]

Commands:
  chat              Start chat (auto-sets up if missing)
  chat --resume ID  Resume session
  serve             Start API server (auto-sets up if missing)
  serve --stop      Stop running server
  prompt "..."      Single prompt (auto-sets up if missing)
  config            Edit configuration
  status            Show system status
  sessions          List sessions
  help              Show this help

Options:
  --config FILE     Use specific config
  -v, --verbose    Verbose output
  -q, --quiet      Quiet mode
```

### Implementation Phases

| Phase | Description |
|-------|-------------|
| **1** | Config system (directory, template, loader, env support) |
| **2** | flashchat core (entry point, dispatcher, help) |
| **3** | Auto-setup (check/prompt for missing components) |
| **4** | Server & Chat (serve, chat, prompt, auto-start) |
| **5** | Session management (list, resume, delete) |
| **6** | Interactive menu (no command = menu mode) |

### Server Port Handling
- Check if port is in use
- If in use: notify user, suggest next available port (port+1, port+2)
- Let user choose

---

## Implementation History

### 2026-03-27: Initial Implementation

**Changes from original plan:**
- None yet - initial implementation matches plan exactly

**Completed:**
- Phase 1: Config System
- Phase 2: flashchat core
- Phase 3: Auto-setup integration
- Phase 4: Server & Chat
- Phase 5: Session Management
- Phase 6: Interactive Menu
- RUN.md documentation updated

**Files created:**
- `flashchat` - Unified management CLI
- `lib/config.sh` - Configuration loader

**Files modified:**
- `metal_infer/infer.m` - Added env var support
- `metal_infer/extract_weights.py` - Added env var support
- `metal_infer/export_tokenizer.py` - Added env var support
- `repack_experts.py` - Added env var support
- `metal_infer/repack_experts_2bit.py` - Added env var support
- `RUN.md` - Updated documentation

---

## Future Considerations

### Potential Enhancements (Not Yet Implemented)

1. **Config reset via `flashchat config`** ✅ IMPLEMENTED
   - Added `--reset` to re-run setup wizard (preserves sessions)
   - Added `--full-reset` to delete all data
   - Added full config prompts: model repo, quantization, max tokens,
     temperature, top-p, server port, server host, show thinking, color output
   - Interactive prompts when run from terminal, defaults when piped

2. **Remove streamchat** ✅ COMPLETED
   - streamchat has been removed

3. **2-bit expert extraction wizard** ⏸ DEFERRED
   - Currently prompts for quantization, code logic is correct
   - Full testing deferred to fresh "out of the box" test at end

4. **Server persistence validation** ✅ COMPLETED
   - Server starts and stops correctly
   - Status shows running with PID
   - Verified serve --stop functionality

5. **Chat client integration** ✅ COMPLETED
   - Verified: chat auto-starts server, connects seamlessly, returns to menu
   - Future: Show chat status in full TUI (see below)

6. **Enhanced Session Resume in Interactive Menu** ✅ COMPLETED
   - Shows numbered list of sessions when resuming
   - Accepts number (1, 2, 3) or session ID directly
   - Example:
     ```
     Select a session to resume:

     [1] chat-1234567890 (5 turns)
     [2] chat-0987654321 (12 turns)

     Enter number or session ID:
     ```

7. **Benchmark Commands** ✅ COMPLETED
   - Added `flashchat benchmark` command
   - Supports: run, verify, bench, moe, moebench, full, fullbench
   - Usage: `flashchat benchmark <target>`

---

## Notes

- Configuration follows XDG Base Directory Specification
- Model path auto-detection from HuggingFace cache
- Environment variables override config file settings
- Interactive menu provides user-friendly interface

---

## Future: Full TUI Considerations

When converting flashchat to a full TUI application, consider:

1. **Show chat status** - Display if chat is active in status view
2. **Inline chat** - Run chat within the TUI instead of spawning separate binary
3. **Rich status** - Show model info, GPU usage, token count, etc.
4. **Keybindings** - Vim-style keybindings (j/k for navigation)
5. **Themes** - Color schemes for light/dark mode
6. **Split views** - Server logs, session history, settings panels
