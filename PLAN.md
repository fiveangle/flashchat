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
| Sessions | `~/.config/flash-moe/sessions/` |
| Migration | None - streamchat features incorporated directly |
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

### streamchat Deprecation
- Use streamchat as reference during development
- Remove entirely when flashchat is complete
- No deprecation warning needed (direct replacement)

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

1. **Config reset via `flashchat config`**
   - Original plan mentioned "reset environment" option
   - Not yet implemented - requires adding interactive config editing

2. **Remove streamchat**
   - Original plan: remove entirely when flashchat complete
   - Not yet done - streamchat still exists for reference

3. **2-bit expert extraction wizard**
   - Currently prompts for quantization but extraction logic needs testing

4. **Server persistence validation**
   - Need to test server --stop functionality

5. **Chat client integration**
   - Need to verify chat client connects properly to server

6. **Enhanced Session Resume in Interactive Menu**
   - Currently: prompts for session ID directly (user must know the ID)
   - Planned: Present numbered list of existing sessions, allow selection by number
   - Fallback: If input is not a number, treat as session ID directly
   - Example:
     ```
     Select a session to resume:
     
     [1] chat-1234567890 (5 turns)
     [2] chat-0987654321 (12 turns)
     [3] my-session-name (3 turns)
     
     Enter number or session ID: 
     ```

7. **Benchmark Commands**
   - Add `flashchat benchmark` command for performance testing
   - Based on existing Makefile targets: run, verify, bench, moe, moebench, full, fullbench
   - Also include infer options: --timing, --freq, --cache-telemetry
   - Could be power-user feature or `flashchat status --verbose` diagnostic

---

## Notes

- Configuration follows XDG Base Directory Specification
- Model path auto-detection from HuggingFace cache
- Environment variables override config file settings
- Interactive menu provides user-friendly interface
