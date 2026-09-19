"""User-facing configuration settings: one schema for every front end.

The config wizard (TUI) and the menubar app (via `modelmgr api`) both render
from these definitions, so a setting added here reaches every surface.
`label` is the TUI prompt text; `title` is the short label native forms use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

SAMPLING_KEYS = (("temperature", "TEMPERATURE"), ("top_p", "TOP_P"),
                 ("top_k", "TOP_K"), ("min_p", "MIN_P"),
                 ("presence_penalty", "PRESENCE_PENALTY"),
                 ("repetition_penalty", "REPETITION_PENALTY"),
                 ("reasoning", "REASONING"))


@dataclass(frozen=True)
class Setting:
    key: str
    section: str
    title: str
    label: str = ""
    help: str | None = None
    kind: str = "text"          # bool | int | float | text | path | choice | mtp
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    clear_word: str | None = None   # empty value allowed; TUI types this word to clear
    empty_title: str | None = None  # what an empty value means, for native forms
    parent: str | None = None

    def to_json(self) -> dict:
        data = asdict(self)
        data["choices"] = list(self.choices)
        return data


GENERATION = (
    Setting("MAX_TOKENS", "generation", "Max response tokens", kind="int", minimum=1),
    Setting("CONTEXT_WINDOW", "generation", "Context window (tokens)", kind="int",
            minimum=512, empty_title="Default (64K, capped at model max)",
            help="Larger windows use more RAM for the context cache."),
    Setting("KV_QUANT", "generation", "Context cache quantization", kind="choice",
            choices=("off", "q8", "q4"),
            help="off = fp32 lossless; q8 = ~lossless, best for large windows; q4 = lossy, smallest."),
    Setting("SAMPLING_PROFILE", "generation", "Sampling profile", kind="choice"),
    Setting("TEMPERATURE", "sampling", "Temperature", kind="float", minimum=0, maximum=2),
    Setting("TOP_P", "sampling", "Top-p", kind="float", minimum=0, maximum=1),
    Setting("TOP_K", "sampling", "Top-k", kind="int", minimum=0),
    Setting("MIN_P", "sampling", "Min-p", kind="float", minimum=0, maximum=1),
    Setting("PRESENCE_PENALTY", "sampling", "Presence penalty", kind="float",
            minimum=-2, maximum=2),
    Setting("REPETITION_PENALTY", "sampling", "Repetition penalty", kind="float",
            minimum=0, maximum=2),
    Setting("REASONING", "sampling", "Reasoning (thinking) mode", kind="bool"),
    Setting("ACTIVE_EXPERTS", "sampling", "Active experts per token (K)", kind="int",
            minimum=1, empty_title="Model default"),
)

SERVER = (
    Setting("SERVER_PORT", "server", "Port", "Port", kind="int", minimum=1, maximum=65535),
    Setting("SERVER_HOST", "server", "Host clients connect to", "Host clients connect to"),
    Setting("SERVER_BIND", "server", "Listen address",
            "Listen on IPv4 address (0.0.0.0 = all interfaces)",
            help="127.0.0.1 = this Mac only; 0.0.0.0 = all network interfaces."),
    Setting("SERVER_LOG_PATH", "server", "Log file", "Log path", kind="path"),
)

STORAGE = (
    Setting("HUGGINGFACE_CACHE_DIR", "storage", "HuggingFace cache folder",
            "HuggingFace cache dir", kind="path"),
    Setting("OFFLOAD_DIR", "storage", "Offload folder for archived models",
            "Offload dir for archived models ('-' to disable)", kind="path",
            clear_word="-", empty_title="Not configured"),
)

ADVANCED = (
    Setting("IO_THREADS", "advanced", "Parallel expert disk readers",
            "Parallel expert disk readers (1-16)", kind="int", minimum=1, maximum=16),
    Setting("GPU_ROPE", "advanced", "GPU rotary position encoding",
            "GPU rotary position encoding (0/1)", kind="bool"),
    Setting("FUSED_ATTN", "advanced", "Fused attention for fp32 context cache",
            "Fused attention for fp32 context cache (0/1)",
            "Quantized context caches use their own attention path.", kind="bool"),
    Setting("PREFILL_RELEASE", "advanced", "Release temporary prompt-processing memory",
            "Release temporary prompt-processing memory (0/1)", kind="bool"),
    Setting("SERVER_DEBUG", "advanced", "Server debug logging",
            "Server debug logging to server.log (0/1)", kind="bool"),
    Setting("SERVER_HTTP_LOG", "advanced", "HTTP request/response log",
            "HTTP request/response log to http.log (0/1)", kind="bool"),
    Setting("BATCH_PREFILL", "advanced", "Batched prompt processing",
            "Batched prompt processing (0/1)",
            "~4-6x faster prompt reading; responses unchanged in normal use.", kind="bool"),
    Setting("PREFILL_CHUNK", "advanced", "Chunk size (tokens)",
            "  ^ chunk size in tokens (8-4096)",
            "Bigger = faster but more working RAM: 1024 ~170 MB, 2048 ~340 MB.",
            kind="int", minimum=8, maximum=4096, parent="BATCH_PREFILL"),
    Setting("ANE_PREFILL", "advanced", "Neural Engine expert offload",
            "  ^ Neural Engine expert offload (0/1)",
            "Faster prompt reading on long prompts; auto-falls back to GPU if the hardware lacks a Neural Engine.",
            kind="bool", parent="BATCH_PREFILL"),
    Setting("ANE_MIN_CHUNK", "advanced", "Use GPU below this prompt length",
            "  ^ GPU below this prompt length in tokens (0=always Neural Engine)",
            "Short prompts run the exact GPU path (measured faster AND bit-faithful); long prompts keep the Neural Engine overlap.",
            kind="int", minimum=0, parent="ANE_PREFILL"),
    Setting("FUSE_LINEAR", "advanced", "Fused GPU scheduling for linear-attention layers",
            "Fused GPU scheduling for linear-attention layers (0/1)",
            "~12% faster generation with byte-identical outputs; disable only for A/B comparisons.",
            kind="bool"),
    Setting("ADAPTIVE_K_MASS", "advanced", "Adaptive expert count by routing mass",
            "Adaptive expert count by routing mass (off, or 0.85-0.99)",
            "Skips tail experts below this probability mass: ~0.90 reads ~13% fewer expert bytes; outputs can differ slightly.",
            kind="float", minimum=0.85, maximum=0.99, clear_word="off", empty_title="Off"),
    Setting("PREFILL_DEBUG", "advanced", "Prompt-processing debug",
            "  ^ debug (0=off, 1=chunk timings, 2=+state dump, slow)",
            kind="choice", choices=("0", "1", "2"), parent="BATCH_PREFILL"),
    Setting("PREAD_PROFILE", "advanced", "Disk-read timing log",
            "Disk-read timing log (off, or a .tsv path)",
            "For diagnosing slow expert streaming; analyze with tools/pread_profile_analyze.py.",
            kind="path", clear_word="off", empty_title="Off"),
    Setting("PREAD_PROFILE_CAP", "advanced", "Max recorded disk-read events",
            "  ^ max recorded events before it stops", kind="int", minimum=1,
            parent="PREAD_PROFILE"),
    Setting("EXPERT_PIN_MAX_EXPERTS", "advanced", "Expert RAM cache target (experts)",
            "Expert RAM cache target in complete experts (auto=use GiB cap)",
            kind="int", minimum=0, clear_word="auto", empty_title="Auto (use GiB cap)"),
    Setting("EXPERT_PIN_MAX_GB", "advanced", "Expert cache limit (GiB)",
            "  ^ maximum GiB cache limit (0 disables cache)", kind="float", minimum=0,
            parent="EXPERT_PIN_MAX_EXPERTS"),
    Setting("EXPERT_PIN_AUTO_FRAC", "advanced", "Expert cache share of free RAM",
            "  ^ also capped to this fraction of free RAM (0.1-0.9)",
            kind="float", minimum=0.1, maximum=0.9, parent="EXPERT_PIN_MAX_EXPERTS"),
    Setting("EXPERT_PIN_MLOCK", "advanced", "Lock expert cache against swap",
            "  ^ lock pin cache against swap (0/1)",
            "Faster hits under memory pressure; skipped automatically if free RAM is too low.",
            kind="bool", parent="EXPERT_PIN_MAX_EXPERTS"),
    Setting("LM_HEAD_MLOCK", "advanced", "Lock vocabulary head in RAM",
            "Lock vocabulary head in RAM (0/1)",
            "~0.3 GB on q4 35B; keeps generation from re-faulting the big final projection. Skipped if free RAM is too low.",
            kind="bool"),
    Setting("EXPERT_SPLIT_IO", "advanced", "Overlap expert disk reads with GPU",
            "Overlap expert disk reads with GPU (0/1)",
            "Starts GPU on gate+up while the down half still streams from SSD; byte-identical outputs.",
            kind="bool"),
    Setting("CONVERSATION_CACHE", "advanced", "Reuse unchanged conversation turns",
            "Reuse unchanged conversation turns (0/1)",
            "Keeps one active conversation in memory; falls back safely when input changes.",
            kind="bool"),
    Setting("SYSTEM_PROMPT_CACHE", "advanced", "System prompt cache",
            "System prompt cache (0/1)",
            "Repeat requests skip re-reading the system prompt — big win for long agent prompts.",
            kind="bool"),
    Setting("SYSTEM_PROMPT_CACHE_MAX_ENTRIES", "advanced", "Max saved system prompts",
            "  ^ max saved prompts (1-64; entries can be tens of MB)",
            kind="int", minimum=1, maximum=64, parent="SYSTEM_PROMPT_CACHE"),
    Setting("SYSTEM_PROMPT_CACHE_DIR", "advanced", "System prompt cache folder",
            "  ^ cache folder ('-' = beside the model)", kind="path", clear_word="-",
            empty_title="Beside the model", parent="SYSTEM_PROMPT_CACHE"),
    Setting("MTP", "advanced", "Multi-token prediction",
            "Multi-token prediction (0=off, auto=model default, 2+=batch size)",
            "Lossless speculative decoding for models that ship a predictor head.",
            kind="mtp", empty_title="Model default"),
    Setting("MTP_BF16", "advanced", "BF16 predictor weights",
            "  ^ BF16 predictor weights (0/1; more RAM, slightly better drafts)",
            kind="bool", parent="MTP"),
    Setting("SHOW_THINKING", "advanced", "Show thinking tokens",
            "Show thinking tokens (0/1)", kind="bool"),
    Setting("COLOR_OUTPUT", "advanced", "Color output", "Color output (0/1)", kind="bool"),
)

ALL = GENERATION + SERVER + STORAGE + ADVANCED
BY_KEY = {s.key: s for s in ALL}
MODEL_KEYS = ("MODEL", "MODEL_BASE", "MODEL_VARIANT")


def mtp_value_enables(raw: str) -> bool:
    """Mirror the engine's parse_mtp_predictions truthiness: 0/off/no/false and
    empty (registry default) do not actively request MTP; auto/on/yes and any
    positive integer do."""
    s = (raw or "").strip().lower()
    if s in ("", "0", "off", "no", "false"):
        return False
    if s in ("auto", "on", "yes", "true"):
        return True
    return s.isdigit() and int(s) > 0


KV_MODES = (("off", "fp32, lossless"),
            ("q8", "~lossless, best for large windows"),
            ("q4", "lossy, smallest"))


def default_context_window(manifest) -> int:
    return min(65536, manifest.max_context) if manifest.max_context > 0 else 65536


def kv_cache_bytes(manifest, window: int, mode: str) -> int:
    """Wired GPU KV-buffer bytes across all full-attention layers (K + V
    [+ fp16 scales]) at `window` tokens; 0 when the geometry is unknown."""
    a = manifest.architecture
    n_kv = int(a.get("num_key_value_heads", 0) or 0)
    head_dim = int(a.get("head_dim", 0) or 0)
    n_full = int(a.get("num_hidden_layers", 0) or 0) // max(
        1, int(a.get("full_attention_interval", 1) or 1))
    kv_dim = n_kv * head_dim
    if kv_dim <= 0 or n_full <= 0:
        return 0
    per_token = {
        "off": 2 * kv_dim * 4,
        "q8": 2 * kv_dim + 2 * (n_kv * 2),
        "q4": 2 * (kv_dim // 2) + 2 * (n_kv * 2),
    }[mode]
    return per_token * n_full * window
