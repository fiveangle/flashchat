# HTTP responsiveness and inference ownership

Flashchat runs one generation at a time. A second generation request receives
HTTP 503 with error type `server_busy`; there is no conversation queue or
concurrent model execution. Clients can retry after the active request finishes.

The HTTP event thread accepts bounded requests, handles `/health`, `/v1`,
`/v1/models` and preflight requests, and relays inference output. The original
inference thread owns all model state, caches, and accelerator operations.
Streaming and non-streaming responses use the same bounded relay. A client
that stops accepting output for ten seconds is disconnected. Request uploads
also have a ten-second deadline, with a 1 MiB request limit and at most sixteen
pending HTTP connections. HTTP payload logging remains available.

The inference worker publishes a small status snapshot under a short mutex.
No inference or network I/O occurs while that mutex is held. `/health` retains
its previous fields and adds:

| Field | Meaning |
| --- | --- |
| `phase` | `idle`, `preparing`, `prefill`, or `generating` |
| `context_used` | Positions committed across the model; retained after completion |
| `max_context` | Actual configured context window |
| `cached_tokens` | Positions restored from the system prompt cache for this request |
| `prompt_tokens` | Tokens requiring prompt processing, excluding restored context |
| `prefill_done` | Completed prompt positions; advances when a whole chunk completes |
| `generated_tokens` | Generated output tokens, including reasoning/tool output |
| `chunk`, `chunks`, `layer`, `layers` | Current batched prefill work; zero when not applicable |

During a chunk, some layers have processed additional positions, but those
positions do not count as occupied context until every layer has completed.
The layer indicator shows progress during this interval. Per-token prefill
updates completed positions. Updates also work for non-streaming requests.
`ready: true` means the loaded server is operational, not that it has capacity
for another generation.

The management menu fetches one snapshot per refresh for quantization,
context usage, and processing progress. An unavailable server produces an
explicit unavailable reading instead of zero usage. Its capacity fallback
uses the configured window (64K by default), clamped by known model limits.
The MiB figure describes allocated context-cache capacity, not memory used by
just the occupied positions. The menu remains a refresh-on-action interface.
The status block always shows the same rows: processing state, prompt progress,
prompt chunk, model layer, reused context, and generated tokens. Inactive rows
say so explicitly; idle and stopped states do not display stale request counters.
Unavailable status preserves every row rather than collapsing the menu layout.

Disconnecting a client releases its network resources and makes subsequent
worker writes fail. Admission remains busy until the inference worker actually
returns. This does not add cancellation inside a running prefill kernel.
SIGTERM/SIGINT stop the HTTP transport and retain the existing inference-drain
behavior before model teardown.

Validation:

```sh
make server-http-smoke       # Real transport with a lightweight fake worker + menu rendering
python3 tests/test_server_live.py --url http://127.0.0.1:9999  # Idle, updated model server
make tool-template-smoke
make bench-api              # Idle machine, canonical performance validation
make bench-report
```

See [live acceptance and baseline comparison](HTTP_RESPONSIVENESS_VALIDATION.md)
for the measured results and test-artifact provenance.
