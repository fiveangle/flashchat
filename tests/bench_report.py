#!/usr/bin/env python3
"""Read assets/api_perf_log.tsv and report performance trends / regressions.

Groups benchmark rows by (hw_model, model, scenario, metric_type) and compares the
latest commit's value against the previous one, flagging changes beyond a threshold.
Hardware is keyed on hw_model (stable) rather than hostname (the box was renamed
fiveangle-6.lan -> FiveAngle.local; same Mac17,2).

Only models installed on this machine count toward the regression exit status. A
model's latest comparison stays in the log after it is offloaded, removed, or retired
from the registry, and would otherwise be flagged on every run until it is benchmarked
again. Those models are still reported, labeled, and counted with --include-uninstalled.

Higher-is-better: decode_tok_per_sec, stream_deltas, tok_per_sec.
Lower-is-better:  prefill_ms, tool_call_ms, duration_ms.

Usage: tests/bench_report.py [--all-hw] [--include-uninstalled] [--threshold 5] [--last 6] [--log FILE]
"""
import argparse, csv, os, subprocess, sys
from collections import defaultdict

LOWER_BETTER = {"prefill_ms", "tool_call_ms", "duration_ms"}
HIGHER_BETTER = {"decode_tok_per_sec", "tok_per_sec", "stream_deltas"}

def this_hw_model():
    try:
        return subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    except Exception:
        return None

def install_states(repo):
    """Registry model id -> install state on this machine, or None if unknown."""
    try:
        sys.path.insert(0, repo)
        from modelmgr.registry import Registry
        from modelmgr.status import all_statuses
        statuses = all_statuses(Registry.load())
    except Exception as e:
        print(f"# install state unavailable ({e}); counting every model\n")
        return None
    states = {}
    for status in statuses:
        for name, v in status.variants.items():
            state = "installed" if v.ready else ("offloaded" if v.offloaded else "not installed")
            for model_id in {v.resolved_id, *status.manifest.variant(name).legacy_ids}:
                states[model_id] = state
    return states

def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(repo, "assets", "api_perf_log.tsv"))
    ap.add_argument("--all-hw", action="store_true", help="don't filter to this machine's hw.model")
    ap.add_argument("--include-uninstalled", action="store_true",
                    help="count regressions for models not installed on this machine")
    ap.add_argument("--threshold", type=float, default=5.0, help="percent change to flag (default 5)")
    ap.add_argument("--last", type=int, default=6, help="history points to show per metric")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"no perf log at {args.log}", file=sys.stderr); return 1

    local_hw = this_hw_model()
    hw = None if args.all_hw else local_hw
    rows = []
    excluded = 0
    with open(args.log) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("server_mode") != "bench":
                continue          # only the canonical benchmark, not ad-hoc smoke rows
            if hw and r.get("hw_model") != hw:
                continue
            if r.get("status") != "pass":
                excluded += 1
                continue
            rows.append(r)

    if not rows:
        scope = "this machine" if hw else "any machine"
        print(f"no bench rows for {scope} in {args.log}", file=sys.stderr)
        print("run `make bench-api` first.", file=sys.stderr); return 1

    # group -> ordered list of (timestamp, commit, value)
    groups = defaultdict(list)
    for r in rows:
        mt = r["metric_type"]
        val = fnum(r["metric_value"]) if fnum(r["metric_value"]) is not None else fnum(r["tok_per_sec"])
        if val is None:
            continue
        groups[(r["hw_model"], r["model"], r["scenario"], mt)].append((r["timestamp"], r["commit"], val))

    print(f"# Flashchat perf report  ({'all hardware' if not hw else hw})")
    print(f"# threshold ±{args.threshold:.0f}%   (▲ better, ▼ worse, = flat)\n")
    if excluded:
        print(f"# excluded {excluded} non-pass/confounded row(s)\n")

    states = None if args.include_uninstalled else install_states(repo)

    def uncounted_reason(hwm, model):
        if states is None or hwm != local_hw:
            return None
        state = states.get(model, "no longer in registry")
        return None if state == "installed" else state

    regressions = 0
    uncounted = {}
    last_model = None
    for key in sorted(groups):
        hwm, model, scenario, mt = key
        reason = uncounted_reason(hwm, model)
        if model != last_model:
            note = f"   ({reason} — not counted)" if reason else ""
            print(f"\n## {model}   [{hwm}]{note}"); last_model = model
        pts = sorted(groups[key], key=lambda x: x[0])
        latest, prev = pts[-1], (pts[-2] if len(pts) > 1 else None)
        tail = pts[-args.last:]
        hist = "  ".join(f"{c}:{v:g}" for _, c, v in tail)
        verdict = "—"
        if prev and prev[2]:
            pct = (latest[2] - prev[2]) / prev[2] * 100.0
            better = pct > 0 if mt in HIGHER_BETTER else pct < 0
            if abs(pct) < args.threshold:
                verdict = f"= {pct:+.1f}%"
            elif better:
                verdict = f"▲ {pct:+.1f}%"
            else:
                verdict = f"▼ {pct:+.1f}%"
                if reason:
                    uncounted[(model, reason)] = uncounted.get((model, reason), 0) + 1
                else:
                    regressions += 1
        unit = "tok/s" if mt in ("decode_tok_per_sec", "tok_per_sec") else ("ms" if mt.endswith("_ms") else "")
        print(f"  {scenario:26} {mt:18} {latest[2]:8g} {unit:5} {verdict:>10}   [{hist}]")

    print(f"\n# {regressions} regression(s) beyond ±{args.threshold:.0f}% vs previous commit.")
    if uncounted:
        models = ", ".join(f"{m} ({n}, {why})" for (m, why), n in sorted(uncounted.items()))
        print(f"# not counted, model not installed here: {models}")
        print("# pass --include-uninstalled to count them.")
    return 1 if regressions else 0

if __name__ == "__main__":
    sys.exit(main())
