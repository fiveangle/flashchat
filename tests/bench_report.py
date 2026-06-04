#!/usr/bin/env python3
"""Read assets/api_perf_log.tsv and report performance trends / regressions.

Groups benchmark rows by (hw_model, model, scenario, metric_type) and compares the
latest commit's value against the previous one, flagging changes beyond a threshold.
Hardware is keyed on hw_model (stable) rather than hostname (the box was renamed
fiveangle-6.lan -> FiveAngle.local; same Mac17,2).

Higher-is-better: decode_tok_per_sec, stream_deltas, tok_per_sec.
Lower-is-better:  prefill_ms, tool_call_ms, duration_ms.

stdlib only. Usage: tests/bench_report.py [--all-hw] [--threshold 5] [--last 6] [--log FILE]
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
    ap.add_argument("--threshold", type=float, default=5.0, help="percent change to flag (default 5)")
    ap.add_argument("--last", type=int, default=6, help="history points to show per metric")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"no perf log at {args.log}", file=sys.stderr); return 1

    hw = None if args.all_hw else this_hw_model()
    rows = []
    with open(args.log) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("server_mode") != "bench":
                continue          # only the canonical benchmark, not ad-hoc smoke rows
            if hw and r.get("hw_model") != hw:
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

    regressions = 0
    last_model = None
    for key in sorted(groups):
        hwm, model, scenario, mt = key
        if model != last_model:
            print(f"\n## {model}   [{hwm}]"); last_model = model
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
                verdict = f"▼ {pct:+.1f}%"; regressions += 1
        unit = "tok/s" if mt in ("decode_tok_per_sec", "tok_per_sec") else ("ms" if mt.endswith("_ms") else "")
        print(f"  {scenario:26} {mt:18} {latest[2]:8g} {unit:5} {verdict:>10}   [{hist}]")

    print(f"\n# {regressions} regression(s) beyond ±{args.threshold:.0f}% vs previous commit.")
    return 1 if regressions else 0

if __name__ == "__main__":
    sys.exit(main())
