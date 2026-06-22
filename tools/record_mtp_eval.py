#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import socket
import subprocess
import sys


HEADER = [
    "timestamp", "branch", "commit", "hostname", "hw_model", "ram_gib", "cpu_summary",
    "model", "server_mode", "scenario", "endpoint", "stream", "tool_mode", "reasoning",
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
    "duration_ms", "metric_type", "metric_value", "tok_per_sec", "status", "notes",
]


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def sysctl(name):
    return run(["sysctl", "-n", name])


def repo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.dirname(here)


def git_branch(root):
    return run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"


def git_commit(root):
    return run(["git", "-C", root, "rev-parse", "--short", "HEAD"]) or "unknown"


def ram_gib():
    raw = sysctl("hw.memsize")
    if not raw:
        return ""
    try:
        return f"{int(raw) / (1024 ** 3):.1f}"
    except ValueError:
        return ""


def cpu_summary():
    p = sysctl("hw.perflevel0.physicalcpu") or sysctl("hw.physicalcpu_max") or "?"
    e = sysctl("hw.perflevel1.physicalcpu") or "?"
    g = sysctl("hw.gpucores") or sysctl("hw.gpu.core_count") or sysctl("hw.optional.gpu_core_count") or "?"
    return f"{p}p {e}e {g}g"


def ensure_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f, delimiter="\t", lineterminator="\n").writerow(HEADER)


def latest_metadata(path):
    fallback = {}
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("hw_model") and row.get("hw_model") != "unknown":
                    fallback = {
                        "hostname": row.get("hostname", ""),
                        "hw_model": row.get("hw_model", ""),
                        "ram_gib": row.get("ram_gib", ""),
                        "cpu_summary": row.get("cpu_summary", ""),
                    }
    except Exception:
        return {}
    return fallback


def metric_pairs(args):
    pairs = []
    for key in [
        "generated_tokens", "mtp_acceptance_pct", "mtp_accepted", "mtp_drafts",
        "mtp_spec_iters", "mtp_verified_positions", "mtp_refwd_iters",
        "mtp_full_rejects", "mtp_batched_iters", "mtp_production_iters",
        "mtp_draft_ms", "mtp_verify_ms", "mtp_refwd_ms", "mtp_iter_ms",
        "mtp_avg_iter_ms", "mtp_forwardn_ms", "mtp_forwardn_layers_ms",
        "mtp_forwardn_matmuln_ms", "mtp_forwardn_delta_ms", "mtp_forwardn_moe_route_ms",
        "mtp_forwardn_moe_io_ms", "mtp_forwardn_moe_expert_ms",
        "mtp_forwardn_moe_shared_ms",
    ]:
        value = getattr(args, key)
        if value is not None:
            pairs.append((key, value))
    if args.tok_per_sec is not None:
        pairs.insert(0, ("decode_tok_per_sec", args.tok_per_sec))
    return pairs


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description="Append MTP experiment metrics to assets/api_perf_log.tsv")
    ap.add_argument("--log", default=os.path.join(root, "assets", "api_perf_log.tsv"))
    ap.add_argument("--model", default="Qwen-Qwen36-35B-A3B")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--server-mode", default="mtp_eval")
    ap.add_argument("--stream", default="true")
    ap.add_argument("--tool-mode", default="none")
    ap.add_argument("--reasoning", default="0")
    ap.add_argument("--temperature", default="0.7")
    ap.add_argument("--top-p", default="0.8")
    ap.add_argument("--top-k", default="20")
    ap.add_argument("--min-p", default="0.0")
    ap.add_argument("--presence-penalty", default="1.5")
    ap.add_argument("--repetition-penalty", default="1.0")
    ap.add_argument("--duration-ms", default="")
    ap.add_argument("--tok-per-sec", type=float)
    ap.add_argument("--status", default="pass")
    ap.add_argument("--notes", default="-")
    for key in [
        "generated_tokens", "mtp_acceptance_pct", "mtp_accepted", "mtp_drafts",
        "mtp_spec_iters", "mtp_verified_positions", "mtp_refwd_iters",
        "mtp_full_rejects", "mtp_batched_iters", "mtp_production_iters",
        "mtp_draft_ms", "mtp_verify_ms", "mtp_refwd_ms", "mtp_iter_ms",
        "mtp_avg_iter_ms", "mtp_forwardn_ms", "mtp_forwardn_layers_ms",
        "mtp_forwardn_matmuln_ms", "mtp_forwardn_delta_ms", "mtp_forwardn_moe_route_ms",
        "mtp_forwardn_moe_io_ms", "mtp_forwardn_moe_expert_ms",
        "mtp_forwardn_moe_shared_ms",
    ]:
        ap.add_argument(f"--{key.replace('_', '-')}", type=float)
    args = ap.parse_args()

    pairs = metric_pairs(args)
    if not pairs:
        print("record_mtp_eval.py: no metrics provided", file=sys.stderr)
        return 2

    ensure_header(args.log)
    fallback = latest_metadata(args.log)
    host = socket.gethostname()
    hw_model = sysctl("hw.model") or fallback.get("hw_model") or "unknown"
    ram = ram_gib() or fallback.get("ram_gib", "")
    cpu = cpu_summary()
    if cpu == "?p ?e ?g" and fallback.get("cpu_summary"):
        cpu = fallback["cpu_summary"]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "timestamp": timestamp,
        "branch": git_branch(root),
        "commit": git_commit(root),
        "hostname": host or fallback.get("hostname") or "unknown",
        "hw_model": hw_model,
        "ram_gib": ram,
        "cpu_summary": cpu,
        "model": args.model,
        "server_mode": args.server_mode,
        "scenario": args.scenario,
        "endpoint": args.endpoint,
        "stream": args.stream,
        "tool_mode": args.tool_mode,
        "reasoning": args.reasoning,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "duration_ms": args.duration_ms,
        "tok_per_sec": args.tok_per_sec if args.tok_per_sec is not None else "",
        "status": args.status,
        "notes": args.notes,
    }
    with open(args.log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        for metric, value in pairs:
            row = dict(base)
            row["metric_type"] = metric
            row["metric_value"] = value
            writer.writerow(row)
    print(f"appended {len(pairs)} MTP metric rows to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
