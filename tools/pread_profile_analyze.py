#!/usr/bin/env python3
"""Analyse a FLASHCHAT_PREAD_PROFILE TSV: expert-streaming cache-hit vs miss.

The engine records one raw row per expert pread (no classification). This script
splits hits (served from OS page cache / RAM) from misses (real SSD I/O) by an
effective-bandwidth threshold and reports whether a userspace expert cache could
buy anything that the kernel page cache isn't already giving for free.

Usage:
    python3 tools/pread_profile_analyze.py debug/pread.tsv
    python3 tools/pread_profile_analyze.py debug/pread.tsv --hit-mbps 2000 --plot out.png

--plot needs matplotlib (pip install matplotlib in a venv/uv env; never system
Python). Without it you still get the full text report.
"""
import argparse
import csv
import statistics as st
import sys


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                rows.append({
                    "t_rel_us": float(r["t_rel_us"]),
                    "dur_us": float(r["dur_us"]),
                    "expert_idx": int(r["expert_idx"]),
                    "offset": int(r["offset"]),
                    "size": int(r["size"]),
                    "mb_per_s": float(r["mb_per_s"]),
                    # gid = globally-unique expert key (layer*num_experts+expert).
                    # Absent in pre-gid TSVs; -1 when the read path didn't know it.
                    "gid": int(r.get("gid", -1)),
                })
            except (KeyError, ValueError):
                continue
    return rows


def cache_key(r):
    """Cache-unit identity. Prefer gid (distinguishes layers); fall back to the
    raw offset for legacy TSVs (which collapses per-layer files — undercounts)."""
    return ("g", r["gid"]) if r["gid"] >= 0 else ("o", r["offset"])


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(label, durs):
    if not durs:
        print(f"  {label:<6} count=0")
        return
    print(f"  {label:<6} count={len(durs):>7}  "
          f"dur_us mean={st.mean(durs):8.1f} p50={pct(durs,50):8.1f} "
          f"p90={pct(durs,90):8.1f} p99={pct(durs,99):8.1f} max={max(durs):9.1f}  "
          f"total_ms={sum(durs)/1000:9.1f}")


def simulate_cache(rows, cache_bytes, hit_mbps):
    """Replay the exact pread access trace through a byte-capacity LRU cache.

    Each unique file offset is a cacheable blob of `size` bytes (the real read
    unit — the gate/up/down weight+scale chunks of an expert). This is an
    assumption-free model: it doesn't need to decode expert/layer identity, it
    just asks "if `cache_bytes` of RAM were pinned and never evicted by memory
    pressure, which of these reads would already be resident?"

    Returns (hits, misses, miss_bytes). A cold first-touch of a blob is always a
    miss (no cache can serve bytes it has never seen).
    """
    from collections import OrderedDict
    cache = OrderedDict()          # key -> size, ordered by recency
    cur = 0
    hits = misses = 0
    miss_bytes = 0
    for r in rows:
        key, sz = cache_key(r), r["size"]
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        misses += 1
        miss_bytes += sz
        # admit, evicting LRU until it fits (skip blobs larger than the cache)
        if sz <= cache_bytes:
            cache[key] = sz
            cur += sz
            while cur > cache_bytes:
                _, esz = cache.popitem(last=False)
                cur -= esz
    return hits, misses, miss_bytes


def report_simulation(rows, sizes_gb, hit_mean_us, miss_mean_us, observed_total_ms):
    from collections import Counter
    freq = Counter(cache_key(r) for r in rows)
    blob_size = {cache_key(r): r["size"] for r in rows}
    distinct = len(freq)
    working_set = sum(blob_size.values())
    keyed_by_gid = any(r["gid"] >= 0 for r in rows)
    print()
    print("=== pinned-cache simulation (LRU over the exact access trace) ===")
    if not keyed_by_gid:
        print("  WARNING: no gid column (legacy TSV) — keyed by per-layer-file offset,")
        print("  which COLLAPSES layers and UNDERCOUNTS the working set by ~num_layers.")
        print("  Re-capture with the current engine for a correct working set.")
    print(f"distinct experts: {distinct}   working set: {working_set/2**30:.2f} GiB   "
          f"accesses: {len(rows)}   (keyed by {'gid' if keyed_by_gid else 'offset'})")

    # Skew: do the hottest blobs dominate accesses? (Decides if pinning helps.)
    ordered = sorted(freq.values(), reverse=True)
    total_acc = sum(ordered)
    for frac in (0.1, 0.25, 0.5):
        n = max(1, int(distinct * frac))
        cov = sum(ordered[:n]) / total_acc
        print(f"  top {frac*100:>4.0f}% of blobs ({n:>6}) serve {cov*100:5.1f}% of accesses")

    print(f"\n  {'cache':>7}  {'hit%':>6}  {'miss%':>6}  {'est pread ms':>13}  {'vs measured':>12}")
    print(f"  {'(none)':>7}  {'—':>6}  {'—':>6}  {observed_total_ms:>10.0f} ms  {'100%':>12}  measured")
    for gb in sizes_gb:
        cap = int(gb * 2**30)
        hits, misses, _ = simulate_cache(rows, cap, 0)
        n = hits + misses
        hitp = 100 * hits / n
        # Cached hits pay the observed HIT mean, the rest the observed MISS mean.
        # Rough (pread overlaps GPU compute) but a fair relative comparison.
        est_ms = (hits * hit_mean_us + misses * miss_mean_us) / 1000
        tag = "  <- full working set fits" if cap >= working_set else ""
        rel = 100 * est_ms / observed_total_ms if observed_total_ms else 0
        print(f"  {gb:>5.1f}G  {hitp:>5.1f}%  {100-hitp:>5.1f}%  {est_ms:>10.0f} ms  "
              f"{rel:>10.0f}%{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--hit-mbps", type=float, default=2000.0,
                    help="effective bandwidth (MB/s) above which a pread is a "
                         "page-cache hit (default 2000; SSD misses sit far below)")
    ap.add_argument("--plot", metavar="PNG", help="write a histogram PNG")
    ap.add_argument("--simulate", action="store_true",
                    help="simulate a pinned LRU expert cache over the access "
                         "trace and report predicted hit rate vs cache size")
    ap.add_argument("--cache-gb", default="0.5,1,2,4,8,16",
                    help="comma-separated cache sizes (GiB) to sweep with --simulate")
    args = ap.parse_args()

    rows = load(args.tsv)
    if not rows:
        print(f"no usable records in {args.tsv}", file=sys.stderr)
        return 1

    hits = [r for r in rows if r["mb_per_s"] >= args.hit_mbps]
    miss = [r for r in rows if r["mb_per_s"] < args.hit_mbps]
    span_s = (max(r["t_rel_us"] for r in rows) - min(r["t_rel_us"] for r in rows)) / 1e6

    hit_ms = sum(r["dur_us"] for r in hits) / 1000
    miss_ms = sum(r["dur_us"] for r in miss) / 1000
    total_ms = hit_ms + miss_ms
    bytes_total = sum(r["size"] for r in rows)

    print(f"file        : {args.tsv}")
    print(f"records     : {len(rows)}   wall span: {span_s:.1f}s   "
          f"bytes read: {bytes_total/1048576:.1f} MiB")
    print(f"hit threshold : >= {args.hit_mbps:.0f} MB/s effective")
    print()
    summarize("HIT", [r["dur_us"] for r in hits])
    summarize("MISS", [r["dur_us"] for r in miss])
    print()
    print(f"hit rate (by count): {100*len(hits)/len(rows):5.1f}%   "
          f"({len(hits)}/{len(rows)})")
    if total_ms > 0:
        print(f"time in misses     : {100*miss_ms/total_ms:5.1f}%   "
              f"({miss_ms:.0f} of {total_ms:.0f} ms of pread wall time)")

    # Re-route locality: how many preads hit an already-seen expert? That is the
    # ceiling on what ANY expert cache (page cache or userspace) can serve. Keyed
    # by gid (correct) when present, else the collapsing offset bucket.
    seen, reroutes = set(), 0
    for r in rows:
        e = cache_key(r)
        if e in seen:
            reroutes += 1
        seen.add(e)
    routed = rows
    if routed:
        print(f"re-route rate      : {100*reroutes/len(routed):5.1f}%   "
              f"({reroutes} re-routes / {len(seen)} distinct experts)")

    print()
    if total_ms > 0 and miss_ms / total_ms < 0.10:
        print("VERDICT: misses are a small slice of pread time — the OS page cache is "
              "already keeping hot experts resident. A userspace expert cache would "
              "mostly duplicate it. Little to gain.")
    elif routed and reroutes / len(routed) < 0.10:
        print("VERDICT: low re-route rate — most preads are first-touch experts that "
              "no cache can serve. Caching helps only the re-route fraction.")
    else:
        print("VERDICT: misses dominate AND re-routes are frequent — page cache is "
              "likely thrashing under memory pressure. An mlock-pinned whole-expert "
              "LRU (size to free RAM) is worth prototyping. Prefixes won't help.")

    if args.simulate:
        try:
            sizes_gb = [float(x) for x in args.cache_gb.split(",") if x.strip()]
        except ValueError:
            print("--cache-gb must be comma-separated numbers", file=sys.stderr)
            return 1
        hit_mean = st.mean([r["dur_us"] for r in hits]) if hits else 0.0
        miss_mean = st.mean([r["dur_us"] for r in miss]) if miss else 0.0
        report_simulation(rows, sizes_gb, hit_mean, miss_mean, total_ms)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n--plot needs matplotlib (install in a venv/uv env)", file=sys.stderr)
            return 0
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].hist([r["mb_per_s"] for r in rows], bins=80)
        ax[0].axvline(args.hit_mbps, color="r", ls="--", label=f"{args.hit_mbps:.0f} MB/s")
        ax[0].set_xlabel("effective MB/s"); ax[0].set_ylabel("preads"); ax[0].legend()
        ax[0].set_title("pread effective bandwidth (hit/miss split)")
        ax[1].scatter([r["t_rel_us"]/1e6 for r in rows],
                      [r["dur_us"] for r in rows], s=3, alpha=0.4)
        ax[1].set_yscale("log"); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("pread dur_us (log)")
        ax[1].set_title("pread latency over the run")
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"\nwrote plot -> {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
