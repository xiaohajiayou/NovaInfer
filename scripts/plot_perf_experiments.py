#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _display_backend_name(backend: str) -> str:
    return "ours" if backend == "novainfer" else backend


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    m = n // 2
    if n % 2 == 1:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(1, len(vals))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ours-vs-vLLM perf experiment results.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("perf_plots"))
    args = parser.parse_args()

    rows = _load_rows(args.input_jsonl)
    if not rows:
        raise RuntimeError(f"no data in {args.input_jsonl}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_case_backend: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_case_backend_time: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_case_rep_backend: dict[tuple[str, int, str], float] = {}
    for r in rows:
        case = str(r.get("case", "unknown"))
        rep = int(r.get("repeat", -1))
        backend = str(r.get("backend", "unknown"))
        result = r.get("result", {}) or {}
        tps = float(result.get("actual_tokens_per_sec", result.get("tokens_per_sec", 0.0)) or 0.0)
        # Normalize runtime metric: prefer run_seconds when available.
        sec = float(result.get("run_seconds", result.get("seconds", 0.0)) or 0.0)
        by_case_backend[(case, backend)].append(tps)
        by_case_backend_time[(case, backend)].append(sec)
        by_case_rep_backend[(case, rep, backend)] = tps

    cases = sorted({k[0] for k in by_case_backend.keys()})
    backends = ["novainfer", "vllm"]

    # Figure 1: throughput (mean with min/max whiskers)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
    x = list(range(len(cases)))
    width = 0.35
    for bi, backend in enumerate(backends):
        means = []
        lows = []
        highs = []
        for c in cases:
            vals = by_case_backend.get((c, backend), [])
            if not vals:
                means.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
                continue
            m = _mean(vals)
            means.append(m)
            lows.append(max(0.0, m - min(vals)))
            highs.append(max(0.0, max(vals) - m))
        offset = (-0.5 + bi) * width
        xs = [xx + offset for xx in x]
        ax.bar(xs, means, width=width, label=_display_backend_name(backend))
        ax.errorbar(xs, means, yerr=[lows, highs], fmt="none", capsize=4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Tokens / second (actual)")
    ax.set_title("Throughput by case")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "throughput_by_case.png")
    plt.close(fig)

    # Figure 2: speedup distribution (novainfer / vllm) per repeat
    speedup_by_case: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        reps = sorted({rep for (case, rep, b) in by_case_rep_backend.keys() if case == c and b in backends})
        for rep in reps:
            n = by_case_rep_backend.get((c, rep, "novainfer"))
            v = by_case_rep_backend.get((c, rep, "vllm"))
            if n is None or v is None or v <= 0:
                continue
            speedup_by_case[c].append(float(n) / float(v))

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
    positions = list(range(1, len(cases) + 1))
    data = [speedup_by_case.get(c, []) for c in cases]
    ax.boxplot(data, positions=positions, widths=0.5, showmeans=True)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Speedup (ours / vLLM)")
    ax.set_title("Speedup distribution by case")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(args.out_dir / "speedup_by_case.png")
    plt.close(fig)

    # Figure 3: run time decomposition from medians
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
    x = list(range(len(cases)))
    width = 0.35
    for bi, backend in enumerate(backends):
        med_run = [_median(by_case_backend_time.get((c, backend), [])) for c in cases]
        offset = (-0.5 + bi) * width
        xs = [xx + offset for xx in x]
        ax.bar(xs, med_run, width=width, label=f"{_display_backend_name(backend)} run_seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Run seconds (median)")
    ax.set_title("Run time by case")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "run_seconds_by_case.png")
    plt.close(fig)

    # Console summary
    print("[plot] summary (median actual throughput)")
    for c in cases:
        n = _median(by_case_backend.get((c, "novainfer"), []))
        v = _median(by_case_backend.get((c, "vllm"), []))
        ratio = (n / v) if v > 0 else math.nan
        print(f"  case={c:>5s} ours={n:8.2f} vllm={v:8.2f} ratio={ratio:6.3f}x")
    print(f"[plot] wrote png files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
