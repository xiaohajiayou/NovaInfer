#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _median(vals: list[float]) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot TP performance experiment matrix.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tp_perf_plots"))
    args = parser.parse_args()

    rows = _load_rows(args.input_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    by_case: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        cfg = row.get("run_config", {}) or {}
        model_alias = str(row.get("model_alias", cfg.get("model_alias", "unknown")))
        size = "small" if "small" in str(row.get("case", "")) else "large"
        tp_size = int(row.get("tp_size", cfg.get("tp_size", 1)))
        result = row.get("result", {}) or {}
        throughput = float(result.get("global_throughput", result.get("actual_tokens_per_sec", 0.0)) or 0.0)
        grouped[(model_alias, size, tp_size)].append(throughput)
        by_case[(model_alias, size, tp_size)].append(throughput)

    for model_alias in sorted({k[0] for k in grouped.keys()}):
        for size in ("small", "large"):
            tp_sizes = sorted({k[2] for k in grouped.keys() if k[0] == model_alias and k[1] == size})
            if not tp_sizes:
                continue
            medians = [_median(grouped[(model_alias, size, tp)]) for tp in tp_sizes]
            base = medians[0] if medians else 0.0
            speedups = [(m / base) if base > 0 else 0.0 for m in medians]
            effs = [(s / tp) if tp > 0 else 0.0 for s, tp in zip(speedups, tp_sizes)]

            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=140)
            axes[0].bar([str(tp) for tp in tp_sizes], medians)
            axes[0].set_title(f"{model_alias} {size}: throughput")
            axes[0].set_ylabel("tok/s")
            axes[0].grid(axis="y", linestyle="--", alpha=0.35)

            axes[1].plot(tp_sizes, speedups, marker="o")
            axes[1].set_title(f"{model_alias} {size}: speedup")
            axes[1].set_xlabel("tp_size")
            axes[1].grid(axis="y", linestyle="--", alpha=0.35)

            axes[2].plot(tp_sizes, effs, marker="o")
            axes[2].set_title(f"{model_alias} {size}: scaling efficiency")
            axes[2].set_xlabel("tp_size")
            axes[2].set_ylim(0, 1.1)
            axes[2].grid(axis="y", linestyle="--", alpha=0.35)

            fig.tight_layout()
            fig.savefig(args.out_dir / f"{model_alias}_{size}_tp_scaling.png")
            plt.close(fig)

    print(f"[plot_tp] wrote png files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
