"""Fits the scheduler's latency model from logged runs (E1): for each stage, a least-squares
line latency = base_s + per_inflight_s * inflight. Write the output JSON and point the
config's `latency:` at it."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from voiceagent.analysis.common import load_run, measured_turns
from voiceagent.config import LATENCY_KEYS, TURBO_LATENCY_KEY


def fit(paths: list[str]) -> dict[str, dict[str, float]]:
    samples: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for path in paths:
        meta, records = load_run(path)
        for r in measured_turns(meta, records):
            for s in r["stages"]:
                samples[s["key"]].append((s["inflight"], s["t1"] - s["t0"]))
    params = {}
    for key in (*LATENCY_KEYS, TURBO_LATENCY_KEY):
        if not samples[key]:
            continue
        x = np.array([i for i, _ in samples[key]], dtype=float)
        y = np.array([d for _, d in samples[key]], dtype=float)
        if np.ptp(x) == 0:
            base, slope = float(y.mean()), 0.0
        else:
            slope, base = (float(v) for v in np.polyfit(x, y, 1))
            if slope < 0:
                base, slope = float(y.mean()), 0.0
        params[key] = {"base_s": base, "per_inflight_s": slope, "samples": len(y)}
    return params


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    params = fit([x for x in a.runs if not x.endswith(".meta.json")])
    missing = set(LATENCY_KEYS) - set(params)
    if missing:
        raise SystemExit(f"no samples for {sorted(missing)}: include always_small and always_large runs "
                         "(and always_turbo on the 16-GPU layout)")
    Path(a.out).write_text(json.dumps(params, indent=2))
    for key, v in params.items():
        print(f"{key:28s} base={v['base_s']:.3f}s  per_inflight={v['per_inflight_s']:.4f}s  n={v['samples']}")


if __name__ == "__main__":
    main()
