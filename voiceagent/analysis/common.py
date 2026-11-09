from __future__ import annotations

import json
from pathlib import Path


def load_run(path: str | Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    with open(path) as f:
        records = [json.loads(line) for line in f]
    return meta, records


def measured_turns(meta: dict, records: list[dict]) -> list[dict]:
    """Turns that started inside the measurement window, after warm-up."""
    return [
        r for r in records
        if r["t_start"] is not None and meta["warmup_s"] <= r["t_start"] < meta["duration_s"]
    ]
