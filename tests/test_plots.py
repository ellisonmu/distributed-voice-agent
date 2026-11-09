import csv
import json

import pytest

pytest.importorskip("matplotlib")

from voiceagent.analysis import plots  # noqa: E402


def test_all_figures_render(tmp_path):
    summary = tmp_path / "summary.csv"
    fields = ["policy", "n", "ttfa_p95", "ttsa_p95", "quality", "stalled", "llm_small", "llm_large", "llm_turbo", "hedged"]
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, policy in enumerate(plots.POLICY_ORDER):
            for n in (1, 8, 32):
                w.writerow({"policy": policy, "n": n, "ttfa_p95": 0.5 + i * 0.4 + n / 50, "ttsa_p95": 0.6 + i * 0.4,
                            "quality": "" if policy == "adaptive_no_filler" else 0.5 + i / 10, "stalled": n / 100,
                            "llm_small": 0.6, "llm_large": 0.2, "llm_turbo": 0.15, "hedged": 0.05})
    run = tmp_path / "ddp"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"strategy": "ddp", "world_size": 4, "peak_mem_gb": 30.0,
                                                  "mean_tokens_per_s": 4000.0}))
    rows = plots.load_summary([str(summary)])
    plots.latency_vs_n(rows, tmp_path / "a.png", 1.0)
    assert plots.quality_vs_latency(rows, tmp_path / "b.png", 1.0)
    assert plots.routing_mix(rows, tmp_path / "c.png")
    assert plots.stalls(rows, tmp_path / "d.png")
    assert plots.training([str(run)], tmp_path / "e.png")
    assert all((tmp_path / f"{c}.png").stat().st_size > 10_000 for c in "abcde")
