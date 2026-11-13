"""Runs real load tests against in-process mock servers, one per policy."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn
import yaml

from servers.mock import make_app
from voiceagent import loadgen
from voiceagent.analysis.calibrate import fit
from voiceagent.analysis.common import load_run
from voiceagent.analysis.summarize import summarize_run

ROOT = Path(__file__).resolve().parents[1]
SERVERS = {  # port: (role, base_s, per_inflight_s, tokens_per_s)
    9501: ("asr", 0.02, 0.002, 1), 9502: ("asr", 0.08, 0.005, 1),
    9511: ("tts", 0.02, 0.002, 1),
    9521: ("llm", 0.05, 0.005, 200), 9531: ("llm", 0.15, 0.01, 100), 9541: ("llm", 0.25, 0.01, 80),
}


def write_config(tmp_path, filler_mode="bank", max_wait_s=4.0, turbo=True, turbo_base_s=1.4):
    cfg = yaml.safe_load((ROOT / "configs/local_mock.yaml").read_text())
    cfg["filler_mode"] = filler_mode
    cfg["max_wait_s"] = max_wait_s
    cfg["latency"]["llm_turbo_first_sentence"]["base_s"] = turbo_base_s
    cfg["services"] = {
        "asr_small": ["http://127.0.0.1:9501"], "asr_large": ["http://127.0.0.1:9502"],
        "tts": ["http://127.0.0.1:9511"],
        "llm_small": {"model": "s", "urls": ["http://127.0.0.1:9521"]},
        "llm_large": {"model": "l", "urls": ["http://127.0.0.1:9531"]},
    }
    if turbo:
        cfg["services"]["llm_turbo"] = {"model": "t", "urls": ["http://127.0.0.1:9541"]}
    path = tmp_path / f"cfg_{filler_mode}_{max_wait_s}_{turbo}_{turbo_base_s}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


async def with_servers(router, coro):
    p_escalate, p_expert = router
    servers = []
    for port, (role, base, per, tps) in SERVERS.items():
        app = make_app(role, base, per, tps, p_escalate, seed=port, chars_per_s=300.0, p_expert=p_expert)
        s = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        servers.append((s, asyncio.create_task(s.serve())))
    while not all(s.started for s, _ in servers):
        await asyncio.sleep(0.02)
    try:
        return await coro
    finally:
        for s, task in servers:
            s.should_exit = True
        await asyncio.gather(*(t for _, t in servers))


@pytest.fixture(scope="module")
def eval_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    subprocess.run([sys.executable, str(ROOT / "scripts/data/make_mock_eval.py"), "--out-dir", str(out),
                    "--count", "10"], check=True, capture_output=True)
    return out / "eval.jsonl"


def run(tmp_path, eval_set, policy, p_escalate=1.0, p_expert=0.0, n=4, **cfg_kwargs):
    out = tmp_path / f"{policy}.jsonl"
    coro = loadgen.run(str(write_config(tmp_path, **cfg_kwargs)), str(eval_set), n, policy,
                       duration=2.0, warmup=0.0, think=0.0, out=str(out))
    asyncio.run(with_servers((p_escalate, p_expert), coro))
    meta, records = load_run(out)
    assert records and all(r["error"] is None for r in records), [r["error"] for r in records]
    return meta, records


@pytest.mark.parametrize("policy", ["always_small", "always_large", "always_turbo"])
def test_baselines(tmp_path, eval_set, policy):
    meta, records = run(tmp_path, eval_set, policy)
    size = policy.removeprefix("always_")
    for r in records:
        assert r["route"] == "plain" and r["llm_model"] == size
        assert r["asr_model"] == ("small" if size == "small" else "large")
        assert 0 < r["ttfa"] <= r["ttsa"] < r["t_playback_end"] - r["t_start"]
        assert r["answer"].startswith("Asthma")
    assert summarize_run(meta, records)["turns"] == len(records)


def test_adaptive_plays_filler_before_large_answer(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive", p_escalate=1.0)
    for r in records:
        assert r["route"] == "escalate_large" and r["filler_text"]
        assert r["ttfa"] < r["ttsa"]
        assert r["router_raw"].startswith("ESCALATE")
    # noisy clips (SNR < 10 dB) go to the large ASR, clean ones to the small one
    assert {r["asr_model"] for r in records if r["snr_db"] is None} == {"small"}
    assert {r["asr_model"] for r in records if r["snr_db"] is not None and r["snr_db"] <= 5} == {"large"}


def test_expert_goes_to_turbo_with_filler(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive", p_escalate=0.0, p_expert=1.0)
    for r in records:
        assert r["router_decision"] == "EXPERT" and r["route"] == "escalate_turbo" and r["llm_model"] == "turbo"
        assert r["filler_text"] and r["ttfa"] < r["ttsa"]
        assert any(s["key"] == "llm_turbo_first_sentence" for s in r["stages"])


def test_expert_falls_back_to_large_when_turbo_is_too_slow(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive", p_escalate=0.0, p_expert=1.0, turbo_base_s=10.0)
    assert all(r["router_decision"] == "EXPERT" and r["route"] == "escalate_large" for r in records)


def test_expert_without_turbo_pool_uses_large(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive", p_escalate=0.0, p_expert=1.0, turbo=False)
    assert all(r["route"] == "escalate_large" for r in records)


def test_always_turbo_needs_turbo_service(tmp_path, eval_set):
    with pytest.raises(ValueError, match="llm_turbo"):
        asyncio.run(loadgen.run(str(write_config(tmp_path, turbo=False)), str(eval_set), 1, "always_turbo",
                                duration=1.0, warmup=0.0, think=0.0, out=str(tmp_path / "x.jsonl")))


def test_adaptive_answers_with_small_when_not_escalated(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive", p_escalate=0.0)
    for r in records:
        assert r["route"] == "answer" and not r["filler_text"] and r["ttfa"] == r["ttsa"]
        assert r["answer"].startswith("Asthma")


def test_generated_filler(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive", filler_mode="generated")
    assert all(r["filler_text"] == "Let me look into that for you, one moment." for r in records)


def test_no_filler_ablation_and_hedge_fallback(tmp_path, eval_set):
    _, records = run(tmp_path, eval_set, "adaptive_no_filler")
    assert all(r["route"] == "escalate_large" and not r["filler_text"] and r["ttfa"] == r["ttsa"] for r in records)
    _, records = run(tmp_path, eval_set, "adaptive", max_wait_s=0.01)
    assert all(r["route"] == "escalate_hedge" and r["llm_model"] == "small" for r in records)


def test_calibration_fits_every_stage(tmp_path, eval_set):
    for policy in ("always_small", "always_large", "always_turbo"):
        run(tmp_path, eval_set, policy, n=6)
    params = fit([str(tmp_path / f"always_{s}.jsonl") for s in ("small", "large", "turbo")])
    assert set(params) == {"asr_small", "asr_large", "llm_small_first_sentence", "llm_large_first_sentence",
                           "llm_turbo_first_sentence", "tts"}
    small, large, turbo = (params[f"llm_{s}_first_sentence"]["base_s"] for s in ("small", "large", "turbo"))
    assert small < large < turbo
    json.dumps(params)


def test_ttsa_waits_for_filler_playback(tmp_path, eval_set, monkeypatch):
    from voiceagent.filler import FillerBank

    filler_duration = 2.0
    monkeypatch.setattr(
        FillerBank,
        "pick",
        lambda self: ("One moment while I check.", filler_duration),
    )

    _, records = run(
        tmp_path, eval_set, "adaptive", n=1, p_escalate=1.0
    )

    for rec in records:
        assert rec["route"] == "escalate_large"
        assert rec["filler_text"] == "One moment while I check."

        filler_end = rec["t_first_audio"] + filler_duration
        assert rec["t_first_answer_audio"] >= filler_end, (
            "Answer playback cannot start before filler ends: "
            f"answer={rec['t_first_answer_audio']:.3f}, "
            f"filler_end={filler_end:.3f}"
        )
        assert rec["ttsa"] == pytest.approx(
            rec["t_first_answer_audio"] - rec["t_start"]
        )
