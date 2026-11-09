from pathlib import Path

import pytest

from voiceagent.config import load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
ENV = {"NODE1": "nid001", "NODE2": "nid002", "NODE3": "nid003", "DATA_DIR": "/data", "LLM_SMALL": "small",
       "LLM_ROUTER": "router", "LLM_LARGE": "large", "LLM_TURBO": "turbo", "FILLER_MODE": "bank", "RAG_KIND": "faiss"}


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_config_loads(path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = load_config(path)
    assert all(u.startswith("http://") and "$" not in u for u in cfg.all_urls())


def test_missing_env_var_is_an_error(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("NODE2")
    with pytest.raises(ValueError, match="NODE2"):
        load_config(Path(__file__).resolve().parents[1] / "configs/cluster_8gpu.yaml")
