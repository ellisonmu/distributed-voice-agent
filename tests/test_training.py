"""Runs training/train_router.py under torchrun with 2 CPU processes on a tiny random Llama,
so the DDP/FSDP plumbing, loss masking and adapter saving are exercised without a GPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from training.data import IGNORE, collate, encode_example  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("<unk> <pad> <eos> system user assistant : ANSWER ESCALATE EXPERT Asthma is a lung condition . "
         "Let me look into that What asthma ? reference passages one moment").split()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    hf.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
                        "{% if add_generation_prompt %}assistant : {% endif %}")
    hf.save_pretrained(d)
    cfg = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    LlamaForCausalLM(cfg).save_pretrained(d)

    data = d / "data"
    data.mkdir()
    rows = [
        {"messages": [{"role": "system", "content": "reference passages"}, {"role": "user", "content": "What is asthma ?"}],
         "target": "ANSWER Asthma is a lung condition ."},
        {"messages": [{"role": "system", "content": "reference passages"}, {"role": "user", "content": "What asthma ?"}],
         "target": "EXPERT : Let me look into that one moment ."},
    ] * 8
    for name in ("train", "val"):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d, hf


def test_only_the_target_is_learned(tiny):
    _, tok = tiny
    msgs = [{"role": "user", "content": "What is asthma ?"}]
    ex = encode_example(tok, msgs, "ANSWER Asthma is a lung condition .", max_len=64)
    learned = [t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE]
    assert tok.decode(learned).split() == "ANSWER Asthma is a lung condition . <eos>".split()
    assert encode_example(tok, msgs, "ANSWER Asthma", max_len=5) is None
    batch = collate([ex, {"input_ids": [1, 2], "labels": [IGNORE, 2]}], pad_id=1)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][1].sum() == 2


def _train(d, out, strategy, port, *extra):
    # Pin gloo to loopback: on macOS it otherwise hangs resolving the host name.
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else os.environ.get("GLOO_SOCKET_IFNAME", "lo")}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
           "--master_addr=127.0.0.1", f"--master_port={port}",
           str(ROOT / "training/train_router.py"), "--strategy", strategy, "--model", str(d),
           "--data-dir", str(d / "data"), "--out", str(out), "--grad-accum", "2", "--log-every", "1",
           "--max-steps", "3", "--lr", "1e-2", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)


@pytest.fixture(scope="module")
def trained(tiny, tmp_path_factory):
    """Trains with both strategies on the same data and seed."""
    d, _ = tiny
    outs = {}
    for port, strategy in ((29621, "ddp"), (29622, "fsdp")):
        out = tmp_path_factory.mktemp(strategy)
        proc = _train(d, out, strategy, port)
        assert proc.returncode == 0, proc.stderr[-3000:]
        outs[strategy] = out
    return outs


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_train_two_processes(trained, strategy):
    out = trained[strategy]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["strategy"] == strategy and summary["world_size"] == 2 and summary["steps"] == 3
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["tokens_per_s"] > 0 for r in rows)


def test_ddp_and_fsdp_do_the_same_math(trained):
    loss = {s: [json.loads(x)["loss"] for x in (trained[s] / "metrics.jsonl").read_text().splitlines()] for s in trained}
    assert loss["ddp"] == pytest.approx(loss["fsdp"], rel=1e-4)


def test_adapters_have_the_same_format_and_load(tiny, trained):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    d, _ = tiny
    weights = {s: load_file(str(trained[s] / "adapter" / "adapter_model.safetensors")) for s in trained}
    assert weights["ddp"] and set(weights["ddp"]) == set(weights["fsdp"])
    for s in trained:
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(d), str(trained[s] / "adapter"))
        loaded = {k: v for k, v in model.state_dict().items() if "lora_B" in k}
        assert loaded and any(v.abs().sum() > 0 for v in loaded.values())


@pytest.mark.parametrize("strategy,port", [("ddp", 29631), ("fsdp", 29633)])
def test_killed_run_resumes_with_the_same_math(tiny, trained, tmp_path, strategy, port):
    from safetensors.torch import load_file

    d, _ = tiny
    killed = _train(d, tmp_path, strategy, port, "--ckpt-every", "1", "--fail-after-step", "2")
    assert killed.returncode != 0 and (tmp_path / "checkpoint" / "meta.json").exists()
    assert not (tmp_path / "summary.json").exists()
    resumed = _train(d, tmp_path, strategy, port + 1, "--ckpt-every", "1")
    assert resumed.returncode == 0, resumed.stderr[-3000:]
    assert "resuming from" in resumed.stdout

    def losses(out):
        return [json.loads(x)["loss"] for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert losses(tmp_path) == pytest.approx(losses(trained[strategy]), rel=1e-5)
    a = load_file(str(tmp_path / "adapter" / "adapter_model.safetensors"))
    b = load_file(str(trained[strategy] / "adapter" / "adapter_model.safetensors"))
    assert set(a) == set(b) and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    again = _train(d, tmp_path, strategy, port, "--ckpt-every", "1")
    assert again.returncode == 0 and "already finished" in again.stdout
