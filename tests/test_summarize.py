"""ASR report checks use fixed text fixtures, not model benchmark results."""
import csv
import json
import sys

import pytest

pytest.importorskip("jiwer")

from voiceagent.analysis import summarize  # noqa: E402
from voiceagent.records import TurnRecord  # noqa: E402


@pytest.fixture
def meta():
    return {"run_id": "adaptive_n4", "policy": "adaptive", "n_callers": 4,
            "warmup_s": 2.0, "duration_s": 10.0}


def turn(reference="What is asthma?", transcript="What is asthma?", **changes):
    rec = TurnRecord(
        run_id="adaptive_n4", policy="adaptive", n_callers=4, caller_id=0,
        turn_idx=0, question_id="q0", question_text=reference, snr_db=0.0,
        transcript=transcript, asr_model="small", t_start=3.0, ttfa=0.5, ttsa=0.5,
    ).to_dict()
    rec.update(changes)
    return rec


def test_asr_groups_use_injected_snr_and_keep_models_separate(meta):
    unknown = turn()
    del unknown["snr_db"]
    records = [
        turn(snr_db=None, est_snr_db=0.0),
        turn(snr_db=0.0, est_snr_db=20.0),
        turn(snr_db=0.0, est_snr_db=40.0),
        turn(asr_model="large", snr_db=0.0),
        turn(snr_db=5.0),
        turn(snr_db=10.0),
        unknown,
    ]
    rows = summarize.summarize_asr(meta, records)
    grouped = {(r["asr_model"], r["noise_condition"]): r for r in rows}
    assert set(grouped) == {
        ("small", "clean"), ("small", "0dB"), ("large", "0dB"),
        ("small", "5dB"), ("small", "10dB"), ("small", "unknown"),
    }
    assert grouped["small", "0dB"]["turns"] == 2
    assert grouped["small", "0dB"]["est_snr_p50"] == 30.0
    assert grouped["large", "0dB"]["est_snr_p50"] is None
    assert [r["noise_condition"] for r in rows if r["asr_model"] == "small"] == [
        "clean", "0dB", "5dB", "10dB", "unknown",
    ]
    assert all(r["run_id"] == meta["run_id"] and r["n"] == 4 for r in rows)


def test_asr_wer_is_word_weighted_not_an_average_of_turns(meta):
    records = [turn("a b c d", "a b c x"), turn("e", "")]
    row, = summarize.summarize_asr(meta, records)
    # One substitution + one deletion over five words: 2/5, not (1/4 + 1)/2.
    assert row["reference_words"] == 5
    assert (row["substitutions"], row["deletions"], row["insertions"]) == (1, 1, 0)
    assert row["wer"] == pytest.approx(0.4)
    assert row["wer"] == pytest.approx(summarize.summarize_run(meta, records)["wer"])


def test_asr_excludes_warmup_errors_and_turns_outside_window(meta):
    records = [
        turn(t_start=1.999, transcript="wrong"),
        turn(t_start=2.0),
        turn(t_start=9.999),
        turn(t_start=10.0, transcript="wrong"),
        turn(t_start=None, transcript="wrong"),
        turn(t_start=4.0, transcript="wrong", error="request failed"),
    ]
    row, = summarize.summarize_asr(meta, records)
    assert row["turns"] == row["scored_turns"] == 2
    assert row["wer"] == 0.0


def test_asr_normalization_and_empty_hypotheses(meta):
    records = [turn("...", "noise"), turn("Hello, world!", ""),
               turn("It's asthma.", "it's ASTHMA")]
    row, = summarize.summarize_asr(meta, records)
    assert row["turns"] == 3 and row["scored_turns"] == 2
    assert row["reference_words"] == 4 and row["deletions"] == 2
    assert row["wer"] == pytest.approx(0.5)


def test_asr_group_with_no_scorable_references_reports_no_wer(meta):
    row, = summarize.summarize_asr(meta, [turn("...", "noise")])
    assert row["turns"] == 1 and row["scored_turns"] == 0
    assert row["reference_words"] == 0 and row["wer"] is None
    assert summarize.summarize_asr(meta, []) == []


def test_asr_counts_insertions_without_clamping_wer(meta):
    row, = summarize.summarize_asr(meta, [turn("a", "a b c")])
    assert row["reference_words"] == 1 and row["insertions"] == 2
    assert row["wer"] == 2.0


def write_run(tmp_path, meta, records):
    path = tmp_path / f"{meta['run_id']}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path


@pytest.mark.parametrize("t_start,expected_groups", [(3.0, 2), (1.0, 0)])
def test_cli_writes_both_reports_and_headers(tmp_path, meta, monkeypatch, t_start, expected_groups):
    path = write_run(tmp_path, meta, [turn(t_start=t_start), turn(t_start=t_start, asr_model="large")])
    summary_path = tmp_path / "reports" / "summary.csv"
    asr_path = tmp_path / "reports" / "asr.csv"
    monkeypatch.setattr(sys, "argv", [
        "summarize", str(path), str(path.with_suffix(".meta.json")),
        "--out", str(summary_path), "--asr-out", str(asr_path),
    ])
    summarize.main()
    with summary_path.open() as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == 1 and "ttfa_p95" in summary_rows[0]
    with asr_path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(summarize.ASR_COLUMNS)
        rows = list(reader)
    assert len(rows) == expected_groups
    if rows:
        assert {r["asr_model"] for r in rows} == {"small", "large"}
        assert all(float(r["wer"]) == 0.0 and r["policy"] == "adaptive" for r in rows)


def test_cli_keeps_runs_with_different_policies_and_loads_separate(tmp_path, meta, monkeypatch):
    first = write_run(tmp_path, meta, [turn("a", "a")])
    other_meta = {**meta, "run_id": "always_small_n8", "policy": "always_small", "n_callers": 8}
    second = write_run(tmp_path, other_meta, [turn("a", "b", run_id="always_small_n8",
                                                policy="always_small", n_callers=8)])
    out = tmp_path / "asr.csv"
    monkeypatch.setattr(sys, "argv", ["summarize", str(first), str(second), "--asr-out", str(out)])
    summarize.main()
    with out.open() as f:
        rows = {r["run_id"]: r for r in csv.DictReader(f)}
    assert len(rows) == 2
    assert rows["adaptive_n4"]["policy"] == "adaptive" and rows["adaptive_n4"]["n"] == "4"
    assert float(rows["adaptive_n4"]["wer"]) == 0.0
    assert rows["always_small_n8"]["policy"] == "always_small" and rows["always_small_n8"]["n"] == "8"
    assert float(rows["always_small_n8"]["wer"]) == 1.0


def test_cli_reports_missing_jiwer(tmp_path, meta, monkeypatch, capsys):
    path = write_run(tmp_path, meta, [turn()])
    out = tmp_path / "asr.csv"
    monkeypatch.setitem(sys.modules, "jiwer", None)
    monkeypatch.setattr(sys, "argv", ["summarize", str(path), "--asr-out", str(out)])
    with pytest.raises(SystemExit) as exc:
        summarize.main()
    assert exc.value.code == 2
    assert "requires jiwer" in capsys.readouterr().err
    assert not out.exists()
