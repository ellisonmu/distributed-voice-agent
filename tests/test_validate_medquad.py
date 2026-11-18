"""Integrity checks use tiny text fixtures; no downloaded dataset or model is used."""
import hashlib
import json
import sys

import pytest

from voiceagent.analysis import validate_medquad as audit


def row(key, doc, question, focus=""):
    return {"question_id": key, "doc_id": doc, "question": question,
            "answer": "Reference answer.", "focus": focus,
            "qtype": "information", "source": "fixture"}


@pytest.fixture
def splits(tmp_path):
    data = {
        "finetune": [row("t1", "doc-t", "Question one?"), row("t2", "doc-t", "Question two?")],
        "dev": [row("d1", "doc-d", "Question three?")],
        "eval": [row("e1", "doc-e", "Question four?")],
    }
    for name, rows in data.items():
        write(tmp_path, name, rows)
    return tmp_path


def write(directory, name, rows):
    (directory / f"{name}.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_valid_splits_allow_multiple_questions_from_one_document(splits):
    report = audit.validate_splits(splits, source_commit="fixture-revision")
    assert report["ok"] and not report["errors"] and not report["warnings"]
    assert report["source_commit"] == "fixture-revision"
    train = report["splits"]["finetune"]
    assert train["rows"] == train["valid_rows"] == 2 and train["documents"] == 1
    assert train["qtypes"] == {"information": 2} and train["sources"] == {"fixture": 2}
    assert train["sha256"] == hashlib.sha256((splits / "finetune.jsonl").read_bytes()).hexdigest()
    assert audit.validate_splits(splits, "fixture-revision") == report


@pytest.mark.parametrize("bad_content", ["", "not JSON\n", "[]\n", '{"question_id":"x"}\n'])
def test_empty_malformed_or_incomplete_split_fails(splits, bad_content):
    (splits / "dev.jsonl").write_text(bad_content)
    report = audit.validate_splits(splits)
    assert not report["ok"] and report["errors"]
    assert report["splits"]["dev"]["valid_rows"] == 0


def test_missing_split_fails_and_still_reports_other_splits(splits):
    (splits / "eval.jsonl").unlink()
    report = audit.validate_splits(splits)
    assert not report["ok"] and report["splits"]["eval"]["sha256"] is None
    assert report["splits"]["finetune"]["valid_rows"] == 2


@pytest.mark.parametrize("field,value", [("question_id", "same"), ("question", " Same QUESTION? ")])
def test_duplicates_within_a_split_fail(splits, field, value):
    rows = [row("a", "doc-t", "Same question?"), row("b", "doc-t", "Different question?")]
    rows[0][field] = value.strip().lower() if field == "question" else value
    rows[1][field] = value
    write(splits, "finetune", rows)
    report = audit.validate_splits(splits)
    assert not report["ok"]
    assert any(f"duplicated {field}" in error for error in report["errors"])


@pytest.mark.parametrize("field,value", [("question_id", "t1"), ("doc_id", "doc-t"),
                                        ("question", "  QUESTION one?  ")])
def test_cross_split_leakage_fails(splits, field, value):
    item = row("d1", "doc-d", "Question three?")
    item[field] = value
    write(splits, "dev", [item])
    report = audit.validate_splits(splits)
    assert not report["ok"]
    assert report["overlaps"]["finetune/dev"]["counts"][field] == 1


def test_shared_focus_is_reported_without_claiming_topic_isolation(splits):
    write(splits, "dev", [row("d1", "doc-d", "Question three?", "Asthma")])
    write(splits, "eval", [row("e1", "doc-e", "Question four?", " asthma ")])
    report = audit.validate_splits(splits)
    assert report["ok"] and not report["errors"]
    assert report["overlaps"]["dev/eval"]["counts"]["focus"] == 1
    assert report["warnings"]


def test_checksum_changes_when_input_changes(splits):
    before = audit.validate_splits(splits)["splits"]["dev"]["sha256"]
    item = row("d1", "doc-d", "Question three?")
    item["answer"] = "A changed reference."
    write(splits, "dev", [item])
    assert audit.validate_splits(splits)["splits"]["dev"]["sha256"] != before


def test_cli_writes_failure_report_without_modifying_inputs(splits, monkeypatch):
    (splits / "eval.jsonl").write_text("broken\n")
    out = splits / "reports" / "audit.json"
    before = {name: (splits / f"{name}.jsonl").read_bytes() for name in audit.SPLITS}
    monkeypatch.setattr(sys, "argv", ["audit", "--data-dir", str(splits), "--out", str(out)])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 1
    assert not json.loads(out.read_text())["ok"]
    assert all((splits / f"{name}.jsonl").read_bytes() == content
               for name, content in before.items())


def test_cli_success_records_source_revision(splits, monkeypatch):
    out = splits / "reports" / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--data-dir", str(splits), "--out", str(out),
                                     "--source-commit", "fixture-revision"])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 0
    report = json.loads(out.read_text())
    assert report["ok"] and report["source_commit"] == "fixture-revision"


def test_cli_refuses_existing_report_without_overwrite(splits, monkeypatch):
    out = splits / "audit.json"
    out.write_text("existing evidence\n")
    monkeypatch.setattr(sys, "argv", ["audit", "--data-dir", str(splits), "--out", str(out)])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2
    assert out.read_text() == "existing evidence\n"


def test_cli_can_explicitly_replace_report(splits, monkeypatch):
    out = splits / "audit.json"
    out.write_text("old report\n")
    monkeypatch.setattr(sys, "argv", ["audit", "--data-dir", str(splits), "--out", str(out),
                                     "--overwrite"])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 0 and json.loads(out.read_text())["ok"]


def test_cli_never_overwrites_split_even_with_overwrite(splits, monkeypatch):
    out = splits / "eval.jsonl"
    before = out.read_bytes()
    monkeypatch.setattr(sys, "argv", ["audit", "--data-dir", str(splits), "--out", str(out),
                                     "--overwrite"])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2 and out.read_bytes() == before
