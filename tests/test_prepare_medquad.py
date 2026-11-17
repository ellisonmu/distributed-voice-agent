"""Regression fixtures for source-local MedQuAD IDs; no real dataset is loaded."""
import json
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts.data import prepare_medquad as prepare
from voiceagent.analysis.validate_medquad import validate_splits


def document(base, collection, doc_id, pairs):
    directory = base / collection
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.Element("Document", id=doc_id, source=collection)
    group = ET.SubElement(root, "QAPairs")
    for qid, pid, question, answer in pairs:
        pair = ET.SubElement(group, "QAPair", **({"pid": pid} if pid is not None else {}))
        attributes = {"qid": qid} if qid is not None else {}
        ET.SubElement(pair, "Question", **attributes).text = question
        ET.SubElement(pair, "Answer").text = answer
    ET.ElementTree(root).write(directory / f"{doc_id}.xml", encoding="utf-8")


@pytest.mark.parametrize("second_collection,second_doc", [
    ("NIDDK", "0000001"), ("GHR", "0000002"),
])
def test_reused_source_qids_have_distinct_reference_keys(tmp_path, second_collection, second_doc):
    document(tmp_path, "GHR", "0000001", [("0000001-1", "1", "First question?", "First answer.")])
    document(tmp_path, second_collection, second_doc,
             [("0000001-1", "1", "Second question?", "Second answer.")])
    rows = [row for group in prepare.parse(tmp_path).values() for row in group]
    references = {row["question_id"]: row["answer"] for row in rows}
    assert len(rows) == len(references) == 2
    assert references["GHR/0000001/0000001-1"] == "First answer."
    assert references[f"{second_collection}/{second_doc}/0000001-1"] == "Second answer."
    assert all(row["source_question_id"] == "0000001-1" for row in rows)


def test_missing_or_blank_qids_use_pair_id_or_stable_position(tmp_path):
    document(tmp_path, "fixture", "doc", [
        (None, "7", "First?", "One."),
        (None, None, "Second?", "Two."),
        ("  ", None, "Third?", "Three."),
    ])
    rows = prepare.parse(tmp_path)["fixture/doc"]
    assert [row["question_id"] for row in rows] == [
        "fixture/doc/pair-7", "fixture/doc/pair-2", "fixture/doc/pair-3",
    ]
    assert all(row["source_question_id"] is None for row in rows)


def test_ids_do_not_depend_on_checkout_location(tmp_path):
    for checkout in (tmp_path / "first", tmp_path / "second"):
        document(checkout, "fixture", "doc", [("q1", "1", "Question?", "Answer.")])
    assert prepare.parse(tmp_path / "first") == prepare.parse(tmp_path / "second")


def test_empty_answers_and_repeated_questions_are_still_skipped(tmp_path):
    document(tmp_path, "fixture", "doc", [
        ("q1", "1", "Same question?", "Keep this answer."),
        ("q2", "2", " SAME question? ", "Duplicate answer."),
        ("q3", "3", "Unanswered?", " "),
    ])
    rows = prepare.parse(tmp_path)["fixture/doc"]
    assert len(rows) == 1
    assert rows[0]["question_id"] == "fixture/doc/q1"
    assert rows[0]["answer"] == "Keep this answer."


def test_prepared_splits_pass_audit_with_reused_upstream_ids(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "prepared"
    for i, collection in enumerate(("GHR", "NIDDK", "NINDS")):
        document(source, collection, "0000001",
                 [("0000001-1", "1", f"Distinct question {i}?", f"Reference {i}.")])
    monkeypatch.setattr(sys, "argv", ["prepare", "--medquad-dir", str(source),
                                     "--out-dir", str(output), "--eval-size", "1",
                                     "--dev-size", "1", "--seed", "0"])
    prepare.main()
    report = validate_splits(output)
    assert report["ok"], report["errors"]
    assert all(summary["valid_rows"] == 1 for summary in report["splits"].values())
    rows = [json.loads((output / f"{split}.jsonl").read_text()) for split in report["splits"]]
    assert len({row["question_id"] for row in rows}) == 3
    assert {row["source_question_id"] for row in rows} == {"0000001-1"}
