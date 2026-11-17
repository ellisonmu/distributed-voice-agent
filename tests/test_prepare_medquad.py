"""Regression fixtures for source-local MedQuAD IDs; no real dataset is loaded."""
import json
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts.data import prepare_medquad as prepare
from voiceagent.analysis.validate_medquad import validate_splits


def document(base, collection, doc_id, pairs, *, filename=None):
    directory = base / collection
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.Element("Document", id=doc_id, source=collection)
    group = ET.SubElement(root, "QAPairs")
    for qid, pid, question, answer in pairs:
        pair = ET.SubElement(group, "QAPair", **({"pid": pid} if pid is not None else {}))
        attributes = {"qid": qid} if qid is not None else {}
        ET.SubElement(pair, "Question", **attributes).text = question
        ET.SubElement(pair, "Answer").text = answer
    ET.ElementTree(root).write(directory / (filename or f"{doc_id}.xml"), encoding="utf-8")


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
    assert references["GHR/0000001.xml#pair-1"] == "First answer."
    assert references[f"{second_collection}/{second_doc}.xml#pair-1"] == "Second answer."
    assert all(row["source_question_id"] == "0000001-1" for row in rows)


def test_missing_or_blank_qids_and_repeated_pids_have_distinct_keys(tmp_path):
    document(tmp_path, "fixture", "doc", [
        (None, "7", "First?", "One."),
        (None, "7", "Second?", "Two."),
        ("  ", None, "Third?", "Three."),
    ])
    rows = prepare.parse(tmp_path)["fixture/doc"]
    assert [row["question_id"] for row in rows] == [
        "fixture/doc.xml#pair-1", "fixture/doc.xml#pair-2", "fixture/doc.xml#pair-3",
    ]
    assert all(row["source_question_id"] is None for row in rows)
    assert [row["source_pair_index"] for row in rows] == [1, 2, 3]


def test_reused_document_and_question_ids_in_distinct_files_keep_all_references(tmp_path):
    document(tmp_path, "CancerGov", "doc", [("q1", "1", "First question?", "First answer.")])
    document(tmp_path, "CancerGov", "doc", [("q1", "1", "Second question?", "Second answer.")],
             filename="doc_extra.xml")
    docs = prepare.parse(tmp_path)
    # Reused document IDs still form one split group, while reference keys differ.
    assert set(docs) == {"CancerGov/doc"}
    rows = docs["CancerGov/doc"]
    assert len(rows) == 2
    assert {row["question_id"]: row["answer"] for row in rows} == {
        "CancerGov/doc.xml#pair-1": "First answer.",
        "CancerGov/doc_extra.xml#pair-1": "Second answer.",
    }
    assert {row["source_question_id"] for row in rows} == {"q1"}
    assert {row["source_file"] for row in rows} == {"CancerGov/doc.xml", "CancerGov/doc_extra.xml"}


def test_reused_qids_within_one_file_keep_distinct_references(tmp_path):
    document(tmp_path, "fixture", "doc", [
        ("q1", "1", "First question?", "First answer."),
        ("q1", "1", "Second question?", "Second answer."),
    ])
    rows = prepare.parse(tmp_path)["fixture/doc"]
    assert len({row["question_id"] for row in rows}) == 2
    assert {row["answer"] for row in rows} == {"First answer.", "Second answer."}
    assert {row["source_question_id"] for row in rows} == {"q1"}


def test_ids_do_not_depend_on_checkout_location(tmp_path):
    for checkout in (tmp_path / "first", tmp_path / "second"):
        document(checkout, "fixture", "doc", [("q1", "1", "Question?", "Answer.")])
    assert prepare.parse(tmp_path / "first") == prepare.parse(tmp_path / "second")


def test_empty_answers_and_repeated_questions_are_still_skipped(tmp_path):
    document(tmp_path, "fixture", "doc", [
        ("q1", "1", "Same question?", "Keep this answer."),
        ("q2", "2", " SAME question? ", "Duplicate answer."),
        ("q3", "3", "Unanswered?", " "),
        ("q4", "4", "Another question?", "Another answer."),
    ])
    rows = prepare.parse(tmp_path)["fixture/doc"]
    assert len(rows) == 2
    assert rows[0]["question_id"] == "fixture/doc.xml#pair-1"
    assert rows[0]["answer"] == "Keep this answer."
    # Original XML positions survive filtering, rather than being renumbered.
    assert rows[1]["question_id"] == "fixture/doc.xml#pair-4"
    assert rows[1]["source_pair_index"] == 4


def test_prepared_splits_pass_audit_with_reused_upstream_ids(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "prepared"
    for i, collection in enumerate(("GHR", "NIDDK", "NINDS")):
        document(source, collection, "0000001",
                 [("0000001-1", "1", f"Distinct question {i}?", f"Reference {i}.")])
    document(source, "GHR", "0000001", [("0000001-1", "1", "Fourth question?", "Fourth answer.")],
             filename="0000001_extra.xml")
    monkeypatch.setattr(sys, "argv", ["prepare", "--medquad-dir", str(source),
                                     "--out-dir", str(output), "--eval-size", "1",
                                     "--dev-size", "1", "--seed", "0"])
    prepare.main()
    report = validate_splits(output)
    assert report["ok"], report["errors"]
    assert sum(summary["valid_rows"] for summary in report["splits"].values()) == 4
    rows = [json.loads(line) for split in report["splits"]
            for line in (output / f"{split}.jsonl").read_text().splitlines()]
    assert len({row["question_id"] for row in rows}) == 4
    assert {row["source_question_id"] for row in rows} == {"0000001-1"}
