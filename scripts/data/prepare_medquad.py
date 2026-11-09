"""Builds the MedQuAD splits: finetune / dev / eval.

Input: a clone of https://github.com/abachaa/MedQuAD (folders of XML files, one Document
per file). Subsets whose answers were removed for copyright are skipped automatically,
since their <Answer> elements are empty.

Splits are by document, so all questions about one topic (e.g. one disease) land in the
same split and no topic leaks from fine-tuning into evaluation."""
from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def clean_question(q: str) -> str:
    q = re.sub(r"\s*\(are\)", "", q)
    q = re.sub(r"\s+\?", "?", q)
    return " ".join(q.split())


def parse(medquad_dir: Path) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for xml_path in sorted(medquad_dir.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        doc_id = f"{xml_path.parent.name}/{root.get('id', xml_path.stem)}"
        source = root.get("source", xml_path.parent.name)
        focus = (root.findtext("Focus") or "").strip()
        for pair in root.iter("QAPair"):
            q_el, a_el = pair.find("Question"), pair.find("Answer")
            if q_el is None or a_el is None:
                continue
            question = clean_question(q_el.text or "")
            answer = " ".join((a_el.text or "").split())
            if not question or not answer or question.lower() in seen:
                continue
            seen.add(question.lower())
            docs.setdefault(doc_id, []).append({
                "question_id": q_el.get("qid") or f"{doc_id}-{pair.get('pid')}",
                "question": question,
                "answer": answer,
                "qtype": q_el.get("qtype", "unknown"),
                "focus": focus,
                "source": source,
                "doc_id": doc_id,
            })
    return docs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--medquad-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--eval-size", type=int, default=2500)
    p.add_argument("--dev-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    docs = parse(a.medquad_dir)
    doc_ids = sorted(docs)
    random.Random(a.seed).shuffle(doc_ids)
    splits: dict[str, list[dict]] = {"eval": [], "dev": [], "finetune": []}
    for doc_id in doc_ids:
        if len(splits["eval"]) < a.eval_size:
            splits["eval"].extend(docs[doc_id])
        elif len(splits["dev"]) < a.dev_size:
            splits["dev"].extend(docs[doc_id])
        else:
            splits["finetune"].extend(docs[doc_id])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in splits.values())
    print(f"{total} QA pairs with answers from {len(docs)} documents")
    for name, rows in splits.items():
        with open(a.out_dir / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        top = ", ".join(f"{k} {v}" for k, v in Counter(r["qtype"] for r in rows).most_common(5))
        print(f"  {name}: {len(rows)} questions; top qtypes: {top}")


if __name__ == "__main__":
    main()
