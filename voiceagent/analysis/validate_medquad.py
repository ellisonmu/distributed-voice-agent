"""Audit prepared MedQuAD splits without changing them or loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

SPLITS = ("finetune", "dev", "eval")
REQUIRED = ("question_id", "question", "answer", "doc_id")


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_splits(data_dir: Path, source_commit: str | None = None) -> dict:
    """Check answered QA rows, within-split duplicates and cross-split leakage.

    Documents are the current splitting unit. Shared focus labels are warnings,
    not proof of duplicate documents and not a claim of a topic-held-out split.
    """
    errors, warnings, summaries, indexes = [], [], {}, {}
    for name in SPLITS:
        path = data_dir / f"{name}.jsonl"
        rows, valid = 0, []
        checksum = None
        try:
            checksum = _sha256(path)
            with path.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        errors.append(f"{name}:{line_no}: invalid JSON")
                        continue
                    if not isinstance(row, dict):
                        errors.append(f"{name}:{line_no}: row must be an object")
                        continue
                    missing = [key for key in REQUIRED
                               if not isinstance(row.get(key), str) or not row[key].strip()]
                    if missing:
                        errors.append(f"{name}:{line_no}: missing/empty string fields {missing}")
                        continue
                    valid.append(row)
        except (OSError, UnicodeError) as exc:
            errors.append(f"{name}: cannot read {path}: {exc}")
        if not valid:
            errors.append(f"{name}: no valid answered QA rows")

        index = {
            "question_id": Counter(row["question_id"].strip() for row in valid),
            "question": Counter(_normalize(row["question"]) for row in valid),
            "doc_id": Counter(row["doc_id"].strip() for row in valid),
            "focus": Counter(_normalize(row["focus"]) for row in valid
                             if isinstance(row.get("focus"), str) and row["focus"].strip()),
        }
        for key in ("question_id", "question"):
            duplicates = [value for value, count in index[key].items() if count > 1]
            if duplicates:
                errors.append(f"{name}: {len(duplicates)} duplicated {key} values; "
                              f"examples={sorted(duplicates)[:5]}")
        indexes[name] = index
        summaries[name] = {
            "path": str(path.resolve()), "sha256": checksum,
            "rows": rows, "valid_rows": len(valid), "documents": len(index["doc_id"]),
            "distinct_focus_labels": len(index["focus"]),
            "qtypes": dict(sorted(Counter(str(row.get("qtype") or "unknown") for row in valid).items())),
            "sources": dict(sorted(Counter(str(row.get("source") or "unknown") for row in valid).items())),
        }

    overlaps = {}
    for left, right in combinations(SPLITS, 2):
        shared = {key: sorted(set(indexes[left][key]) & set(indexes[right][key]))
                  for key in ("question_id", "question", "doc_id", "focus")}
        overlaps[f"{left}/{right}"] = {
            "counts": {key: len(values) for key, values in shared.items()},
            "examples": {key: values[:5] for key, values in shared.items()},
        }
        for key in ("question_id", "question", "doc_id"):
            if shared[key]:
                errors.append(f"{left}/{right}: {len(shared[key])} overlapping {key} values; "
                              f"examples={shared[key][:5]}")
        if shared["focus"]:
            warnings.append(f"{left}/{right}: {len(shared['focus'])} shared focus labels; "
                            "document splitting does not guarantee topic isolation")

    return {"kind": "medquad_split_audit", "source_commit": source_commit,
            "ok": not errors, "splits": summaries, "overlaps": overlaps,
            "errors": errors, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-commit", help="MedQuAD source revision recorded during download")
    parser.add_argument("--out", type=Path, required=True, help="write the JSON audit here")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing audit report")
    args = parser.parse_args()
    if args.out.resolve() in {(args.data_dir / f"{name}.jsonl").resolve() for name in SPLITS}:
        parser.error("the report path must not replace a split file")
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} already exists; use a new report path or --overwrite")
    report = validate_splits(args.data_dir, args.source_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, summary in report["splits"].items():
        print(f"{name}: {summary['valid_rows']}/{summary['rows']} valid rows, "
              f"{summary['documents']} documents")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    for message in report["errors"][:20]:
        print(f"ERROR: {message}")
    print(f"{'PASS' if report['ok'] else 'FAIL'}: {len(report['errors'])} errors; report={args.out}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
