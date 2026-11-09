from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Passage:
    title: str
    text: str
    score: float = 0.0


class StaticRetriever:
    """Returns the same passages for every query. Only for mock runs and tests."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages

    def search(self, query: str, k: int) -> list[Passage]:
        return self.passages[:k]


class FaissRetriever:
    """Reads an index built by scripts/rag/build_index.py. Embeds on CPU: one short query
    takes milliseconds, and it keeps GPUs free for the models."""

    def __init__(self, index_dir: str | Path):
        import faiss
        from sentence_transformers import SentenceTransformer

        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]
        self.query_prefix = meta.get("query_prefix", "")
        self.model = SentenceTransformer(meta["model"], device="cpu")
        self.lock = threading.Lock()

    def search(self, query: str, k: int) -> list[Passage]:
        with self.lock:
            q = self.model.encode([self.query_prefix + query], normalize_embeddings=True).astype("float32")
            scores, ids = self.index.search(q, k)
        return [
            Passage(self.chunks[i]["title"], self.chunks[i]["text"], float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]


def make_retriever(rag_cfg: dict):
    if rag_cfg["kind"] == "faiss":
        return FaissRetriever(rag_cfg["index_dir"])
    if rag_cfg["kind"] == "none":  # E2: RAG off
        return StaticRetriever([])
    if rag_cfg["kind"] == "static":
        return StaticRetriever([Passage(**p) for p in rag_cfg["passages"]])
    raise ValueError(f"unknown rag kind: {rag_cfg['kind']}")
