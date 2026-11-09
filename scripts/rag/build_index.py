"""Builds the RAG index from MedlinePlus health topics.

Input: the MedlinePlus health topic XML (mplus_topics_YYYY-MM-DD.xml, from
https://medlineplus.gov/xml.html). English topics' <full-summary> HTML is stripped to
text, split into overlapping word chunks, embedded and stored in a flat inner-product
FAISS index (exact search; the corpus is small enough that approximate search isn't needed).

Output dir: index.faiss, chunks.jsonl, meta.json (read by voiceagent.rag.FaissRetriever)."""
from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# bge models are trained to see this prefix on short search queries (not on passages).
QUERY_PREFIXES = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


def strip_html(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words, out, i = text.split(), [], 0
    while True:
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            return out
        i += size - overlap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mplus-xml", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--chunk-words", type=int, default=250)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    chunks = []
    for topic in ET.parse(a.mplus_xml).getroot().iter("health-topic"):
        if topic.get("language") != "English":
            continue
        text = strip_html(topic.findtext("full-summary") or "")
        if text:
            title = topic.get("title", "")
            chunks += [{"title": title, "text": c, "url": topic.get("url", "")}
                       for c in chunk_words(text, a.chunk_words, a.overlap)]
    if not chunks:
        raise SystemExit("no English health topics found; is this the health topic XML?")

    model = SentenceTransformer(a.model, device=a.device)
    emb = model.encode([f"{c['title']}. {c['text']}" for c in chunks], normalize_embeddings=True,
                       batch_size=64, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(a.out_dir / "index.faiss"))
    with open(a.out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    meta = {"model": a.model, "query_prefix": QUERY_PREFIXES.get(a.model, ""), "chunks": len(chunks),
            "source": a.mplus_xml.name, "chunk_words": a.chunk_words, "overlap": a.overlap}
    (a.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"indexed {len(chunks)} chunks into {a.out_dir}")


if __name__ == "__main__":
    main()
