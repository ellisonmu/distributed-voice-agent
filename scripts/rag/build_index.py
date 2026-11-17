"""Builds the RAG index from MedlinePlus health topics.

Input: MedlinePlus Health Topic XML and, optionally, its separate Genetics compendium.
Narrative summaries are split into overlapping word chunks, embedded and stored in a flat
inner-product FAISS index. Link listings are not treated as narrative medical content.

Output dir: index.faiss, chunks.jsonl, meta.json (read by voiceagent.rag.FaissRetriever)."""
from __future__ import annotations

import argparse
import hashlib
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
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk size must be positive and 0 <= overlap < size")
    words, out, i = text.split(), [], 0
    if not words:
        return []
    while True:
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            return out
        i += size - overlap


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in node if _local_name(child.tag) == name), None)


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    # Supports both escaped HTML in .text and nested (possibly namespaced) HTML.
    inner = (node.text or "") + "".join(ET.tostring(child, encoding="unicode") for child in node)
    return strip_html(inner)


def health_topic_chunks(path: Path, size: int, overlap: int) -> list[dict]:
    chunks = []
    for topic in ET.parse(path).getroot().iter():
        if _local_name(topic.tag) != "health-topic" or topic.get("language") != "English":
            continue
        text = _text(_child(topic, "full-summary"))
        for part in chunk_words(text, size, overlap):
            chunks.append({"title": topic.get("title", ""), "text": part,
                           "url": topic.get("url", ""), "corpus": "health_topics",
                           "source_file": path.name, "record_id": topic.get("id")})
    return chunks


GENETICS_TYPES = {"health-condition-summary", "gene-summary", "chromosome-summary", "mtdna-summary"}


def genetics_chunks(path: Path, size: int, overlap: int) -> list[dict]:
    root = ET.parse(path).getroot()
    if _local_name(root.tag) != "summaries":
        raise ValueError("expected the Genetics summaries compendium XML")
    chunks = []
    for record in root:
        kind = _local_name(record.tag)
        if kind not in GENETICS_TYPES:
            continue
        name, url = _text(_child(record, "name")), _text(_child(record, "ghr-page"))
        if not name or not url:
            raise ValueError(f"Genetics record {record.get('id')}: missing name or page URL")
        symbol = _text(_child(record, "gene-symbol"))
        title = f"{symbol}: {name}" if symbol else name
        if kind == "chromosome-summary":
            title = f"Chromosome {name}"
        synonym_list = _child(record, "synonym-list")
        synonyms = [] if synonym_list is None else [
            text for child in synonym_list if _local_name(child.tag) == "synonym"
            if (text := _text(child))
        ]
        sections = _child(record, "text-list")
        if sections is None:
            continue
        for section in sections:
            if _local_name(section.tag) != "text":
                continue
            role = _text(_child(section, "text-role"))
            narrative = _text(_child(section, "html"))
            for part in chunk_words(narrative, size, overlap):
                chunks.append({"title": f"{title} ({role})" if role else title,
                               "text": part, "url": url, "corpus": "genetics",
                               "source_file": path.name, "record_id": record.get("id"),
                               "record_type": kind, "section": role, "synonyms": synonyms})
    return chunks


def prepare_corpus(mplus_xml: Path, genetics_xml: Path | None, size: int, overlap: int):
    chunks, sources = [], []
    inputs = [(mplus_xml, "health_topics", health_topic_chunks)]
    if genetics_xml is not None:
        inputs.append((genetics_xml, "genetics", genetics_chunks))
    for path, kind, loader in inputs:
        rows = loader(path, size, overlap)
        if not rows:
            raise ValueError(f"no narrative chunks from {path}; check the XML format")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        sources.append({"corpus": kind, "file": path.name,
                        "sha256": digest.hexdigest(), "chunks": len(rows)})
        chunks.extend(rows)
    return chunks, sources


def embedding_text(chunk: dict) -> str:
    # Limit alias context so it does not crowd out the narrative in the encoder.
    aliases = "; ".join(chunk.get("synonyms", [])[:8])
    context = f"{chunk['title']}. "
    if aliases:
        context += f"Also called: {aliases}. "
    return context + chunk["text"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mplus-xml", required=True, type=Path)
    p.add_argument("--genetics-xml", type=Path, help="optional ghr-summaries.xml compendium")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--chunk-words", type=int, default=250)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.chunk_words <= 0 or not 0 <= a.overlap < a.chunk_words:
        p.error("chunk size must be positive and 0 <= overlap < chunk size")
    if a.out_dir.exists() or a.out_dir.is_symlink():
        p.error("output directory already exists; choose a new directory to preserve the old index")
    try:
        chunks, sources = prepare_corpus(a.mplus_xml, a.genetics_xml, a.chunk_words, a.overlap)
    except ValueError as exc:
        p.error(str(exc))

    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(a.model, device=a.device)
    emb = model.encode([embedding_text(c) for c in chunks], normalize_embeddings=True,
                       batch_size=64, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    a.out_dir.mkdir(parents=True, exist_ok=False)
    faiss.write_index(index, str(a.out_dir / "index.faiss"))
    with open(a.out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    meta = {"model": a.model, "query_prefix": QUERY_PREFIXES.get(a.model, ""), "chunks": len(chunks),
            "source": a.mplus_xml.name, "sources": sources,
            "embedding_context": "title, first 8 Genetics synonyms, narrative",
            "chunk_words": a.chunk_words, "overlap": a.overlap}
    (a.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"indexed {len(chunks)} chunks into {a.out_dir}")
    for source in sources:
        print(f"  {source['corpus']}: {source['chunks']} chunks; sha256={source['sha256']}")


if __name__ == "__main__":
    main()
