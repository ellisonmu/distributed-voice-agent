"""Small XML fixtures only: no model downloads, inference, GPU or FAISS dependency."""
import hashlib
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts.rag import build_index as build


NS = "https://medlineplus.gov/download/ghr-summaries-20250602.xsd"


def write_xml(path, root):
    ET.ElementTree(root).write(path, encoding="utf-8")
    return path


def health_xml(path):
    root = ET.Element("health-topics")
    topic = ET.SubElement(root, "health-topic", language="English", title="Lung health",
                          url="https://example.org/lungs", id="h1")
    ET.SubElement(topic, "full-summary").text = "<p>Healthy lungs need clean air.</p>"
    ET.SubElement(topic, "site", title="LINK ONLY TERM", url="https://example.org/link")
    other = ET.SubElement(root, "health-topic", language="Spanish", title="Spanish")
    ET.SubElement(other, "full-summary").text = "Excluded Spanish text."
    return write_xml(path, root)


def genetics_xml(path, kind="health-condition-summary", namespace=NS):
    def tag(name):
        return f"{{{namespace}}}{name}" if namespace else name
    root = ET.Element(tag("summaries"))
    record = ET.SubElement(root, tag(kind), id="g1")
    ET.SubElement(record, tag("name")).text = "Example condition" if kind != "chromosome-summary" else "1"
    ET.SubElement(record, tag("ghr-page")).text = "https://example.org/genetics/example"
    if kind == "gene-summary":
        ET.SubElement(record, tag("gene-symbol")).text = "EXAMPLE1"
    synonyms = ET.SubElement(record, tag("synonym-list"))
    ET.SubElement(synonyms, tag("synonym")).text = "Alternate name"
    texts = ET.SubElement(record, tag("text-list"))
    section = ET.SubElement(texts, tag("text"))
    ET.SubElement(section, tag("text-role")).text = "description"
    body = ET.SubElement(section, tag("html"))
    paragraph = ET.SubElement(body, tag("p"))
    paragraph.text = "First paragraph."
    ET.SubElement(paragraph, tag("em")).text = "Important words."
    ET.SubElement(body, tag("p")).text = "Second paragraph."
    inherited = ET.SubElement(texts, tag("text"))
    ET.SubElement(inherited, tag("text-role")).text = "inheritance"
    ET.SubElement(inherited, tag("html")).text = "<p>Inheritance section &amp; context.</p>"
    # References alone must not become invented clinical passages.
    refs = ET.SubElement(record, tag("related-health-condition-list"))
    ET.SubElement(refs, tag("name")).text = "LINK ONLY CONDITION"
    return write_xml(path, root)


def test_health_topics_keep_english_summary_without_link_listings(tmp_path):
    rows = build.health_topic_chunks(health_xml(tmp_path / "topics.xml"), 250, 50)
    assert len(rows) == 1
    assert rows[0]["text"] == "Healthy lungs need clean air."
    assert rows[0]["title"] == "Lung health" and rows[0]["corpus"] == "health_topics"
    assert "LINK ONLY" not in build.embedding_text(rows[0])


@pytest.mark.parametrize("namespace", [NS, "urn:future-schema", ""])
def test_genetics_namespace_and_html_variants_keep_sections_and_provenance(tmp_path, namespace):
    rows = build.genetics_chunks(genetics_xml(tmp_path / "genetics.xml", namespace=namespace), 250, 50)
    assert len(rows) == 2
    assert rows[0]["text"] == "First paragraph. Important words. Second paragraph."
    assert rows[1]["text"] == "Inheritance section & context."
    assert [r["section"] for r in rows] == ["description", "inheritance"]
    assert all(r["url"] == "https://example.org/genetics/example" for r in rows)
    assert all(r["source_file"] == "genetics.xml" and r["record_id"] == "g1" for r in rows)
    assert all(r["synonyms"] == ["Alternate name"] for r in rows)
    assert all("LINK ONLY" not in build.embedding_text(r) for r in rows)


@pytest.mark.parametrize("kind,title", [
    ("health-condition-summary", "Example condition"),
    ("gene-summary", "EXAMPLE1: Example condition"),
    ("chromosome-summary", "Chromosome 1"),
    ("mtdna-summary", "Example condition"),
])
def test_all_observed_genetics_record_types_are_supported(tmp_path, kind, title):
    rows = build.genetics_chunks(genetics_xml(tmp_path / "genetics.xml", kind), 250, 50)
    assert len(rows) == 2 and all(r["record_type"] == kind for r in rows)
    assert rows[0]["title"] == title + " (description)"


def test_combined_corpus_records_hashes_counts_and_preserves_health_topic_baseline(tmp_path):
    health = health_xml(tmp_path / "topics.xml")
    genetics = genetics_xml(tmp_path / "genetics.xml")
    baseline, base_sources = build.prepare_corpus(health, None, 250, 50)
    combined, sources = build.prepare_corpus(health, genetics, 250, 50)
    assert combined[:len(baseline)] == baseline
    assert len(baseline) == 1 and len(combined) == 3
    assert sources[0] == base_sources[0]
    assert [s["chunks"] for s in sources] == [1, 2]
    assert sources[1]["sha256"] == hashlib.sha256(genetics.read_bytes()).hexdigest()


def test_empty_genetics_narrative_is_not_replaced_with_references(tmp_path):
    path = genetics_xml(tmp_path / "genetics.xml")
    tree = ET.parse(path)
    record = tree.getroot()[0]
    record.remove(next(c for c in record if c.tag.endswith("}text-list")))
    tree.write(path, encoding="utf-8")
    assert build.genetics_chunks(path, 250, 50) == []
    with pytest.raises(ValueError, match="no narrative chunks"):
        build.prepare_corpus(health_xml(tmp_path / "topics.xml"), path, 250, 50)


def test_wrong_genetics_document_type_is_rejected(tmp_path):
    path = health_xml(tmp_path / "wrong.xml")
    with pytest.raises(ValueError, match="Genetics summaries compendium"):
        build.genetics_chunks(path, 250, 50)


@pytest.mark.parametrize("field", ["name", "ghr-page"])
def test_genetics_record_without_name_or_source_url_fails(tmp_path, field):
    path = genetics_xml(tmp_path / "genetics.xml")
    tree = ET.parse(path)
    record = tree.getroot()[0]
    record.remove(next(c for c in record if c.tag.endswith("}" + field)))
    tree.write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="missing name or page URL"):
        build.genetics_chunks(path, 250, 50)


def test_chunks_overlap_without_empty_passages():
    assert build.chunk_words("one two three four five", 3, 1) == [
        "one two three", "three four five",
    ]
    assert build.chunk_words("  ", 3, 1) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (3, -1), (3, 3), (3, 4)])
def test_invalid_chunk_settings_fail_instead_of_looping(size, overlap):
    with pytest.raises(ValueError):
        build.chunk_words("one two three four", size, overlap)


def test_existing_index_is_preserved_before_loading_any_model(tmp_path, monkeypatch):
    existing = tmp_path / "baseline"
    existing.mkdir()
    marker = existing / "index.faiss"
    marker.write_bytes(b"existing index")
    monkeypatch.setattr(sys, "argv", ["build", "--mplus-xml", "unused.xml", "--out-dir", str(existing)])
    with pytest.raises(SystemExit) as exc:
        build.main()
    assert exc.value.code == 2 and marker.read_bytes() == b"existing index"


def test_embedding_context_preserves_default_and_limits_synonyms():
    plain = {"title": "Topic", "text": "Narrative."}
    assert build.embedding_text(plain) == "Topic. Narrative."
    enriched = dict(plain, synonyms=[f"Alias{i}" for i in range(10)])
    text = build.embedding_text(enriched)
    assert "Alias7" in text and "Alias8" not in text and text.endswith("Narrative.")


def add_source_metadata(path, patterns=(("ar", "Autosomal recessive"),),
                        genes=(("RECQL4", "https://example.org/genetics/recql4"),)):
    tree = ET.parse(path)
    root = tree.getroot()
    prefix = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    record = root[0]
    inheritance = ET.SubElement(record, prefix + "inheritance-pattern-list")
    for code, memo in patterns:
        pattern = ET.SubElement(inheritance, prefix + "inheritance-pattern")
        ET.SubElement(pattern, prefix + "code").text = code
        ET.SubElement(pattern, prefix + "memo").text = memo
    related = ET.SubElement(record, prefix + "related-gene-list")
    for symbol, url in genes:
        gene = ET.SubElement(related, prefix + "related-gene")
        ET.SubElement(gene, prefix + "gene-symbol").text = symbol
        ET.SubElement(gene, prefix + "ghr-page").text = url
    tree.write(path, encoding="utf-8")


def test_metadata_is_opt_in_and_original_narratives_are_preserved(tmp_path):
    path = genetics_xml(tmp_path / "genetics.xml")
    add_source_metadata(path)
    original = build.genetics_chunks(path, 250, 50)
    expanded = build.genetics_chunks(path, 250, 50, include_metadata=True)
    assert len(original) == 2 and len(expanded) == 4
    assert [r for r in expanded if r.get("evidence_kind") != "source_metadata"] == original
    metadata = [r for r in expanded if r.get("evidence_kind") == "source_metadata"]
    inheritance = next(r for r in metadata if r["section"] == "inheritance patterns")
    assert "Autosomal recessive" in inheritance["text"]
    assert inheritance["inheritance_patterns"] == [{"code": "ar", "memo": "Autosomal recessive"}]
    assert inheritance["url"] == "https://example.org/genetics/example"
    assert "source metadata" in inheritance["title"]
    genes = next(r for r in metadata if r["section"] == "related genes")
    assert genes["related_genes"] == [{"symbol": "RECQL4", "url": "https://example.org/genetics/recql4"}]
    assert "not specific mutations or variant effects" in genes["text"]


def test_multiple_inheritance_labels_are_not_assigned_to_individual_genes(tmp_path):
    path = genetics_xml(tmp_path / "genetics.xml")
    add_source_metadata(path, patterns=(("ar", "Autosomal recessive"), ("ad", "Autosomal dominant")))
    rows = build.genetics_chunks(path, 250, 50, include_metadata=True)
    inheritance = next(r for r in rows if r["section"] == "inheritance patterns")
    assert "Autosomal recessive; Autosomal dominant" in inheritance["text"]
    assert len(inheritance["inheritance_patterns"]) == 2
    assert "RECQL4" not in inheritance["text"]
    related = next(r for r in rows if r["section"] == "related genes")
    assert "Autosomal" not in related["text"]


def test_codes_without_publisher_labels_are_not_translated_or_invented(tmp_path):
    path = genetics_xml(tmp_path / "genetics.xml")
    add_source_metadata(path, patterns=(("ar", ""), ("unknown-code", "")), genes=())
    rows = build.genetics_chunks(path, 250, 50, include_metadata=True)
    assert len(rows) == 2
    assert not any(r.get("evidence_kind") == "source_metadata" for r in rows)


def test_metadata_can_be_indexed_when_narrative_is_absent_without_fabricating_narrative(tmp_path):
    path = genetics_xml(tmp_path / "genetics.xml")
    add_source_metadata(path)
    tree = ET.parse(path)
    record = tree.getroot()[0]
    record.remove(next(c for c in record if c.tag.endswith("}text-list")))
    tree.write(path, encoding="utf-8")
    assert build.genetics_chunks(path, 250, 50) == []
    rows = build.genetics_chunks(path, 250, 50, include_metadata=True)
    assert len(rows) == 2 and all(r["evidence_kind"] == "source_metadata" for r in rows)


def test_metadata_mode_preserves_health_topic_baseline_and_records_new_counts(tmp_path):
    health, genetics = health_xml(tmp_path / "topics.xml"), genetics_xml(tmp_path / "genetics.xml")
    add_source_metadata(genetics)
    original, sources = build.prepare_corpus(health, genetics, 250, 50)
    expanded, new_sources = build.prepare_corpus(health, genetics, 250, 50, include_genetics_metadata=True)
    assert expanded[:1] == original[:1]
    assert [s["chunks"] for s in sources] == [1, 2]
    assert [s["chunks"] for s in new_sources] == [1, 4]
    assert sources[1]["sha256"] == new_sources[1]["sha256"]


def test_metadata_mode_requires_genetics_input(tmp_path):
    with pytest.raises(ValueError, match="requires --genetics-xml"):
        build.prepare_corpus(tmp_path / "unused.xml", None, 250, 50, include_genetics_metadata=True)


def test_cli_rejects_metadata_flag_without_genetics_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build", "--mplus-xml", "unused.xml", "--out-dir",
                                     str(tmp_path / "new"), "--include-genetics-metadata"])
    with pytest.raises(SystemExit) as exc:
        build.main()
    assert exc.value.code == 2 and not (tmp_path / "new").exists()
