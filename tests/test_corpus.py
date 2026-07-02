"""Tests for corpus loading and chunking."""

from pathlib import Path

from rag.corpus.chunker import chunk_corpus
from rag.corpus.loader import load_parts_csv, load_suppliers_csv


def write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_load_parts_csv_v1(tmp_path: Path) -> None:
    csv = write_file(
        tmp_path,
        "parts_v1.csv",
        "part_number,name,category,uom,status\n"
        "PN-0001,Hex Bolt M6,Fasteners,EA,active\n"
        "PN-0002,Nylon Washer,Fasteners,EA,active\n",
    )
    docs = load_parts_csv(csv)
    assert len(docs) == 2
    assert docs[0].doc_id == "part:PN-0001"
    assert "Hex Bolt M6" in docs[0].text
    assert "Fasteners" in docs[0].text
    assert docs[0].metadata["type"] == "part"


def test_load_parts_csv_legacy_columns(tmp_path: Path) -> None:
    """Also handles legacy column names (partNo, partName, cat, measure)."""
    csv = write_file(
        tmp_path,
        "parts_legacy.csv",
        "partNo,partName,cat,measure,status\nP7,Bearing Race,Bearings,EA,obsolete\n",
    )
    docs = load_parts_csv(csv)
    assert len(docs) == 1
    assert "Bearing Race" in docs[0].text


def test_load_suppliers_csv(tmp_path: Path) -> None:
    csv = write_file(
        tmp_path,
        "suppliers.csv",
        "supplier_code,name,country,contact_email,active\n"
        "VRT-001,Vortex Metals,US,proc@vortex.com,1\n",
    )
    docs = load_suppliers_csv(csv)
    assert len(docs) == 1
    assert docs[0].doc_id == "supplier:VRT-001"
    assert "Vortex Metals" in docs[0].text
    assert docs[0].metadata["type"] == "supplier"


def test_chunk_corpus_short_docs_unchanged(tmp_path: Path) -> None:
    """Short documents (< max_chars) should not be split."""
    csv = write_file(
        tmp_path,
        "parts.csv",
        "part_number,name,category,uom,status\nPN-0001,Bolt,Fasteners,EA,active\n",
    )
    docs = load_parts_csv(csv)
    chunks = chunk_corpus(docs, max_chars=512)
    assert len(chunks) == 1
    assert chunks[0].doc_id == docs[0].doc_id


def test_chunk_corpus_long_doc_splits(tmp_path: Path) -> None:
    """Documents longer than max_chars should be split into multiple chunks."""
    csv = write_file(
        tmp_path,
        "parts.csv",
        "part_number,name,category,uom,status\n"
        + "PN-0001,"
        + ("A" * 200)
        + ","
        + ("B" * 200)
        + ",EA,active\n",
    )
    docs = load_parts_csv(csv)
    chunks = chunk_corpus(docs, max_chars=100)
    assert len(chunks) >= 2
    assert all("chunk" in c.doc_id or c.doc_id == docs[0].doc_id for c in chunks)
