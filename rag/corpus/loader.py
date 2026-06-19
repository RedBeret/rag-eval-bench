"""Load synthetic manufacturing data into Document objects."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A single document in the corpus."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_parts_csv(path: Path) -> list[Document]:
    """Load a parts CSV (v1 or v2 canonical schema) into Documents.

    Each part becomes one document. Text is a natural-language description
    built from the available fields.
    """
    docs: list[Document] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pn = row.get("part_number") or row.get("partNo") or f"UNKNOWN-{i}"
            name = row.get("name") or row.get("partName") or ""
            category = row.get("category") or row.get("cat") or ""
            uom = row.get("uom") or row.get("measure") or ""
            status = row.get("status") or ""
            text = (
                f"Part number {pn} is a {name} in the {category} category. "
                f"Unit of measure: {uom}. Status: {status}."
            ).strip()
            docs.append(
                Document(
                    doc_id=f"part:{pn}",
                    text=text,
                    metadata={
                        "type": "part",
                        "part_number": pn,
                        "category": category,
                        "status": status,
                    },
                )
            )
    return docs


def load_suppliers_csv(path: Path) -> list[Document]:
    """Load a suppliers CSV into Documents."""
    docs: list[Document] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("supplier_code") or ""
            name = row.get("name") or ""
            country = row.get("country") or ""
            email = row.get("contact_email") or ""
            active = row.get("active") or ""
            text = (
                f"Supplier {name} (code: {code}) is located in {country}. "
                f"Contact: {email}. Active: {active}."
            ).strip()
            docs.append(
                Document(
                    doc_id=f"supplier:{code}",
                    text=text,
                    metadata={"type": "supplier", "code": code, "name": name, "country": country},
                )
            )
    return docs


def load_jsonl(path: Path) -> list[Document]:
    """Load a JSONL file where each line is {doc_id, text, metadata}."""
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    doc_id=obj["doc_id"],
                    text=obj["text"],
                    metadata=obj.get("metadata", {}),
                )
            )
    return docs


def load_corpus(path: Path) -> list[Document]:
    """Auto-detect format and load corpus from file."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".csv":
        # Peek at header to determine schema
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if "supplier_code" in header or "contact_email" in header:
            return load_suppliers_csv(path)
        return load_parts_csv(path)
    msg = f"Unsupported corpus format: {suffix}"
    raise ValueError(msg)
