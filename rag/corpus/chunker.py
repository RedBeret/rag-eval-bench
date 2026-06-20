"""Chunk documents into passages for indexing."""

from __future__ import annotations

from rag.corpus.loader import Document


def chunk_by_sentences(doc: Document, max_chars: int = 512) -> list[Document]:
    """Split a document into chunks of at most max_chars characters.

    Splits on sentence boundaries ('. ') where possible.
    Each chunk retains the parent doc_id as a prefix.
    """
    text = doc.text
    if len(text) <= max_chars:
        return [doc]

    sentences = text.split(". ")
    chunks: list[Document] = []
    current = ""
    chunk_idx = 0

    for sentence in sentences:
        candidate = (current + ". " + sentence).strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(
                Document(
                    doc_id=f"{doc.doc_id}#chunk{chunk_idx}",
                    text=current.strip(),
                    metadata={**doc.metadata, "parent_id": doc.doc_id, "chunk_index": chunk_idx},
                )
            )
            chunk_idx += 1
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(
            Document(
                doc_id=f"{doc.doc_id}#chunk{chunk_idx}",
                text=current.strip(),
                metadata={**doc.metadata, "parent_id": doc.doc_id, "chunk_index": chunk_idx},
            )
        )

    return chunks


def chunk_corpus(docs: list[Document], max_chars: int = 512) -> list[Document]:
    """Chunk all documents in the corpus."""
    result: list[Document] = []
    for doc in docs:
        result.extend(chunk_by_sentences(doc, max_chars=max_chars))
    return result
