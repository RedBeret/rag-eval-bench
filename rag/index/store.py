"""ChromaDB vector store wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.corpus.loader import Document
from rag.index.embedder import Embedder, SentenceTransformerEmbedder

COLLECTION_NAME = "rag_bench"
BATCH_SIZE = 256


class VectorStore:
    """ChromaDB-backed vector store for document retrieval.

    Uses persistent storage when a path is provided, in-memory otherwise.
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedder: Embedder | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        settings = Settings(anonymized_telemetry=False)
        logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
        if persist_dir:
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        else:
            self._client = chromadb.EphemeralClient(settings=settings)

        self._embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and add documents to the collection in batches."""
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            texts = [d.text for d in batch]
            embeddings = self._embedder.embed(texts)
            self._collection.add(
                ids=[d.doc_id for d in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[d.metadata for d in batch],
            )

    def query(self, query_text: str, k: int = 5) -> list[dict]:
        """Retrieve the top-k most similar documents for a query.

        Returns list of dicts with keys: doc_id, text, metadata, distance.
        """
        embedding = self._embedder.embed([query_text])[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            items.append({"doc_id": doc_id, "text": text, "metadata": meta, "distance": dist})
        return items

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate this store's collection."""
        try:
            self._client.delete_collection(self._collection_name)
        except (ValueError, KeyError):
            pass  # collection did not exist yet
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
