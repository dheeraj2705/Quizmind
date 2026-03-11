"""
embeddings.py — ChromaDB + SentenceTransformer embedding management.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "quiz_materials"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """Manage embedding generation and vector storage in ChromaDB.

    Attributes:
        model:      SentenceTransformer embedding model.
        client:     ChromaDB persistent client.
        collection: ChromaDB collection storing all material chunks.
    """

    def __init__(self, persist_dir: str = "./chroma_db") -> None:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        # Load embedding model once at startup
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection with cosine distance
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_and_store(
        self,
        chunks: List[str],
        metadata: List[Dict],
    ) -> int:
        """Embed *chunks* and upsert them into the ChromaDB collection.

        Args:
            chunks:   List of text chunk strings.
            metadata: Parallel list of metadata dicts for each chunk.
                      Must include at least {"filename": str, "chunk_index": int}.

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        embeddings = self._embed(chunks)

        ids = [
            f"{meta['filename']}_{meta['chunk_index']}"
            for meta in metadata
        ]

        # Upsert in batches of 100 to avoid memory spikes
        batch_size = 100
        stored = 0
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = chunks[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]
            batch_meta = metadata[i : i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_meta,
            )
            stored += len(batch_ids)

        return stored

    def query_relevant_chunks(
        self,
        query: str,
        n_results: int = 10,
        where: Dict | None = None,
    ) -> List[str]:
        """Return the *n_results* most semantically relevant chunks.

        Args:
            query:     Natural-language query string.
            n_results: How many top chunks to retrieve.
            where:     Optional ChromaDB metadata filter dict.

        Returns:
            List of document text strings ordered by relevance.
        """
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(n_results, total)

        query_embedding = self._embed([query])[0]

        kwargs: Dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        documents: List[List[str]] = results.get("documents", [[]])
        return documents[0] if documents else []

    def get_all_materials(self) -> List[Dict]:
        """Return a deduplicated list of materials with chunk counts.

        Returns:
            List of dicts: [{"filename": str, "chunks": int, "uploaded_at": str}]
        """
        total = self.collection.count()
        if total == 0:
            return []

        # Fetch all metadata (no embeddings needed)
        results = self.collection.get(include=["metadatas"])
        metadatas: List[Dict] = results.get("metadatas", []) or []

        # Aggregate per filename
        agg: Dict[str, Dict] = {}
        for meta in metadatas:
            fname = meta.get("filename", "unknown")
            if fname not in agg:
                agg[fname] = {
                    "filename": fname,
                    "chunks": 0,
                    "uploaded_at": meta.get("uploaded_at", ""),
                }
            agg[fname]["chunks"] += 1

        return list(agg.values())

    def delete_material(self, filename: str) -> bool:
        """Delete all chunks belonging to *filename* from the collection.

        Returns:
            True if any chunks were deleted, False if none found.
        """
        results = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        ids: List[str] = results.get("ids", []) or []

        if not ids:
            return False

        self.collection.delete(ids=ids)
        return True

    def is_ready(self) -> bool:
        """Check whether the collection is accessible."""
        try:
            self.collection.count()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Return normalised embeddings for *texts*."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
