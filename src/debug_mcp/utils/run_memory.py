"""
In-memory storage for LangSmith run content with search capabilities.

This module provides a memory store that holds full run data and allows
for efficient retrieval via keyword search or semantic similarity search.

Semantic search uses sentence-transformers (all-MiniLM-L6-v2 model) to embed
text chunks and find semantically related content.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TextChunk:
    """A chunk of text with its location in the original data."""

    text: str
    path: str  # JSON path like "outputs.chat_history.2.content"
    chunk_index: int  # Index within the path if content was split
    embedding: NDArray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "path": self.path,
            "chunk_index": self.chunk_index,
        }


@dataclass
class StoredRun:
    """A stored run with its full data and searchable chunks."""

    reference_id: str
    full_data: dict[str, Any]
    chunks: list[TextChunk] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    summary: dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider:
    """Embedding provider using sentence-transformers (all-MiniLM-L6-v2 model)."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Use a small, fast model suitable for semantic search
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def embed(self, texts: list[str]) -> list[NDArray]:
        """Embed a list of texts."""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return list(embeddings)

    def embed_single(self, text: str) -> NDArray:
        """Embed a single text."""
        return self.embed([text])[0]


class RunMemoryStore:
    """
    In-memory store for LangSmith run data with search capabilities.

    Stores full run data and provides:
    - Keyword search across all text content
    - Semantic similarity search using sentence-transformers embeddings
    - Direct field access via JSON path
    """

    def __init__(self, max_runs: int = 50, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the memory store.

        Args:
            max_runs: Maximum number of runs to keep in memory (LRU eviction)
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks for context preservation
        """
        self._store: dict[str, StoredRun] = {}
        self._access_order: list[str] = []  # For LRU eviction
        self._max_runs = max_runs
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_provider = EmbeddingProvider()

    def store(self, reference_id: str, data: dict[str, Any], summary: dict[str, Any] | None = None) -> StoredRun:
        """
        Store run data and create searchable chunks.

        Args:
            reference_id: Unique identifier for this run (e.g., "dev:run-uuid")
            data: Full run data dictionary
            summary: Optional pre-computed summary

        Returns:
            The stored run object
        """
        # Evict old entries if at capacity
        self._evict_if_needed()

        # Extract text content and create chunks
        chunks = self._extract_chunks(data)

        # Generate embeddings for chunks
        if chunks:
            texts = [c.text for c in chunks]
            embeddings = self._embedding_provider.embed(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        stored_run = StoredRun(
            reference_id=reference_id,
            full_data=data,
            chunks=chunks,
            summary=summary or {},
        )

        self._store[reference_id] = stored_run
        self._update_access(reference_id)

        return stored_run

    def get(self, reference_id: str) -> StoredRun | None:
        """Get a stored run by reference ID."""
        run = self._store.get(reference_id)
        if run:
            self._update_access(reference_id)
        return run

    def get_field(self, reference_id: str, field_path: str) -> Any:
        """
        Get a specific field from stored run data using dot notation.

        Args:
            reference_id: The run reference ID
            field_path: Dot-notation path like "outputs.chat_history.2.content"

        Returns:
            The value at the path, or None if not found
        """
        run = self.get(reference_id)
        if not run:
            return None

        return self._get_nested(run.full_data, field_path)

    def search_keyword(
        self, reference_id: str, query: str, max_results: int = 5, context_chars: int = 200
    ) -> list[dict[str, Any]]:
        """
        Search for keyword matches in stored run content.

        Args:
            reference_id: The run reference ID
            query: Search query (case-insensitive)
            max_results: Maximum number of results to return
            context_chars: Characters of context around matches

        Returns:
            List of matches with path, snippet, and relevance info
        """
        run = self.get(reference_id)
        if not run:
            return []

        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for chunk in run.chunks:
            text_lower = chunk.text.lower()

            # Check for query presence
            if query_lower in text_lower:
                # Find match position and extract context
                pos = text_lower.find(query_lower)
                start = max(0, pos - context_chars)
                end = min(len(chunk.text), pos + len(query) + context_chars)
                snippet = chunk.text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(chunk.text):
                    snippet = snippet + "..."

                results.append(
                    {
                        "path": chunk.path,
                        "chunk_index": chunk.chunk_index,
                        "snippet": snippet,
                        "match_type": "exact",
                        "score": 1.0,
                    }
                )
            else:
                # Check for word matches
                chunk_words = set(text_lower.split())
                matching_words = query_words & chunk_words
                if matching_words:
                    # Calculate relevance score
                    score = len(matching_words) / len(query_words)
                    if score >= 0.5:  # At least half the words match
                        results.append(
                            {
                                "path": chunk.path,
                                "chunk_index": chunk.chunk_index,
                                "snippet": chunk.text[:context_chars]
                                + ("..." if len(chunk.text) > context_chars else ""),
                                "match_type": "partial",
                                "score": score,
                                "matching_words": list(matching_words),
                            }
                        )

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def search_similar(
        self, reference_id: str, query: str, max_results: int = 5, min_similarity: float = 0.3
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar content using embeddings.

        Args:
            reference_id: The run reference ID
            query: Search query
            max_results: Maximum number of results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of matches with path, text, and similarity score
        """
        run = self.get(reference_id)
        if not run or not run.chunks:
            return []

        # Get query embedding
        query_embedding = self._embedding_provider.embed_single(query)

        results = []
        for chunk in run.chunks:
            if chunk.embedding is None:
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            if similarity >= min_similarity:
                results.append(
                    {
                        "path": chunk.path,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "similarity": float(similarity),
                    }
                )

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    def list_stored_runs(self) -> list[dict[str, Any]]:
        """List all stored runs with basic info."""
        return [
            {
                "reference_id": ref_id,
                "created_at": run.created_at,
                "chunk_count": len(run.chunks),
                "has_embeddings": any(c.embedding is not None for c in run.chunks),
                "summary": run.summary,
            }
            for ref_id, run in self._store.items()
        ]

    def clear(self, reference_id: str | None = None) -> None:
        """Clear stored runs. If reference_id is provided, only clear that run."""
        if reference_id:
            self._store.pop(reference_id, None)
            if reference_id in self._access_order:
                self._access_order.remove(reference_id)
        else:
            self._store.clear()
            self._access_order.clear()

    def _extract_chunks(self, data: dict[str, Any], path: str = "") -> list[TextChunk]:
        """Recursively extract text chunks from nested data structure."""
        chunks = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                chunks.extend(self._extract_chunks(value, new_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}.{i}"
                chunks.extend(self._extract_chunks(item, new_path))
        elif isinstance(data, str) and len(data) > 50:  # Only chunk substantial text
            # Split into chunks
            text_chunks = self._split_text(data)
            for i, text in enumerate(text_chunks):
                chunks.append(TextChunk(text=text, path=path, chunk_index=i))

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size

            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self._chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - self._chunk_overlap

        return [c for c in chunks if c]  # Filter empty chunks

    def _get_nested(self, data: Any, path: str) -> Any:
        """Get nested value using dot notation path."""
        parts = path.split(".")
        current = data

        for part in parts:
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

            if current is None:
                return None

        return current

    @staticmethod
    def _cosine_similarity(a: NDArray, b: NDArray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _update_access(self, reference_id: str) -> None:
        """Update access order for LRU tracking."""
        if reference_id in self._access_order:
            self._access_order.remove(reference_id)
        self._access_order.append(reference_id)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if at capacity."""
        while len(self._store) >= self._max_runs and self._access_order:
            oldest = self._access_order.pop(0)
            self._store.pop(oldest, None)


# Global instance for the MCP server
_memory_store: RunMemoryStore | None = None


def get_memory_store() -> RunMemoryStore:
    """Get the global memory store instance."""
    global _memory_store
    if _memory_store is None:
        _memory_store = RunMemoryStore()
    return _memory_store
