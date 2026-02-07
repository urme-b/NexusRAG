"""Context assembly for LLM prompts."""

import logging
from dataclasses import dataclass
from typing import Any

from nexusrag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class AssembledContext:
    """Assembled context ready for LLM."""

    context_text: str
    source_count: int
    total_chars: int
    sources_used: list[dict[str, Any]]


class ContextBuilder:
    """
    Assembles retrieved chunks into well-structured context for the LLM.

    Features:
    - Clear source attribution
    - Deduplication
    - Length management
    - Priority ordering
    """

    def __init__(
        self,
        max_context_chars: int = 6000,
        max_sources: int = 5,
        include_metadata: bool = True,
    ):
        self.max_context_chars = max_context_chars
        self.max_sources = max_sources
        self.include_metadata = include_metadata
        logger.info(
            f"Initialized ContextBuilder: max_chars={max_context_chars}, max_sources={max_sources}"
        )

    def build(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
    ) -> AssembledContext:
        """
        Build context from retrieval results.

        Args:
            query: User's question
            results: Retrieved chunks with scores
            doc_names: Mapping of document_id to display name

        Returns:
            AssembledContext ready for LLM
        """
        if not results:
            return AssembledContext(
                context_text="No relevant documents found.",
                source_count=0,
                total_chars=0,
                sources_used=[],
            )

        doc_names = doc_names or {}

        # Deduplicate and limit
        unique_results = self._deduplicate(results)[: self.max_sources]

        # Build context parts
        context_parts = []
        sources_used = []
        total_chars = 0

        for i, result in enumerate(unique_results, 1):
            chunk = result.chunk

            # Get document name
            doc_name = self._get_doc_name(chunk, doc_names)

            # Build source header
            header = self._build_source_header(i, doc_name, chunk)

            # Get content (use full_context if available)
            content = getattr(chunk, "content", "") or ""

            # Check length limit
            source_text = f"{header}\n{content}\n"
            if total_chars + len(source_text) > self.max_context_chars:
                # Truncate if needed
                remaining = self.max_context_chars - total_chars - len(header) - 50
                if remaining > 200:
                    content = content[:remaining] + "..."
                    source_text = f"{header}\n{content}\n"
                else:
                    break

            context_parts.append(source_text)
            total_chars += len(source_text)

            # Track source info
            sources_used.append(
                {
                    "index": i,
                    "document_name": doc_name,
                    "section_title": getattr(chunk, "section_title", "")
                    or chunk.metadata.get("section_title", ""),
                    "page_number": getattr(chunk, "page_number", None)
                    or chunk.metadata.get("page_number"),
                    "score": round(result.score, 3),
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                }
            )

        # Assemble final context
        context_text = self._assemble_context(query, context_parts)

        logger.info(f"Built context: {len(sources_used)} sources, {total_chars} chars")

        return AssembledContext(
            context_text=context_text,
            source_count=len(sources_used),
            total_chars=total_chars,
            sources_used=sources_used,
        )

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove duplicate chunks, keeping highest scored."""
        seen_ids = set()
        unique = []

        # Results should already be sorted by score
        for result in results:
            if result.chunk.id not in seen_ids:
                seen_ids.add(result.chunk.id)
                unique.append(result)

        return unique

    def _get_doc_name(self, chunk: Any, doc_names: dict[str, str]) -> str:
        """Get display name for document."""
        doc_id = chunk.document_id

        # Try doc_names mapping first
        if doc_id in doc_names:
            name = doc_names[doc_id]
            if not name.startswith("tmp"):
                return name

        # Try chunk attributes
        name = getattr(chunk, "document_name", "")
        if name and not name.startswith("tmp"):
            return name

        # Try metadata
        meta = chunk.metadata
        for key in ["original_filename", "filename", "display_name", "document_name"]:
            name = str(meta.get(key, ""))
            if name and not name.startswith("tmp"):
                return name

        return "Document"

    def _build_source_header(self, index: int, doc_name: str, chunk: Any) -> str:
        """Build header line for a source."""
        parts = [f"[Source {index}]", doc_name]

        # Add section if available
        section = getattr(chunk, "section_title", "") or chunk.metadata.get("section_title", "")
        if section:
            parts.append(f"Section: {section}")

        # Add page if available
        page = getattr(chunk, "page_number", None) or chunk.metadata.get("page_number")
        if page:
            parts.append(f"Page {page}")

        return " | ".join(parts)

    def _assemble_context(self, _query: str, context_parts: list[str]) -> str:
        """Assemble final context string."""
        separator = "\n" + "-" * 40 + "\n"
        sources_text = separator.join(context_parts)

        return sources_text

    def build_with_query(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
    ) -> str:
        """
        Build complete prompt context including query framing.

        Args:
            query: User's question
            results: Retrieved chunks
            doc_names: Document name mapping

        Returns:
            Complete context string for LLM prompt
        """
        assembled = self.build(query, results, doc_names)

        if assembled.source_count == 0:
            return "No relevant information found in the uploaded documents."

        return f"""QUESTION: {query}

RELEVANT EXCERPTS FROM YOUR DOCUMENTS:

{assembled.context_text}

Based ONLY on the sources above, answer the question."""


def build_context(
    query: str,
    results: list[RetrievalResult],
    doc_names: dict[str, str] | None = None,
    max_chars: int = 6000,
) -> str:
    """
    Convenience function to build context.

    Args:
        query: User's question
        results: Retrieved chunks
        doc_names: Optional document name mapping
        max_chars: Maximum context length

    Returns:
        Formatted context string
    """
    builder = ContextBuilder(max_context_chars=max_chars)
    return builder.build_with_query(query, results, doc_names)
