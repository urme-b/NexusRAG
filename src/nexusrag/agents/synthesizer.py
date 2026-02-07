"""Production-grade answer synthesis with structured citations."""

import logging
import re
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from nexusrag.agents.llm import LLMClient
from nexusrag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A cited source with metadata."""

    index: int
    chunk_id: str
    document_id: str
    content: str
    score: float = 0.0
    page_number: int | None = None
    section_title: str | None = None
    document_name: str = ""

    @property
    def citation_key(self) -> str:
        """Format citation reference."""
        parts = [f"[{self.index}]"]
        if self.document_name:
            parts.append(self.document_name)
        if self.section_title:
            parts.append(self.section_title)
        return " - ".join(parts)


@dataclass
class SynthesisResult:
    """Result of answer synthesis."""

    answer: str
    sources: list[Source]
    confidence: float
    raw_response: str = ""
    tokens_used: int = 0


# Production system prompt - strict, concise, accurate
SYSTEM_PROMPT = """You are a precise research assistant. Your job is to answer questions using ONLY the provided document excerpts.

STRICT RULES:
1. Use ONLY information explicitly stated in the sources below
2. If the answer is NOT in the sources, say: "This information is not found in the provided documents."
3. NEVER make up facts, dates, names, or statistics
4. NEVER use your general knowledge - only use the sources
5. Quote key phrases directly when possible

CITATION RULES:
- Cite every fact with [1], [2], etc.
- Each citation must match a real source
- Put citation immediately after the fact

RESPONSE FORMAT:
**Answer:** [1-2 sentence direct answer with citations]

**Key Details:**
- [Specific fact from sources] [citation]
- [Another fact] [citation]
- [Another fact] [citation]

**Confidence:** [HIGH if multiple sources confirm, MEDIUM if one source, LOW if inference needed]

Keep response under 150 words. Be specific, not vague."""


USER_PROMPT_TEMPLATE = """QUESTION: {question}

DOCUMENT EXCERPTS:
{formatted_sources}

INSTRUCTIONS:
- Answer the question using ONLY the excerpts above
- Cite with [1], [2], [3] matching the source numbers
- If answer not found, say so clearly
- Be specific and quote when helpful

YOUR ANSWER:"""


class Synthesizer:
    """
    Production-grade answer synthesizer.

    Features:
    - Structured prompts for consistent output
    - Citation verification
    - Confidence scoring
    - Source deduplication
    """

    def __init__(
        self,
        llm: LLMClient,
        max_context_length: int = 6000,
        max_sources: int = 5,
    ):
        self.llm = llm
        self.max_context_length = max_context_length
        self.max_sources = max_sources
        logger.info(
            f"Initialized Synthesizer: max_context={max_context_length}, max_sources={max_sources}"
        )

    def synthesize(
        self,
        query: str,
        results: list[RetrievalResult],
        temperature: float = 0.1,
        doc_names: dict[str, str] | None = None,
    ) -> SynthesisResult:
        """
        Generate answer with verified citations.

        Args:
            query: User's question
            results: Retrieved chunks with scores
            temperature: LLM temperature (lower = more focused, default 0.1)
            doc_names: Mapping of document_id to filename

        Returns:
            SynthesisResult with answer and sources
        """
        logger.info(f"Synthesizing answer for: {query[:50]}...")

        if not results:
            logger.warning("No results provided for synthesis")
            return SynthesisResult(
                answer="No relevant documents found. Please upload documents and try again.",
                sources=[],
                confidence=0.0,
            )

        # Build sources with document names
        doc_names = doc_names or {}
        sources = self._build_sources(results, doc_names)

        # Limit sources
        sources = sources[: self.max_sources]

        # Format sources for LLM
        formatted_sources = self._format_sources_for_llm(sources)

        # Truncate if needed
        if len(formatted_sources) > self.max_context_length:
            formatted_sources = formatted_sources[: self.max_context_length]
            formatted_sources += "\n\n[Additional sources truncated for length]"
            logger.warning("Sources truncated due to length")

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=query, formatted_sources=formatted_sources
        )

        # Generate answer
        logger.info(f"Calling LLM with {len(sources)} sources...")
        try:
            answer = self.llm.generate(
                user_prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=300,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SynthesisResult(
                answer="Unable to generate response. Please try again.",
                sources=sources,
                confidence=0.0,
            )

        # Verify and clean citations
        answer = self._verify_citations(answer, len(sources))

        # Calculate confidence
        confidence = self._calculate_confidence(results, answer, sources)

        logger.info(f"Synthesis complete: confidence={confidence:.2f}")

        return SynthesisResult(
            answer=answer.strip(),
            sources=sources,
            confidence=confidence,
            raw_response=answer,
        )

    def synthesize_streaming(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """Stream the synthesis response token by token."""
        if not results:
            yield "No relevant documents found. Please upload documents and try again."
            return

        doc_names = doc_names or {}
        sources = self._build_sources(results, doc_names)[: self.max_sources]
        formatted_sources = self._format_sources_for_llm(sources)

        if len(formatted_sources) > self.max_context_length:
            formatted_sources = formatted_sources[: self.max_context_length]
            formatted_sources += "\n\n[truncated]"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=query, formatted_sources=formatted_sources
        )

        try:
            yield from self.llm.stream(
                user_prompt, system=SYSTEM_PROMPT, temperature=0.1, max_tokens=300
            )
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "Unable to generate response."

    def _build_sources(
        self, results: list[RetrievalResult], doc_names: dict[str, str]
    ) -> list[Source]:
        """Convert retrieval results to sources with proper names."""
        sources: list[Source] = []
        seen_chunks = set()

        for i, result in enumerate(results, start=1):
            # Skip duplicates
            if result.chunk.id in seen_chunks:
                continue
            seen_chunks.add(result.chunk.id)

            chunk = result.chunk
            metadata = chunk.metadata
            doc_id = chunk.document_id

            # Get document name - multiple fallbacks
            doc_name = self._get_document_name(chunk, doc_id, doc_names, i)

            sources.append(
                Source(
                    index=i,
                    chunk_id=chunk.id,
                    document_id=doc_id,
                    content=chunk.content,
                    score=float(result.score),
                    page_number=getattr(chunk, "page_number", None) or metadata.get("page_number"),  # type: ignore[arg-type]
                    section_title=getattr(chunk, "section_title", "")
                    or metadata.get("section_title"),  # type: ignore[arg-type]
                    document_name=doc_name,
                )
            )

        return sources

    def _get_document_name(
        self,
        chunk: Any,
        doc_id: str,
        doc_names: dict[str, str],
        fallback_index: int,
    ) -> str:
        """Get document name with multiple fallbacks."""
        # Try doc_names mapping
        if doc_id in doc_names:
            name = doc_names[doc_id]
            if not name.startswith("tmp"):
                return name

        # Try chunk attributes
        name = getattr(chunk, "document_name", "")
        if name and not name.startswith("tmp"):
            return name

        # Try metadata
        metadata = chunk.metadata
        for key in ["original_filename", "filename", "display_name", "document_name"]:
            name = str(metadata.get(key, ""))
            if name and not name.startswith("tmp"):
                return name

        return f"Document {fallback_index}"

    def _format_sources_for_llm(self, sources: list[Source]) -> str:
        """Format sources in a clear, numbered structure for the LLM."""
        formatted_parts = []

        for source in sources:
            # Build clear header
            section_info = f", Section: {source.section_title}" if source.section_title else ""
            page_info = f", Page {source.page_number}" if source.page_number else ""

            # Format source block with clear delineation
            source_block = f"""====================
[{source.index}] Source: {source.document_name}{section_info}{page_info}
"{source.content}"
===================="""
            formatted_parts.append(source_block)

        return "\n".join(formatted_parts)

    def _verify_citations(self, answer: str, num_sources: int) -> str:
        """Remove hallucinated citations that don't exist in sources."""
        # Find all citations like [1], [2], etc.
        citation_pattern = r"\[(\d+)\]"

        def replace_invalid(match: re.Match[str]) -> str:
            num = int(match.group(1))
            if 1 <= num <= num_sources:
                return match.group(0)  # Keep valid citation
            logger.warning(f"Removed invalid citation: [{num}]")
            return ""  # Remove invalid citation

        verified = re.sub(citation_pattern, replace_invalid, answer)
        # Clean up double spaces
        verified = re.sub(r"\s+", " ", verified)
        return verified.strip()

    def _calculate_confidence(
        self, results: list[RetrievalResult], answer: str, sources: list[Source]
    ) -> float:
        """Calculate confidence based on multiple factors."""
        if not results:
            return 0.0

        scores = []

        # 1. Source relevance (avg similarity score) - 30%
        avg_similarity = sum(r.score for r in results) / len(results)
        scores.append(min(1.0, avg_similarity) * 0.30)

        # 2. Citation coverage (does answer cite sources?) - 25%
        cited_sources = set(re.findall(r"\[(\d+)\]", answer))
        citation_ratio = len(cited_sources) / len(sources) if sources else 0
        scores.append(min(1.0, citation_ratio) * 0.25)

        # 3. Answer completeness - 25%
        completeness = 0.5  # Base
        if "not found" in answer.lower() or "not in the" in answer.lower():
            completeness = 0.2
        elif len(answer) > 100:
            completeness = 0.9
        else:
            completeness = 0.6
        scores.append(completeness * 0.25)

        # 4. Query word coverage (does answer address the question?) - 20%
        # This is approximated by checking if sources are diverse
        unique_docs = {s.document_id for s in sources}
        doc_diversity = min(1.0, len(unique_docs) / 2)  # 2+ docs = full score

        # Multiple citations from same doc = lower diversity
        if len(cited_sources) >= 2:
            doc_diversity = min(1.0, doc_diversity + 0.3)

        scores.append(doc_diversity * 0.20)

        total = sum(scores)

        # Clamp to reasonable range
        return max(0.15, min(0.95, total))
