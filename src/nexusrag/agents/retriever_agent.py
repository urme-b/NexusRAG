"""CRAG-style retrieval agent with self-correction."""

from dataclasses import dataclass, field
from enum import Enum

from nexusrag.agents.llm import LLMClient
from nexusrag.agents.planner import QueryPlanner
from nexusrag.retrieval import HybridRetriever, Reranker, RetrievalResult


class RetrievalQuality(Enum):
    """Assessment of retrieval quality."""

    CORRECT = "correct"  # Relevant documents found
    INCORRECT = "incorrect"  # Documents not relevant
    AMBIGUOUS = "ambiguous"  # Partially relevant


@dataclass
class VerifiedResult:
    """Retrieval result with quality verification."""

    results: list[RetrievalResult]
    quality: RetrievalQuality
    confidence: float
    attempts: int = 1
    reformulated_queries: list[str] = field(default_factory=list)


GRADING_PROMPT = """Evaluate if this document is relevant to the query.

Query: {query}

Document:
{document}

Is this document relevant to answering the query?
Consider:
- Does it contain information that helps answer the query?
- Is the information directly related or only tangentially?

Respond with exactly one word: RELEVANT or IRRELEVANT"""


class RetrieverAgent:
    """Self-correcting retrieval agent using CRAG methodology."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        planner: QueryPlanner | None = None,
        reranker: Reranker | None = None,
        max_attempts: int = 3,
        relevance_threshold: float = 0.5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.planner = planner or QueryPlanner(llm)
        self.reranker = reranker
        self.max_attempts = max_attempts
        self.relevance_threshold = relevance_threshold

    def retrieve_with_verification(
        self,
        query: str,
        top_k: int = 5,
    ) -> VerifiedResult:
        """
        Retrieve documents with quality verification and self-correction.

        Implements CRAG (Corrective RAG) pattern:
        1. Retrieve initial results
        2. Grade each result for relevance
        3. If quality is low, reformulate query and retry
        4. Return verified results with confidence

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            VerifiedResult with quality assessment
        """
        reformulated_queries: list[str] = []
        current_query = query

        for attempt in range(1, self.max_attempts + 1):
            # Retrieve candidates
            results = self.retriever.retrieve(current_query, top_k=top_k * 2)

            # Optional reranking
            if self.reranker and results:
                results = self.reranker.rerank(current_query, results, top_k=top_k)
            else:
                results = results[:top_k]

            # Grade results
            quality, confidence, filtered = self._grade_results(query, results)

            if quality == RetrievalQuality.CORRECT:
                return VerifiedResult(
                    results=filtered,
                    quality=quality,
                    confidence=confidence,
                    attempts=attempt,
                    reformulated_queries=reformulated_queries,
                )

            if attempt < self.max_attempts:
                # Reformulate query for next attempt
                context = self._build_reformulation_context(results)
                current_query = self.planner.reformulate(query, context)
                reformulated_queries.append(current_query)

        # Return best effort after max attempts
        return VerifiedResult(
            results=filtered if filtered else results[:top_k],
            quality=quality,
            confidence=confidence,
            attempts=self.max_attempts,
            reformulated_queries=reformulated_queries,
        )

    def _grade_results(
        self, query: str, results: list[RetrievalResult]
    ) -> tuple[RetrievalQuality, float, list[RetrievalResult]]:
        """
        Grade each result for relevance.

        Returns:
            Tuple of (overall quality, confidence, filtered results)
        """
        if not results:
            return RetrievalQuality.INCORRECT, 0.0, []

        relevant_results: list[RetrievalResult] = []
        relevance_scores: list[float] = []

        for result in results:
            is_relevant, score = self._grade_single(query, result)
            relevance_scores.append(score)
            if is_relevant:
                relevant_results.append(result)

        # Calculate quality
        relevance_ratio = len(relevant_results) / len(results)
        avg_score = sum(relevance_scores) / len(relevance_scores)

        if relevance_ratio >= 0.6:
            quality = RetrievalQuality.CORRECT
        elif relevance_ratio >= 0.3:
            quality = RetrievalQuality.AMBIGUOUS
        else:
            quality = RetrievalQuality.INCORRECT

        return quality, avg_score, relevant_results

    def _grade_single(self, query: str, result: RetrievalResult) -> tuple[bool, float]:
        """Grade a single result using LLM."""
        # Use retrieval score as baseline
        if result.score >= self.relevance_threshold:
            # High-confidence results: quick check
            prompt = GRADING_PROMPT.format(
                query=query,
                document=result.chunk.content[:500],
            )
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=10)

            is_relevant = "RELEVANT" in response.upper()
            score = result.score if is_relevant else result.score * 0.5
            return is_relevant, score

        # Low-confidence: mark as irrelevant
        return False, result.score

    def _build_reformulation_context(self, results: list[RetrievalResult]) -> str:
        """Build context string for query reformulation."""
        if not results:
            return "No relevant documents found."

        topics = set()
        for r in results[:3]:
            words = r.chunk.content.split()[:20]
            topics.add(" ".join(words))

        return f"Retrieved documents discuss: {'; '.join(topics)}"

    def retrieve_multi_step(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VerifiedResult]:
        """
        Execute multi-step retrieval for complex queries.

        Uses QueryPlanner to decompose query, then retrieves for each step.
        """
        plan = self.planner.plan(query)

        if not plan.is_multi_step:
            return [self.retrieve_with_verification(query, top_k)]

        results: list[VerifiedResult] = []
        for step in plan.steps:
            step_result = self.retrieve_with_verification(step.query, top_k)
            results.append(step_result)

        return results
