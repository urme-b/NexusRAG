"""Main orchestrator coordinating the RAG pipeline."""

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from nexusrag.agents.llm import LLMClient
from nexusrag.agents.planner import QueryPlan, QueryPlanner
from nexusrag.agents.retriever_agent import RetrievalQuality, RetrieverAgent, VerifiedResult
from nexusrag.agents.synthesizer import Source, Synthesizer
from nexusrag.retrieval import HybridRetriever, Reranker, RetrievalResult


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""

    stage: str
    action: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    sources: list[Source]
    confidence: float
    reasoning_trace: list[ReasoningStep]
    query_plan: QueryPlan | None = None
    retrieval_quality: RetrievalQuality = RetrievalQuality.CORRECT
    total_chunks_retrieved: int = 0
    processing_time_ms: float = 0.0


class Orchestrator:
    """
    Main coordinator for the self-correcting RAG pipeline.

    Flow:
    1. Query Planning: Analyze and decompose complex queries
    2. Retrieval: Execute CRAG-style retrieval with verification
    3. Synthesis: Generate answer with citations

    The orchestrator tracks reasoning at each step for transparency.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        reranker: Reranker | None = None,
        max_retrieval_attempts: int = 2,  # Reduced for speed
        top_k: int = 3,  # Reduced from 5 for speed
        document_store: Any = None,  # For getting document names
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.document_store = document_store

        # Initialize components
        self.planner = QueryPlanner(llm)
        self.retriever_agent = RetrieverAgent(
            retriever=retriever,
            llm=llm,
            planner=self.planner,
            reranker=reranker,
            max_attempts=max_retrieval_attempts,
        )
        self.synthesizer = Synthesizer(llm)

    def query(self, question: str) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            question: User's research question

        Returns:
            RAGResponse with answer, sources, and reasoning trace
        """
        import time

        start_time = time.perf_counter()

        trace: list[ReasoningStep] = []

        # Stage 1: Query Planning
        trace.append(
            ReasoningStep(
                stage="planning",
                action="Analyzing query complexity",
                result=f"Query: {question[:100]}...",
            )
        )

        plan = self.planner.plan(question)

        trace.append(
            ReasoningStep(
                stage="planning",
                action="Query plan created",
                result=f"Complexity: {plan.complexity.value}, Steps: {len(plan.steps)}",
            )
        )

        # Stage 2: Retrieval with Verification
        if plan.is_multi_step:
            verified_results = self._execute_multi_step(plan, trace)
        else:
            verified_results = [self._execute_single_step(question, trace)]

        # Merge results from all steps
        all_results = []
        overall_quality = RetrievalQuality.CORRECT
        total_attempts = 0

        for vr in verified_results:
            all_results.extend(vr.results)
            total_attempts += vr.attempts
            if vr.quality == RetrievalQuality.INCORRECT:
                overall_quality = RetrievalQuality.INCORRECT
            elif (
                vr.quality == RetrievalQuality.AMBIGUOUS
                and overall_quality != RetrievalQuality.INCORRECT
            ):
                overall_quality = RetrievalQuality.AMBIGUOUS

        # Deduplicate by chunk ID, keeping highest scores
        seen_ids: set[str] = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.chunk.id not in seen_ids:
                seen_ids.add(r.chunk.id)
                unique_results.append(r)

        unique_results = unique_results[: self.top_k]

        trace.append(
            ReasoningStep(
                stage="retrieval",
                action="Retrieval complete",
                result=f"Quality: {overall_quality.value}, Chunks: {len(unique_results)}, Attempts: {total_attempts}",
            )
        )

        # Stage 3: Synthesis
        trace.append(
            ReasoningStep(
                stage="synthesis",
                action="Generating answer with citations",
                result=f"Using {len(unique_results)} source chunks",
            )
        )

        # Get document names for better citations
        doc_names = self._get_document_names(unique_results)

        synthesis = self.synthesizer.synthesize(question, unique_results, doc_names=doc_names)

        trace.append(
            ReasoningStep(
                stage="synthesis",
                action="Answer generated",
                result=f"Confidence: {synthesis.confidence:.2f}, Citations: {len(synthesis.sources)}",
            )
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=synthesis.answer,
            sources=synthesis.sources,
            confidence=synthesis.confidence,
            reasoning_trace=trace,
            query_plan=plan,
            retrieval_quality=overall_quality,
            total_chunks_retrieved=len(unique_results),
            processing_time_ms=elapsed_ms,
        )

    def _execute_single_step(self, query: str, trace: list[ReasoningStep]) -> VerifiedResult:
        """Execute retrieval for a single query."""
        trace.append(
            ReasoningStep(
                stage="retrieval",
                action="Starting retrieval with verification",
                result=f"Query: {query[:80]}...",
            )
        )

        result = self.retriever_agent.retrieve_with_verification(query, top_k=self.top_k)

        if result.reformulated_queries:
            trace.append(
                ReasoningStep(
                    stage="retrieval",
                    action="Query reformulation triggered",
                    result=f"Reformulations: {result.reformulated_queries}",
                )
            )

        return result

    def _get_document_names(self, results: Sequence[RetrievalResult]) -> dict[str, str]:
        """Get document ID to original filename mapping."""
        doc_names = {}

        # Primary: get from document store (has correct original filenames)
        if self.document_store:
            try:
                docs = self.document_store.list_with_metadata()
                for doc_id, meta in docs.items():
                    # Prefer original_filename, then filename, then display_name
                    name = (
                        meta.get("original_filename")
                        or meta.get("filename")
                        or meta.get("display_name")
                        or doc_id[:8]
                    )
                    # Skip temp file names
                    if not name.startswith("tmp"):
                        doc_names[doc_id] = name
            except Exception:
                pass

        # Fallback: get from chunk metadata
        for r in results:
            doc_id = r.chunk.document_id
            if doc_id not in doc_names:
                meta = r.chunk.metadata
                name = str(
                    meta.get("original_filename")
                    or meta.get("filename")
                    or meta.get("display_name")
                    or ""
                )
                if name and not name.startswith("tmp"):
                    doc_names[doc_id] = name

        return doc_names

    def _execute_multi_step(
        self, plan: QueryPlan, trace: list[ReasoningStep]
    ) -> list[VerifiedResult]:
        """Execute retrieval for each step in the plan."""
        results: list[VerifiedResult] = []

        for i, step in enumerate(plan.steps, start=1):
            trace.append(
                ReasoningStep(
                    stage="retrieval",
                    action=f"Executing step {i}/{len(plan.steps)}",
                    result=f"Sub-query: {step.query[:60]}... | Purpose: {step.purpose}",
                )
            )

            step_result = self.retriever_agent.retrieve_with_verification(
                step.query, top_k=self.top_k
            )
            results.append(step_result)

            trace.append(
                ReasoningStep(
                    stage="retrieval",
                    action=f"Step {i} complete",
                    result=f"Found {len(step_result.results)} chunks, Quality: {step_result.quality.value}",
                )
            )

        return results

    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """
        Execute pipeline with streaming synthesis.

        Yields tokens as they're generated, then yields final metadata.
        """
        # Non-streaming stages
        plan = self.planner.plan(question)

        if plan.is_multi_step:
            verified_results = self.retriever_agent.retrieve_multi_step(question, top_k=self.top_k)
        else:
            verified_results = [
                self.retriever_agent.retrieve_with_verification(question, top_k=self.top_k)
            ]

        # Collect results
        all_results = []
        for vr in verified_results:
            all_results.extend(vr.results)

        # Deduplicate
        seen_ids: set[str] = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.chunk.id not in seen_ids:
                seen_ids.add(r.chunk.id)
                unique_results.append(r)
        unique_results = unique_results[: self.top_k]

        # Streaming synthesis
        yield from self.synthesizer.synthesize_streaming(question, unique_results)
