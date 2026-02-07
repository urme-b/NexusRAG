"""Agents module for LLM orchestration and self-correction."""

from nexusrag.agents.context_builder import AssembledContext, ContextBuilder, build_context
from nexusrag.agents.llm import GenerationConfig, LLMClient
from nexusrag.agents.orchestrator import Orchestrator, RAGResponse, ReasoningStep
from nexusrag.agents.planner import QueryComplexity, QueryPlan, QueryPlanner, QueryStep
from nexusrag.agents.query_analyzer import AnalyzedQuery, QueryAnalyzer, QueryType
from nexusrag.agents.retriever_agent import (
    RetrievalQuality,
    RetrieverAgent,
    VerifiedResult,
)
from nexusrag.agents.synthesizer import Source, SynthesisResult, Synthesizer
from nexusrag.agents.verifier import AnswerVerifier, VerificationResult, verify_answer

__all__ = [
    # LLM
    "GenerationConfig",
    "LLMClient",
    # Orchestrator
    "Orchestrator",
    "QueryComplexity",
    "QueryPlan",
    "QueryPlanner",
    "QueryStep",
    "RAGResponse",
    "ReasoningStep",
    # Retrieval
    "RetrievalQuality",
    "RetrieverAgent",
    "VerifiedResult",
    # Synthesis
    "Source",
    "Synthesizer",
    "SynthesisResult",
    # Context Building
    "ContextBuilder",
    "AssembledContext",
    "build_context",
    # Query Analysis
    "QueryAnalyzer",
    "AnalyzedQuery",
    "QueryType",
    # Verification
    "AnswerVerifier",
    "VerificationResult",
    "verify_answer",
]
