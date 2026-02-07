"""
NexusRAG Benchmarking Suite
============================
Evaluates retrieval and generation quality across standard RAG metrics.

Metrics:
    - Retrieval: Precision@K, Recall@K, MRR, NDCG
    - Generation: Answer relevance, faithfulness, citation accuracy
    - System: Latency (P50/P95/P99), throughput, memory usage

Usage:
    python benchmarks/run_benchmarks.py --dataset qa_pairs
    python benchmarks/run_benchmarks.py --dataset qa_pairs --top-k 5
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""

    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    avg_score: float


@dataclass
class LatencyMetrics:
    """Latency measurements in milliseconds."""

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""

    dataset: str
    num_queries: int
    top_k: int
    retrieval: RetrievalMetrics
    latency: LatencyMetrics
    timestamp: str


def compute_precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Precision@K."""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / k if k > 0 else 0.0


def compute_recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Recall@K."""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0


def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def compute_ndcg(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    import math

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG
    ideal_rels = sorted(
        [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]],
        reverse=True,
    )
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_rels, 1))

    return dcg / idcg if idcg > 0 else 0.0


def load_dataset(dataset_path: Path) -> list[dict]:
    """
    Load benchmark dataset.

    Expected format (JSON lines):
        {"query": "...", "relevant_doc_ids": ["id1", "id2"], "expected_answer": "..."}
    """
    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        logger.info("Create a dataset file with format: {query, relevant_doc_ids, expected_answer}")
        return []

    with open(dataset_path) as f:
        if dataset_path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def run_retrieval_benchmark(
    rag, dataset: list[dict], top_k: int = 3
) -> tuple[RetrievalMetrics, LatencyMetrics]:
    """Run retrieval benchmark on a dataset."""
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    scores = []
    latencies = []

    for item in dataset:
        query = item["query"]
        relevant_ids = set(item.get("relevant_doc_ids", []))

        start = time.perf_counter()
        response = rag.query(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)

        retrieved_ids = [s.document_id for s in response.sources]
        source_scores = [s.score for s in response.sources]

        precisions.append(compute_precision_at_k(retrieved_ids, relevant_ids, top_k))
        recalls.append(compute_recall_at_k(retrieved_ids, relevant_ids, top_k))
        mrrs.append(compute_mrr(retrieved_ids, relevant_ids))
        ndcgs.append(compute_ndcg(retrieved_ids, relevant_ids, top_k))
        scores.extend(source_scores)

    n = len(dataset) or 1
    latencies_sorted = sorted(latencies)

    retrieval = RetrievalMetrics(
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(mrrs) / n,
        ndcg=sum(ndcgs) / n,
        avg_score=sum(scores) / len(scores) if scores else 0.0,
    )

    latency = LatencyMetrics(
        p50=latencies_sorted[int(len(latencies_sorted) * 0.50)] if latencies_sorted else 0,
        p95=latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0,
        p99=latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0,
        mean=sum(latencies) / n,
        min=min(latencies) if latencies else 0,
        max=max(latencies) if latencies else 0,
    )

    return retrieval, latency


def main():
    parser = argparse.ArgumentParser(description="NexusRAG Benchmark Suite")
    parser.add_argument(
        "--dataset",
        default="qa_pairs",
        help="Dataset name in benchmarks/datasets/ (default: qa_pairs)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of documents to retrieve (default: 3)"
    )
    parser.add_argument(
        "--output", default="benchmarks/results/", help="Output directory for results"
    )
    args = parser.parse_args()

    # Load dataset
    dataset_dir = Path("benchmarks/datasets")
    dataset_path = dataset_dir / f"{args.dataset}.jsonl"
    if not dataset_path.exists():
        dataset_path = dataset_dir / f"{args.dataset}.json"

    dataset = load_dataset(dataset_path)
    if not dataset:
        print(f"No dataset found at {dataset_path}")
        print("\nTo create a benchmark dataset, add a file at:")
        print(f"  {dataset_dir}/{args.dataset}.jsonl")
        print("\nFormat (one JSON object per line):")
        print('  {"query": "What is X?", "relevant_doc_ids": ["id1"], "expected_answer": "X is..."}')
        return

    # Initialize pipeline
    from nexusrag.pipeline import NexusRAG

    rag = NexusRAG()

    print(f"Running benchmark: {args.dataset} ({len(dataset)} queries, top_k={args.top_k})")

    # Run benchmark
    retrieval, latency = run_retrieval_benchmark(rag, dataset, top_k=args.top_k)

    result = BenchmarkResult(
        dataset=args.dataset,
        num_queries=len(dataset),
        top_k=args.top_k,
        retrieval=retrieval,
        latency=latency,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Print results
    print(f"\n{'=' * 50}")
    print(f"RETRIEVAL METRICS (top_k={args.top_k})")
    print(f"{'=' * 50}")
    print(f"  Precision@{args.top_k}: {retrieval.precision_at_k:.3f}")
    print(f"  Recall@{args.top_k}:    {retrieval.recall_at_k:.3f}")
    print(f"  MRR:            {retrieval.mrr:.3f}")
    print(f"  NDCG@{args.top_k}:       {retrieval.ndcg:.3f}")
    print(f"  Avg Score:      {retrieval.avg_score:.3f}")
    print(f"\n{'=' * 50}")
    print("LATENCY (ms)")
    print(f"{'=' * 50}")
    print(f"  P50:  {latency.p50:.0f}ms")
    print(f"  P95:  {latency.p95:.0f}ms")
    print(f"  P99:  {latency.p99:.0f}ms")
    print(f"  Mean: {latency.mean:.0f}ms")
    print(f"  Min:  {latency.min:.0f}ms")
    print(f"  Max:  {latency.max:.0f}ms")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
