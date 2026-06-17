# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-18

### Added
- Local hybrid retrieval (BGE-small dense + BM25, fused with reciprocal rank
  fusion) with a confidence-gated corrective PRF loop.
- Reproducible BEIR ablation on SciFact and NFCorpus with bootstrap confidence
  intervals, paired randomization tests, and Holm correction.
- Faithfulness evaluation: NLI, lexical-overlap, and cross-encoder evidence
  detectors with ROC-AUC, PR-AUC, and risk-coverage.
- FastAPI app and local web UI with inline citations and grounding checks.
- Auto-generated paper tables, figures, and macros from committed results.

### Fixed
- nDCG@10 now uses graded relevance gains (BEIR/pytrec_eval convention), so
  NFCorpus numbers are comparable to published BEIR results.
- Corrective-loop `tau` is selected on a held-out split instead of the test set,
  removing test-set hyperparameter tuning.
- Vector store now scores by cosine similarity; the corrective confidence gate
  previously compared against raw L2 distance.
- LLM client gained retry/backoff and timeout classification.
- Streaming responses now run citation and grounding verification.
- Quickstart notebook ships with a tracked sample document.

### Infrastructure
- GitHub Actions CI (lint, type-check, test matrix + offline eval smoke).
- Pinned `requirements.lock` for reproducible environments.
- Contributing guide, code of conduct, and issue/PR templates.

[0.1.0]: https://github.com/urme-b/NexusRAG/releases/tag/v0.1.0
