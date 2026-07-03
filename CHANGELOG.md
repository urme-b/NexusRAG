# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-07-03

### Changed
- Duplicate detection ignores whitespace-only differences: document IDs hash
  whitespace-normalized content, so re-uploading a trivially reformatted file
  is recognized as the same document (IDs of previously ingested documents
  change once on re-ingest).
- `GET /api/documents` is paginated (`limit` default 100, max 1000; `offset`);
  `total_documents` still reports the full corpus size.
- CORS now allows only the methods and headers the API uses
  (GET/POST/DELETE, Content-Type/X-API-Key) instead of wildcards.
- The hierarchical chunker's two near-duplicate accumulation loops were
  unified into one shared, branch-tested paragraph packer (identical output).
- `Embedder.unload()` is public; the pipeline no longer pokes private state.

### Fixed
- Out-of-range citations (e.g. `[9]` with five sources) are now detected and
  reported by the answer verifier; a redundant pre-strip in the synthesizer
  had silenced that warning, and `raw_response` now holds the actual raw LLM
  output.
- `LLM_MAX_TOKENS` is honored: the synthesizer's token budget still scales
  with source count but is capped by config (default 768, the previous
  effective cap) instead of a hardcoded constant.
- Concurrent API writes can no longer corrupt the BM25 index: a pipeline
  write lock serializes ingest/delete/clear (previously two simultaneous
  uploads could silently drop a document from sparse retrieval).
- DocumentStore heals itself after a crash between its two atomic writes:
  orphaned doc files are re-adopted and phantom index entries dropped on
  load; delete order no longer risks entries that block re-ingestion.
- Retrieval scores are clamped to [0, 1] (an opposite-direction vector could
  yield a negative cosine similarity that broke threshold assumptions).
- Password-protected and scanned (image-only) PDFs now fail with actionable
  messages instead of a generic ingestion error; `ingest_directory` reports
  skipped unsupported files instead of silently ignoring them.
- SECURITY.md describes rate limiting accurately (per-IP fixed-window,
  in-process, single-instance scope); ARCHITECTURE.md documents executor
  concurrency limits and why multi-worker scaling needs shared state.
- API reported a stale `0.1.1` version (OpenAPI schema and `/api/metrics`);
  both now use `nexusrag.__version__`, and pyproject reads the same value via
  a dynamic version, leaving a single declaration.
- Single source of truth for API limits: the dead `MAX_FILE_SIZE_*` constants
  are gone (upload cap was already `API_MAX_UPLOAD_MB`), the query-length cap
  now comes from `RETRIEVAL_MAX_QUERY_LENGTH` (default 2000, matching the
  previously enforced value — the old 512 default was never read), and the
  document-ID length check reuses the store's `MAX_ID_LENGTH`.
- `configs/default.yaml` documented `0.0.0.0` as the default API host; the
  actual (and safer) default is `127.0.0.1`.
- CI now checks formatting (`ruff format --check`), with ruff pinned so the
  formatter cannot drift between local, pre-commit, and CI.
- README coverage figure corrected to the measured 60% branch coverage.
- PyPI classifier updated from Alpha to Production/Stable to match 1.0.

## [1.0.0] - 2026-07-02

First stable release: hybrid retrieval with sentence-level faithfulness
checks and a fully local pipeline (Ollama). No functional changes since
0.1.1; this release marks the API and evaluation methodology as stable.

## [0.1.1] - 2026-06-18

### Fixed
- API-key auth returns 401 (not a 500 crash) on a non-ASCII `X-API-Key`.
- Citations no longer flatten multi-line answers into a single line.
- Ingestion is atomic: a partial write rolls back instead of orphaning the doc.
- Upload size cap is driven by `max_upload_mb` (no shadowed hardcoded limit);
  libmagic no longer false-rejects valid `.txt`/`.md` files.
- SPLADE empty-corpus crash, negative adaptive RRF weights, and citation-index
  gaps from de-duplication.
- `\SciCorrTau` paper macro reports the held-out `best_tau`, not the config tau.
- Frontend renders numbered lists and 422 validation errors correctly.

### Added
- `ExactDenseRetriever` enforces its normalized-embedder (cosine) contract.
- CI guards: README test-count sync check; `pip-audit` with a documented ignore
  for the unpatched torch `CVE-2025-3000`.

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

### Methodology
- Report paired bootstrap CIs of the per-query delta vs BM25 (the strongest
  baseline); both Hybrid-vs-BM25 CIs exclude zero.
- Honest negative result: the cross-encoder reranker lowers nDCG@10 and
  Recall@20 at ~45x latency on abstract corpora.
- Pinned BEIR dataset revisions; record RRF k=60 and the bootstrap seed.
- Paper gains a Limitations and Reproducibility section; `docs/ARCHITECTURE.md`
  documents component-level limitations.

### Infrastructure
- GitHub Actions CI (lint, type-check, test matrix + offline eval smoke,
  gitleaks, pip-audit) and Dependabot.
- API security: key auth, slowapi rate limits, libmagic upload validation,
  /docs gating, pinned model revisions, hash-locked Docker dependencies.
- Pinned `requirements.lock` / `requirements-runtime.lock`, contributing guide,
  code of conduct, and issue/PR templates.

[1.0.0]: https://github.com/urme-b/NexusRAG/releases/tag/v1.0.0
[0.1.1]: https://github.com/urme-b/NexusRAG/releases/tag/v0.1.1
[0.1.0]: https://github.com/urme-b/NexusRAG/releases/tag/v0.1.0
