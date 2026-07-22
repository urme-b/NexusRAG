# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-07-12

### Changed
- The import package is renamed `nexusrag` → `scinexusrag` to match the PyPI
  distribution name; user code now uses `import scinexusrag` and the
  `scinexusrag` / `scinexusrag-eval` / `scinexusrag-report` console scripts.
- Dependencies are installed from `pyproject.toml` (Docker build and the CI
  audit); the hash-pinned lockfiles were removed.

### Fixed
- BM25 ingestion no longer raises `ZeroDivisionError` on an all-stop-word corpus.
- Non-UTF-8 `.txt`/`.md` uploads raise an actionable `DocumentParseError`
  instead of a raw `UnicodeDecodeError`.
- Document IDs hash the full normalized content, so two documents sharing a long
  prefix are no longer silently de-duplicated.
- `LLMClient.generate` wraps a missing `response` field as `LLMError` instead of
  leaking a `KeyError`.

## [1.0.1] - 2026-07-03

### Added
- Every Hugging Face model is revision-pinned: the reranker, NLI, and MiniLM
  baseline join BGE-small, all resolved through one `HF_REVISIONS` map in
  config, so faithfulness and baseline numbers cannot drift on a re-fetch.
- CI enforces a 60% branch-coverage floor and a meta-test that fails on any
  config field with no reader (guards the dead-knob class of bug).
- `make eval-gate` target; CONTRIBUTING documents running it before pushing.

### Fixed
- `.env` files are actually loaded: `load_dotenv()` runs at config import.
  The dependency was declared but never called, so a documented `.env` was
  silently ignored; a subprocess test now guards the behavior.
- Corrective retrieval no longer runs the dense pass twice per query — the
  confidence score comes from the single first-pass dense retrieval.
- The adaptive-fusion technical-word heuristic strips punctuation, so a
  natural-language query ending in "…classification." is not misread as lexical.
- `Embedder.similarity` guards against a zero vector (returns 0, not NaN).
- The LLM client no longer retries on a read timeout (a slow model previously
  multiplied the hang up to ~3 minutes); it fails fast after one timeout.
- `clear_all` resets the BM25 handle directly instead of rebuilding the index
  from the store only to discard it.
- The server logs why interactive docs are disabled (API key / config).
- End-to-end test: ingest → query → assert the answer cites a real source.
- README `Citation` section with BibTeX.
- Chunking no longer loses text: a short document (below `min_chunk_size`) now
  yields a chunk instead of zero, a section's under-min tail keeps its own
  section/page metadata instead of being merged into the previous section's
  chunk, a boundary-less oversized paragraph is hard-wrapped under
  `max_chunk_size`, and a short trailing sentence of an oversized paragraph is
  merged rather than dropped.
- `LLMSettings` reads `LLM_TEMPERATURE`/`LLM_TIMEOUT`, not bare
  `TEMPERATURE`/`TIMEOUT`, so unrelated environment variables can no longer
  override the LLM timeout or generation temperature.
- The query API returns grounding/citation `warnings`, and the web UI renders
  them so ungrounded or uncited answers are flagged to the user.
- Parsing keeps body text that appears before the first heading in DOCX and
  Markdown (previously dropped and unretrievable).
- Vague-query rewriting only fires on the whole query; a specific question like
  "summarize the CRISPR methods" is no longer replaced with a generic prompt.
- BM25 retrieval reads a single index snapshot, so a concurrent ingest cannot
  desync scores from chunks.
- `BM25Retriever.add_incremental` is idempotent by chunk id, so a lazy
  cold-start rebuild that observes a document's just-written chunks can no
  longer double-count them; `_persist`/`delete_document` now assert the sparse
  and dense indexes hold the same number of chunks so any future desync fails
  loudly instead of silently.
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
- README coverage figure corrected to the measured branch coverage.
- PyPI classifier updated from Alpha to Production/Stable to match 1.0.

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
- GitHub Actions are pinned to commit SHAs (supply-chain hardening for the
  Trusted-Publishing workflow); dead code removed across the package.
- Paper/README corrections: nDCG@10 described as graded (not binary) relevance,
  the reranker's latency multiplier is a consistent ~67x, the cross-encoder CI
  claim no longer overstates separation from NLI, and grounding is documented
  as opt-in.
- One canonical sentence-boundary rule (`nexusrag.utils.text.split_sentences`)
  shared by chunking and grounding, replacing two slightly different regexes.
- `ingest` and `ingest_bytes` share one `_ingest_document` body instead of
  duplicating the chunk/embed/persist logic.
- Config knobs are all live now: `SELF_CORRECTION_ENABLED=false` actually
  disables the corrective loop, and `LOG_LEVEL` is applied at startup. The two
  never-read knobs (`RETRIEVAL_RERANK_TOP_K`, `RETRIEVAL_SIMILARITY_THRESHOLD`)
  are removed — the reranker is eval-only and no similarity gate exists.
- Config `temperature` is honored end-to-end (threaded through to the LLM call);
  the reranker reports a neutral score when all candidates tie; grounding uses a
  sigmoid for single-logit cross-encoders instead of a degenerate softmax;
  `is_available()` treats a malformed Ollama response as unavailable.
- The eval significance reference is pinned to the corrective pipeline, so an
  optional `--rerank`/`--splade` rung can't silently become the baseline.
- Docker image is multi-stage: the compiler toolchain stays in the build stage
  and no longer ships in the runtime image; the frontend path resolves in the
  container; compose drives the pulled and queried model from one variable.
- Committed benchmark JSONs carry the derived paired-delta CIs, p-values,
  `rrf_k`, and pinned dataset revisions that `run.py` now emits.
- CI now checks formatting (`ruff format --check`), with ruff pinned so the
  formatter cannot drift between local, pre-commit, and CI.
- PyYAML is a dev-only dependency (only tests read YAML); removed the dead
  `R20` alias and the unused `DocumentStore.list_all`; SECURITY.md's
  supported-versions table reflects the 1.0.x line; the config reference no
  longer documents logging knobs that do not exist.

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
  Recall@20 at ~67x latency on abstract corpora.
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

[1.0.1]: https://github.com/urmeo/NexusRAG/releases/tag/v1.0.1
[1.0.0]: https://github.com/urmeo/NexusRAG/releases/tag/v1.0.0
[0.1.1]: https://github.com/urmeo/NexusRAG/releases/tag/v0.1.1
[0.1.0]: https://github.com/urmeo/NexusRAG/releases/tag/v0.1.0
