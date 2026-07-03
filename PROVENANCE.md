# Provenance

Every model and corpus NexusRAG relies on is **off-the-shelf and downloaded from
its published source** — nothing here is trained, fine-tuned, or redistributed by
this project. This file records the exact source, version, and license of each so
the benchmark numbers can be traced to fixed artifacts.

## Models

All models are pulled at run time from the Hugging Face Hub (or Ollama for the
generator) and cached locally; none are vendored in this repo.

| Model | Role | Source | Pinned version | License | Trained here? |
| --- | --- | --- | --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | dense embeddings (default) | [HF](https://huggingface.co/BAAI/bge-small-en-v1.5) | revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (pinned in `src/nexusrag/config.py`) | MIT | No — off-the-shelf |
| `sentence-transformers/all-MiniLM-L6-v2` | dense embeddings (baseline for the ablation) | [HF](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | latest at fetch (not pinned; baseline only) | Apache-2.0 | No — off-the-shelf |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder reranker (evaluated, off by default) | [HF](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | latest at fetch (not pinned) | Apache-2.0 | No — off-the-shelf |
| `cross-encoder/nli-deberta-v3-small` | grounding / evidence-detection NLI | [HF](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | latest at fetch (not pinned) | Apache-2.0 | No — off-the-shelf |
| `llama3.2:3b` | answer generation (via Ollama) | [Ollama](https://ollama.com/library/llama3.2) | tag `3b` (pinned in `src/nexusrag/config.py`) | Llama 3.2 Community License | No — off-the-shelf |

The embedding model is pinned to a specific Hugging Face git revision as a
supply-chain control: the SHA is a content-addressed commit id, so the exact
weight files that produced the reported numbers can be re-fetched with
`revision=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The reranker, NLI, and
MiniLM baseline are not revision-pinned today — they affect only ancillary
comparisons (reranker is off by default; MiniLM is the "worse baseline" in the
ablation), not the headline BGE-small pipeline. Pinning them is tracked as
follow-up. Ollama pins the generator by tag only (Ollama exposes a digest per
tag, not committed here).

## Corpora

The retrieval and faithfulness benchmarks use published BEIR / SciFact data,
downloaded at eval time. Dataset git revisions are pinned in
`src/nexusrag/eval/datasets.py` so a reported number always references a fixed
snapshot.

| Dataset | Role | Source | Pinned revision | License |
| --- | --- | --- | --- | --- |
| SciFact (`BeIR/scifact`) | retrieval ablation (300 claims / 5,183 abstracts) | [HF](https://huggingface.co/datasets/BeIR/scifact) | corpus/queries `b3b5335604bf5ee3c4447671af975ea25143d4f5`, qrels `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029` | CC-BY-SA-4.0 |
| NFCorpus (`BeIR/nfcorpus`) | retrieval ablation (323 queries / 3,633 docs) | [HF](https://huggingface.co/datasets/BeIR/nfcorpus) | corpus/queries `b5026a0e96e8a7ac4f95f482a596389289d46269`, qrels `a451b3b26d3ae1358f259c1a3a4dd61fcea35a65` | CC-BY-SA-4.0 |
| SciFact claims (evidence detection) | faithfulness / evidence-sentence eval | [SciFact release](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz) ([Allen AI](https://github.com/allenai/scifact)) | `release/latest` archive; a small subset is vendored in `benchmarks/datasets/scifact_claims_sample/` for the offline path | claims & evidence annotations CC-BY-4.0; abstracts (S2ORC) ODC-By-1.0 |

The pinned revisions are the verifiable identifiers here — each is a Hugging
Face git commit that content-addresses the exact dataset snapshot, so no separate
SHA256 manifest is maintained. The vendored offline samples under
`benchmarks/datasets/` are small excerpts of the above corpora, used only for the
CPU smoke path (`make eval-sample`, the CI eval gate); they are subject to the
same upstream licenses.

## Weights trained in this project

None. NexusRAG is a retrieval + orchestration + evaluation layer over
off-the-shelf models. There are no checkpoints, adapters, or fine-tunes produced
or shipped by this repository.
