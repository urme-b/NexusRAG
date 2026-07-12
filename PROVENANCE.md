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
| `BAAI/bge-small-en-v1.5` | dense embeddings (default) | [HF](https://huggingface.co/BAAI/bge-small-en-v1.5) | revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT | No — off-the-shelf |
| `sentence-transformers/all-MiniLM-L6-v2` | dense embeddings (baseline for the ablation) | [HF](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 | No — off-the-shelf |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder reranker (evaluated, off by default) | [HF](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | revision `c5ee24cb16019beea0893ab7796b1df96625c6b8` | Apache-2.0 | No — off-the-shelf |
| `cross-encoder/nli-deberta-v3-small` | grounding / evidence-detection NLI | [HF](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | revision `fa2804872c3b4bd748f38c0185cc85775361e735` | Apache-2.0 | No — off-the-shelf |
| `llama3.2:3b` | answer generation (via Ollama) | [Ollama](https://ollama.com/library/llama3.2) | tag `3b` (pinned in `src/scinexusrag/config.py`) | Llama 3.2 Community License | No — off-the-shelf |

Every Hugging Face model is pinned to a specific git revision as a supply-chain
control: the SHA is a content-addressed commit id, so the exact weight files
that produced the reported numbers can be re-fetched at any time. The single
source of truth is `HF_REVISIONS` in `src/scinexusrag/config.py`; every loader
resolves through it. Ollama pins the generator by tag only (Ollama exposes a
digest per tag, not committed here).

## Corpora

The retrieval and faithfulness benchmarks use published BEIR / SciFact data,
downloaded at eval time. Dataset git revisions are pinned in
`src/scinexusrag/eval/datasets.py` so a reported number always references a fixed
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
