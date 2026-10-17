# RAG Assignment — Baseline vs. Re-Ranking

Two retrieval pipelines over the same chunked corpus, compared on retrieval quality,
answer quality and latency.

| | Pipeline 1 — Baseline | Pipeline 2 — Re-ranking |
|---|---|---|
| Stage 1 | `all-MiniLM-L6-v2` bi-encoder, Top 10 by cosine (FAISS) | same Top 10 |
| Stage 2 | — | `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores all 10 pairs |
| Context to LLM | Top 3 by cosine | Top 3 by cross-encoder score |
| LLM | `qwen2.5:3b` via Ollama, temperature 0, identical prompt | same |

**Document:** Apple Inc. Form 10-K, fiscal 2024 — `data/apple_10k_2024.pdf`
(121 pages, ~64,000 words, 547 chunks).

Everything is in **`rag_baseline_vs_rerank.ipynb`**, committed with its outputs.

## Running it

Runs entirely on CPU; no GPU needed. A full run takes about 15–20 minutes, almost all of it
the 30 local LLM calls (20 answers, plus 10 cold calls for latency).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # pulls the CPU build of torch

# local LLM
ollama serve &
ollama pull qwen2.5:3b

.venv/bin/jupyter nbconvert --to notebook --execute --inplace rag_baseline_vs_rerank.ipynb
```

Both sentence-transformer models download automatically on first run (~100 MB).

## Notebook contents

1. Setup and configuration
2. PDF loading and token-budget chunking (no chunk exceeds the bi-encoder's 256-token limit)
3. Bi-encoder index
4. Pipeline 1
5. Pipeline 2
6. Shared prompt and LLM
7. Failure-mode queries — rescues, controls, and cases where re-ranking does not help
8. Rank-shift detail
9. Answer 1 vs. Answer 2
10. Latency profiling
11. Scaling question, measured
12. Summary
