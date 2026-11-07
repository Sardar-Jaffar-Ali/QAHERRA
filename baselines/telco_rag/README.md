# Telco-RAG controlled adaptation

This adapter uses the original Telco-RAG neural series router and glossary
enhancement from a local checkout of:

https://github.com/netop-team/Telco-RAG

The following components are standardized for comparison:

- 3GPP Releases 8--19 rather than the original restricted corpus.
- 1,600-character chunks with 250-character overlap.
- `all-MiniLM-L6-v2` for corpus/query retrieval embeddings.
- GPT-4o-mini for candidate-answer refinement and final answer generation.
- Online web retrieval disabled.

The original top-five neural series routing and two-stage retrieval procedure
remain active. Per-series FAISS indexes and `chunk_metadata.db` must be built
before running `scripts/run_telcorag.py`.

