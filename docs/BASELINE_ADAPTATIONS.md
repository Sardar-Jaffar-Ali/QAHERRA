# Controlled baseline adaptations

The baseline results are controlled adaptations, not claims of exact
reproduction of every original experimental condition.

| Method | Preserved mechanism | Standardized components |
|---|---|---|
| TSpec-LLM | Global dense retrieval | Corpus, chunking, encoder, generator, prompt |
| BM25 RAG | Global lexical retrieval | Corpus, generator, prompt |
| Telco-RAG | Neural top-five series routing, glossary enhancement, candidate-answer query refinement, second retrieval | R8--R19 corpus, chunking, local dense encoder, generator, no online search |
| Chat3GPP | Global lexical+dense discovery with rank fusion and no reranker | R8--R19 corpus, chunking, encoder, generator, question-only retrieval |
| QAHERRA | Text/table evidence, weighted lexical+dense fusion, table boost, cross-encoder refinement | Locked settings in `configs/qaherra.yaml` |

Answer options are supplied to every method's final answer generator. For
QAHERRA option-conditioned retrieval, options are additionally introduced only
at cross-encoder reranking; initial candidate discovery remains question-only.

Original implementations:

- Telco-RAG: commit `df92a34d3ad824fe4ef45b81fc956498a353a943`
- Chat3GPP: commit `ce65e04da7474ff972516dd807b0278c1bfbf3b3`
