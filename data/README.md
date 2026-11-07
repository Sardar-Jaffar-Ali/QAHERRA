# Data preparation

## Not redistributed

This repository does not redistribute 3GPP specifications, extracted standards
content, OpenAI credentials, trained model weights, embeddings, or search
indexes.

## Expected evidence file

QAHERRA expects a UTF-8 CSV with these columns:

| Column | Meaning |
|---|---|
| `doc_id` | Source specification identifier |
| `evidence_type` | `text` or `table` |
| `section_path` | Hierarchical source location |
| `text` | Searchable evidence content |

The reported full corpus contains 2,396,136 evidence units: 1,683,800 text
units and 712,336 table units, spanning 3GPP Releases 8--19.

Text units use 1,600 characters with 250-character overlap. Tables are stored
as independent, variable-length evidence units rather than being flattened
into nearby text chunks.

## Evaluation datasets

- Q-small: use the 100-question file released with TSpec-LLM.
- TeleQnA: retain the 2,000 `Standards specifications` questions used by the
  paper.
- Telco-DPR: load the public `ismailduru/Telco-DPR` dataset and evaluate all
  523 held-out test queries.

## Q-small four-document scope

The restricted setting uses only:

- `36777-f00_1.docx`
- `38811-f40.docx`
- `38901-g10.docx`
- `38821-g20.docx`

