# Query-aware table-gate validation protocol

The validation sample contains all 100 Q-small questions and 200 TeleQnA
questions sampled with `random_state=42`. A question is labelled table-relevant
only when answering requires locating, comparing, or interpreting structured
rows, columns, parameter values, fields, codes, mappings, enumerations, or
configurations in a table. A table that is merely helpful is insufficient.

Two independent deterministic GPT-4o-mini annotations are collected. Cases
without high-confidence agreement are adjudicated in a third deterministic
pass. The resulting 300 labels are used only to validate/train the query gate,
not to train the dense evidence retriever or answer generator.

Annotation statistics:

- Questions: 300
- Initial agreement: 89.00%
- Cohen's kappa: 0.7309
- Adjudicated cases: 36
- Final table-relevant questions: 75
- Final non-table questions: 225

The learned gate is evaluated by stratified five-fold out-of-fold prediction.
No question is predicted by a classifier trained on that question.

