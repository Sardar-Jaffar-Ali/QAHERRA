# The four-document chunk lookup defect

This note records a defect in an earlier version of the fixed-size chunk
baselines, how it was detected, which numbers it affected, and how the current
code prevents it from recurring. It is included because the corrected numbers
differ from an earlier internal draft, and a reader comparing the two is
entitled to know why.

## What the defect was

The four-document chunk runs retrieved a chunk position from the four-document
index and then resolved that position to text through the corpus-wide chunk
store:

```python
# defective
con.execute(
    'select c.text, c.source, c.release from faiss_rows f '
    'join chunks c on c.series = f.series and c.text_hash = f.text_hash '
    'where f.series = ? and f.row_id = ?',
    (int(md.series.iloc[i]), int(md.row_id.iloc[i])),
)
```

The `row_id` column of the four-document metadata is a **local** position
within that index, between 0 and 433. The `faiss_rows` table is keyed by
corpus-wide row. The join therefore succeeded and returned text, but the text
belonged to an unrelated specification.

## How it was detected

The chunk metadata records a `text_hash` beside every chunk. Comparing the
fetched text against that hash settles the question without reference to any
accuracy number:

| Index | Fetched text matches the recorded hash | Fetched source lies inside the four documents |
|---|---:|---:|
| Dense (`faiss_qsmall_4docs`) | 0.000 | 0.005 |
| Lexical (`qsmall_4docs_bm25`) | 0.000 | 0.005 |

Not one of the 434 chunks resolved to its own text, and the resolved text came
from one of the four source documents for two of them. Separately, SHA-1 of the
`text` column stored *in the metadata file itself* reproduces `text_hash` for
100% of rows, which is what makes that column the correct source.

## The fix

The chunk text is read from the metadata file. There is no lookup:

```python
# current, src/qaherra_repro/chunks.py
frame = pickle.load(open(metadata_pickle, "rb")).reset_index(drop=True)
texts = frame.text.astype(str).tolist()
docs = [os.path.basename(str(s)).replace(".docx", "") for s in frame.source]
```

`verify_chunk_metadata` then asserts the property the defect violated, and
raises before any generation happens if it does not hold:

* the collection holds exactly 434 chunks;
* SHA-1 of every chunk's text equals its recorded `text_hash`;
* the lexical and dense indexes cover the same chunk set.

`scripts/run_chunk_baseline_4doc.py` calls these checks at startup and also
reports, after the run, the share of answered questions whose ten sources all
lie within the four source documents. That share is 1.000 in the corrected
runs.

## Which numbers changed

Only the two four-document chunk configurations were affected. Nothing in the
evidence-unit pipeline, the full-corpus runs, the retrieval evaluation, or the
deduplication experiment used that lookup.

| Configuration | Defective run | Corrected run |
|---|---:|---:|
| Chunks / BM25, Q-small, four documents | 43.0 | 66.0 |
| Chunks / Dense, Q-small, four documents | not reported | 60.0 |

The corrected comparison against evidence units is the one the paper reports:

| Comparison | Units | Chunks | Units only | Chunks only | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| BM25, four documents | 84.0 | 66.0 | 22 | 4 | 0.000534 | 0.003 |
| Dense, four documents | 74.0 | 60.0 | 24 | 10 | 0.024307 | 0.073 |

The corrected gap is smaller than the defective one, so the correction weakens
the paper's own result rather than strengthening it.
