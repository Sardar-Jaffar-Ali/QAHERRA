# QAHERRA reproduction targets. Override PATHS to use another path file.
PATHS ?= configs/paths.yaml
PY    ?= python

.PHONY: help install verify evidence indexes retrieval qa chunks dedup figure test scan clean

help:
	@echo "install   install the package and its pinned dependencies"
	@echo "verify    recompute every reported number and check it (no API key needed)"
	@echo "evidence  build the evidence units from the specifications"
	@echo "indexes   build the BM25 and FAISS indexes"
	@echo "retrieval Telco-DPR retrieval evaluation (Figure 2, Table IV)"
	@echo "qa        end-to-end QA for the evidence-unit configurations (Table III)"
	@echo "chunks    the corrected four-document fixed-size chunk baselines"
	@echo "dedup     deduplication stages A to D (Section V-D)"
	@echo "figure    Figure 3 from the verified results"
	@echo "test      unit tests that need no 3GPP data"
	@echo "scan      fail if a credential or local path is committed"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

verify:
	$(PY) scripts/verify_results.py --paths $(PATHS)

evidence:
	$(PY) scripts/prepare_evidence.py --paths $(PATHS)

indexes:
	$(PY) scripts/build_indexes.py --paths $(PATHS)

retrieval:
	$(PY) scripts/run_retrieval_eval.py --paths $(PATHS)

qa:
	$(PY) scripts/run_qa.py --method qaherra_qonly --dataset qsmall  --scope four_doc
	$(PY) scripts/run_qa.py --method qaherra_qonly --dataset qsmall  --scope full
	$(PY) scripts/run_qa.py --method qaherra_qonly --dataset teleqna --scope full

chunks:
	$(PY) scripts/run_chunk_baseline_4doc.py --retriever bm25  --paths $(PATHS)
	$(PY) scripts/run_chunk_baseline_4doc.py --retriever dense --paths $(PATHS)

dedup:
	$(PY) scripts/dedup_stage_a_build.py     --paths $(PATHS)
	$(PY) scripts/dedup_stage_b_retrieval.py --paths $(PATHS)
	$(PY) scripts/dedup_stage_c_prompts.py   --paths $(PATHS)
	$(PY) scripts/dedup_stage_d_qa.py        --paths $(PATHS)

figure:
	$(PY) scripts/verify_results.py --paths $(PATHS) --write-results
	$(PY) scripts/make_figure3.py

test:
	$(PY) -m pytest -q tests

scan:
	$(PY) scripts/security_scan.py

clean:
	rm -rf __pycache__ .pytest_cache src/*.egg-info
