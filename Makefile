.PHONY: setup test lint sample-corpus smoke sample-bench sample-generation sample-judge clean

PYTHON ?= python
EMBED_MODEL ?= all-MiniLM-L6-v2
GEN_MODEL ?= gemma3:4b
JUDGE_MODEL ?= phi4:latest
GEN_REPORT ?= reports/sample-run/generation-details.json
OLLAMA_URL ?= http://localhost:11434

setup:
	$(PYTHON) -m pip install -r requirements.txt -e .

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .
	ruff format --check .

sample-corpus:
	$(PYTHON) scripts/make_sample_corpus.py

smoke:
	mkdir -p reports
	$(PYTHON) -m rag.cli bench samples/parts_sample.csv \
		--embed-model mock \
		--max-qa 20 \
		--skip-generation \
		--out reports/smoke.json

sample-bench:
	$(PYTHON) scripts/run_sample_bench.py --embed-model $(EMBED_MODEL)

sample-generation:
	$(PYTHON) scripts/run_sample_bench.py --embed-model $(EMBED_MODEL) \
		--model $(GEN_MODEL) --ollama-url $(OLLAMA_URL)

# Run sample-generation first to create the answer-level report. The committed
# generation.json contains only aggregate results from the published run and
# cannot be reconstructed into answer-level data.
sample-judge:
	$(PYTHON) -m rag.cli judge $(GEN_REPORT) \
		--judge-model $(JUDGE_MODEL) --ollama-url $(OLLAMA_URL) \
		--out reports/sample-run/judged.json \
		--grade-sheet reports/sample-run/grade_sheet.csv

clean:
	rm -rf .rag-index reports
