.PHONY: setup test lint sample-corpus smoke sample-bench clean

PYTHON ?= python
EMBED_MODEL ?= all-MiniLM-L6-v2
GEN_MODEL ?= gemma3:4b
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

clean:
	rm -rf .rag-index reports
