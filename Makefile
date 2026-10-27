# CLIENTS is the leak-free client export (python -m windturbine build-clients ...).
CLIENTS ?= data/clients
SEEDS ?= 42 43 44 45 46
EXPS ?= D central F1 F2 F3_mu0.1
JOBS ?= 3

.PHONY: install test lint list seeds all-seeds table figures smoke
install:
	pip install -e ".[train,dev]"
test:
	pytest -q
lint:
	ruff check src tests
list:
	python -m windturbine list
# Priority subset over five seeds; rerun to resume after an interruption.
seeds:
	python -m windturbine run $(EXPS) --seeds $(SEEDS) --clients $(CLIENTS) --jobs $(JOBS)
all-seeds:
	python -m windturbine run all --seeds $(SEEDS) --clients $(CLIENTS) --jobs $(JOBS)
table:
	python -m windturbine summarise --runs results/runs --out results/paper
figures:
	python -m windturbine figures
# Two rounds / two epochs of every experiment, to check an environment before a long batch.
smoke:
	python -m windturbine run all --seeds 0 --clients $(CLIENTS) --out results/smoke --rounds 2 --set max_epochs=2 --jobs $(JOBS)
