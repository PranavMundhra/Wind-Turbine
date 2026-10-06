.PHONY: install test lint check graph audit-demo compare
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests scripts
graph:
	python scripts/render_graph.py
check:
	python scripts/check_registry.py
	python scripts/render_graph.py --check
compare:
	python scripts/compare_runs.py
audit-demo:
	python scripts/make_fake_data.py
