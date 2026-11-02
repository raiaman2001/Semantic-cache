.PHONY: help install test lint type-check run docker-up docker-down benchmark evaluate

help:
	@echo "Available commands:"
	@echo "  make install     - Install all production & dev dependencies"
	@echo "  make test        - Run test suite with pytest & coverage"
	@echo "  make lint        - Run ruff linter check"
	@echo "  make type-check  - Run mypy type checker"
	@echo "  make run         - Run application with uvicorn"
	@echo "  make docker-up   - Spin up app, redis, and qdrant via docker-compose"
	@echo "  make docker-down - Stop docker-compose services"
	@echo "  make benchmark   - Run hit rate & latency benchmark"
	@echo "  make evaluate    - Run precision/recall evaluation on 100+ pair dataset"

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov ruff mypy matplotlib

test:
	pytest -v --cov=core --cov=api --cov=db

lint:
	ruff check .

type-check:
	mypy core api db config.py

run:
	python app.py

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

benchmark:
	python scripts/benchmark.py

evaluate:
	python scripts/evaluate.py
