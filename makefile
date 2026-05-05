# Makefile for Major Project 2026-2027

.PHONY: help install run lint format fix clean

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies using uv"
	@echo "  run        - Start the FastAPI server"
	@echo "  lint       - Run Ruff for linting"
	@echo "  format     - Run Ruff for formatting"
	@echo "  fix        - Run Ruff to fix linting issues"
	@echo "  clean      - Remove temporary files and caches"

install:
	cd server && uv sync

run:
	cd server && uv run python main.py

lint:
	cd server && uv run ruff check .

format:
	cd server && uv run ruff format .

fix:
	cd server && uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
