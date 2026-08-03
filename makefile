# Makefile for Major Project 2026-2027

.PHONY: help install run client-install lint format fix clean migrate makemigration seed structure test

help:
	@echo "Available commands:"
	@echo "  install        - Install backend dependencies using uv"
	@echo "  client-install - Install client ML/FL dependencies using uv"
	@echo "  run            - Start the FastAPI server"
	@echo "  lint           - Run Ruff for linting across server & client"
	@echo "  format         - Run Ruff for formatting across server & client"
	@echo "  fix            - Run Ruff to fix linting issues across server & client"
	@echo "  test           - Run pytest unit and integration tests"
	@echo "  migrate        - Apply database migrations (alembic)"
	@echo "  makemigration  - Generate a new migration (requires m='message')"
	@echo "  seed           - Create initial super admin and seed data"
	@echo "  structure      - Show folder structure"
	@echo "  clean          - Remove temporary files and caches"

install:
	cd server && uv sync

client-install:
	cd client && uv sync

run:
	cd server && uv run python main.py

migrate:
	cd server && uv run alembic upgrade head

makemigration:
	cd server && uv run alembic revision --autogenerate -m "$(m)"

seed:
	cd server && uv run python seed.py

test:
	cd server && uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

structure:
	@tree -I "__pycache__|.git|.venv|venv|.ruff_cache|.pytest_cache|.mypy_cache|.vscode|.idea|.env|*.pyc|uv.lock"
