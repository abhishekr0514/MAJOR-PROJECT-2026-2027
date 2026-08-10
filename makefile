# Makefile for MedShield FL Framework

.PHONY: help install client-install frontend-install run fl-server fl-client frontend test test-fl test-client lint format fix clean migrate makemigration seed structure

help:
	@echo "MedShield FL - Available Commands:"
	@echo "  install          - Install backend server dependencies (uv sync)"
	@echo "  client-install   - Install client ML/FL dependencies (uv sync)"
	@echo "  frontend-install - Install React.js frontend dependencies (npm install)"
	@echo "  run              - Start FastAPI backend server (port 8000)"
	@echo "  fl-server        - Start central Flower FL server aggregator (port 8080)"
	@echo "  fl-client        - Start hospital client node (hospital_alpha)"
	@echo "  frontend         - Start React.js frontend dev server (Vite)"
	@echo "  test             - Run all backend server and FL round integration tests"
	@echo "  test-fl          - Run multi-hospital FL round simulation test"
	@echo "  test-client      - Run client ML, privacy, and XAI test suite"
	@echo "  lint             - Run Ruff linting across server and client"
	@echo "  format           - Run Ruff formatting across server and client"
	@echo "  fix              - Run Ruff automatic fixes across server and client"
	@echo "  migrate          - Apply database migrations (alembic upgrade head)"
	@echo "  makemigration    - Generate a new database migration (requires m='message')"
	@echo "  seed             - Seed initial database with super admin & hospital data"
	@echo "  structure        - Show directory structure tree"
	@echo "  clean            - Clear cache directories (__pycache__, .pytest_cache, etc.)"

install:
	cd server && uv sync --index-strategy unsafe-best-match --index https://download.pytorch.org/whl/cpu

client-install:
	cd client && uv sync --index-strategy unsafe-best-match --index https://download.pytorch.org/whl/cpu

frontend-install:
	cd frontend && npm install

run:
	cd server && uv run python main.py

fl-server:
	cd server && uv run python -m app.features.federation.fl_server --rounds 5 --port 8080 --min-clients 1

fl-client:
	cd client && uv run python fl_client.py --server 127.0.0.1:8080 --hospital-id hospital_alpha

frontend:
	cd frontend && npm run dev

test:
	cd server && uv run pytest tests/ -v -s

test-fl:
	cd server && uv run pytest tests/test_fl_round.py -v -s

test-client:
	cd client && uv run pytest tests/ -v

lint:
	cd server && uv run ruff check .
	cd client && uv run ruff check .

format:
	cd server && uv run ruff format .
	cd client && uv run ruff format .

fix:
	cd server && uv run ruff check --fix . && uv run ruff format .
	cd client && uv run ruff check --fix . && uv run ruff format .

migrate:
	cd server && uv run alembic upgrade head

makemigration:
	cd server && uv run alembic revision --autogenerate -m "$(m)"

seed:
	cd server && uv run python seed.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +

structure:
	@tree -I "__pycache__|.git|.venv|venv|.ruff_cache|.pytest_cache|.mypy_cache|.vscode|.idea|.env|*.pyc|uv.lock|node_modules|dist"
