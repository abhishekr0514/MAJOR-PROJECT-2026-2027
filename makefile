# Makefile for MedShield FL Framework (Cross-Platform: Windows, Linux, macOS)

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    CLEAN_CMD := powershell -Command "Get-ChildItem -Recurse -Include '__pycache__','.ruff_cache','.pytest_cache' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
    TREE_CMD := powershell -Command "Get-ChildItem -Recurse -Exclude '__pycache__','.git','.venv','node_modules','dist' | Select-Object FullName"
else
    DETECTED_OS := Unix
    CLEAN_CMD := find . -type d \( -name "__pycache__" -o -name ".ruff_cache" -o -name ".pytest_cache" \) -exec rm -rf {} + 2>/dev/null || true
    TREE_CMD := tree -I "__pycache__|.git|.venv|venv|.ruff_cache|.pytest_cache|.mypy_cache|.vscode|.idea|.env|*.pyc|uv.lock|node_modules|dist"
endif

.PHONY: help install client-install frontend-install run fl-server fl-client frontend generate-api test test-fl test-client lint format fix clean migrate makemigration seed structure

help:
	@echo "MedShield FL - Available Commands (OS: $(DETECTED_OS)):"
	@echo "  install          - Install backend server dependencies (uv sync)"
	@echo "  client-install   - Install client ML/FL dependencies (uv sync)"
	@echo "  frontend-install - Install React.js frontend dependencies (npm install)"
	@echo "  run              - Start FastAPI backend server (port 8000)"
	@echo "  fl-server        - Start central Flower FL server aggregator (port 8080)"
	@echo "  fl-client        - Start hospital client node (hospital_alpha)"
	@echo "  frontend         - Start React.js frontend dev server (Vite)"
	@echo "  generate-api     - Regenerate API SDK from OpenAPI specification"
	@echo "  test             - Run all backend server and FL round integration tests"
	@echo "  test-fl          - Run multi-hospital FL round simulation test"
	@echo "  test-client      - Run client ML, privacy, and XAI test suite"
	@echo "  lint             - Run Ruff linting across server and client"
	@echo "  format           - Run Ruff formatting across server and client"
	@echo "  fix              - Run Ruff automatic fixes across server and client"
	@echo "  migrate          - Apply database migrations (alembic upgrade head)"
	@echo "  makemigration    - Generate a new database migration (requires m='message')"
	@echo "  seed             - Seed initial database with super admin & hospital data"
	@echo "  clean            - Clear cache directories (__pycache__, .pytest_cache, etc.)"
	@echo "  structure        - Show directory structure tree"

install:
	cd server && uv sync --index-strategy unsafe-best-match

client-install:
	cd client && uv sync --index-strategy unsafe-best-match

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

generate-api:
	cd frontend && npm run generate:api

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
	@$(CLEAN_CMD)

structure:
	@$(TREE_CMD)
