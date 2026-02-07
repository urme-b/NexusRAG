.PHONY: install install-dev test test-cov lint format type-check run run-api clean help docker-build docker-up docker-down docker-logs

PYTHON := python3
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src/nexusrag --cov-report=html --cov-report=term

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

type-check:
	$(PYTHON) -m mypy src/

run:
	$(PYTHON) -m uvicorn nexusrag.api:app --reload --host 0.0.0.0 --port 8000

run-gradio:
	$(PYTHON) -m gradio app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

help:
	@echo "Available commands:"
	@echo "  install        Install package"
	@echo "  install-dev    Install with dev dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Check code style"
	@echo "  format         Format code"
	@echo "  type-check     Run type checking"
	@echo "  run            Start FastAPI server with premium UI"
	@echo "  run-gradio     Start legacy Gradio UI"
	@echo "  clean          Remove build artifacts"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-up      Start services with Docker Compose"
	@echo "  docker-down    Stop Docker Compose services"
	@echo "  docker-logs    Tail Docker Compose logs"
