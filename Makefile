.PHONY: install run test clean docker-build docker-run help

help:
	@echo "Available commands:"
	@echo "  install      - Create virtual environment and install dependencies"
	@echo "  run          - Run the server locally using uvicorn"
	@echo "  test         - Run unit tests with pytest"
	@echo "  clean        - Remove build artifacts, cached files, and virtual environment"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Build and run the server using docker-compose"

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

test:
	.venv/bin/pytest -v

clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t mlx-inference-server:latest .

docker-run:
	docker-compose up --build
