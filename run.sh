#!/bin/bash
# Stop immediately if a command fails
set -e

# Change directory to the workspace root
cd "$(dirname "$0")"

# Check if .venv exists, if not initialize it
if [ ! -d ".venv" ]; then
    echo "[run.sh] Virtual environment (.venv) not found. Setting up..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Load settings from .env file if it exists
if [ -f ".env" ]; then
    echo "[run.sh] Loading configuration from .env..."
    # Export all variables from .env except commented lines
    export $(grep -v '^#' .env | xargs)
fi

# Set host/port defaults if they weren't defined in .env or system environment
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "[run.sh] Starting Iris Engine on http://$HOST:$PORT..."
exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
