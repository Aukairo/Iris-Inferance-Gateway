# Iris Engine - Local MLX LLM Server for Apple Silicon

Iris Engine is a production-quality local LLM inference server optimized for Apple Silicon (M-series Macs) using **MLX** and **mlx-lm**.

This server exposes an OpenAI-compatible API (`/v1/chat/completions`), supports dynamic continuous batching, streaming, optional Bearer token authentication, Prometheus metrics, and structured JSON logging. It acts similarly to a lightweight, Metal-accelerated version of vLLM built specifically for macOS.

---

## Why We Built Iris Engine

While excellent tools exist for running LLMs locally, they cater to different needs:
* **`mlx-lm`**: This is Apple's official framework, meaning it is highly optimized for Mac hardware and unified memory. However, it is designed primarily for CLI use or basic Python scripts—it lacks a robust sharing gateway, client isolation, token/budget rate limits, and an endpoint that can be easily/securely accessed externally by multiple client apps.
* **`Ollama`**: While it offers many features and desktop integrations, it runs on a general-purpose C++ framework (llama.cpp) designed for cross-platform compatibility rather than being specifically tailored and natively optimized for the Apple Silicon MLX ecosystem. Additionally, it operates as a single-tenant system without fine-grained key management, token capping, or developer-facing cost controls.

We built **Iris Engine** to serve as a high-performance local AI gateway. It combines Apple’s official low-level **MLX framework** with professional gateway capabilities (capping, token allocation, pricing simulator, and live dashboards) to share local unified memory resources across multiple developer tools, local agents, and test suites.

## Core Advantages

* **True Concurrency via Continuous Batching:** Merges concurrent requests mid-generation to maximize Apple Silicon unified memory bandwidth.
* **One-Click Public Sharing:** Start a secure, public HTTPS tunnel directly from the Admin Panel to share your local models globally with zero manual port forwarding or command-line installation.
* **Multi-Tenant Key Management:** Issue distinct API keys with individual token and cost caps, preventing buggy loops or excessive loads from starving other client applications.
* **Cost & Budget Attribution:** Define input, output, and cached token prices to calculate actual USD usage equivalents and enforce limits automatically.
* **Developer-Ready Observability:** Export performance metrics to Prometheus and view live analytics inside a clean dark-mode Admin Dashboard.

## Feature Comparison

| Feature | `mlx-lm` (CLI/Server) | Ollama | Iris Engine |
| :--- | :---: | :---: | :---: |
| **Concurrency Model** | Sequential | Queue-based (Blocks under load) | **Continuous Dynamic Batching** |
| **Multi-Key Isolation** | ❌ No | ❌ No | **Yes (Bearer Tokens)** |
| **Token & USD Budget Caps** | ❌ No | ❌ No | **Yes (Per-Key Limits)** |
| **Built-in Public Tunneling** | ❌ No | ❌ No | **Yes (One-click Cloudflare Tunnels)** |
| **Observability** | Console Logs | System Logs | **Prometheus Metrics & Dashboard** |
| **Cost Attribution** | ❌ No | ❌ No | **Yes (Pricing configuration)** |
| **Admin Control Dashboard** | ❌ No | ❌ No | **Yes (Built-in Web Interface)** |

---

## Architecture Overview

To ensure the FastAPI event loop is never blocked during heavy Metal computations, the server runs a decoupled multi-threaded processing architecture:

```
[Client Request]
       │ (HTTP POST /v1/chat/completions)
       ▼
[FastAPI Route Handler]
       │
       ▼ (Enqueue Request Context)
[Async Queue]
       │
       ▼ (dynamic window wait / maximum batch sizes)
[Batch Scheduler]
       │
       ▼ (continuous batch insertion)
[Inference Worker (Background Thread)] ──► [MLX Model / Metal GPU Acceleration]
       │
       ▼ (Incremental Streaming Detokenization)
[Client Response Queue (asyncio.Queue)]
       │
       ▼ (yield SSE format chunks)
[StreamingResponse / JSON Payload]
```

* **Continuous Dynamic Batching**: Incoming prompts are collected during a configurable batching window (default `25ms`). New requests are dynamically inserted into the running model generation batch on-the-fly, avoiding head-of-line blocking and maximizing unified memory bandwidth.
* **Persistent In-Memory Model**: The model is loaded once at server startup and kept in memory. No reloading occurs between requests.

---

## Requirements

* **Hardware**: Apple Silicon Mac (M1, M2, M3, M4, M5, etc.) with Unified Memory (16GB+ recommended).
* **OS**: macOS
* **Software**: Python 3.12+

---

## Getting Started

### Local Setup & Launch

The server can be fully initialized and started using the provided `run.sh` script or the `Makefile`:

```bash
# 1. Clone or copy this repository into your local directory
cd llm_server

# 2. Make run.sh executable (if not already done)
chmod +x run.sh

# 3. Start the server (this automatically creates the .venv and installs dependencies)
./run.sh
```

Or using `make`:
```bash
# Install dependencies into .venv
make install

# Start the server locally
make run
```

---

## Configuration

Settings can be customized using environment variables or a local `.env` file. Copy the example file to begin:

```bash
cp .env.example .env
```

### Configurable parameters:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MODEL_NAME` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Hugging Face model ID or path to a local MLX model. |
| `HOST` | `127.0.0.1` | Network interface to bind the server to. |
| `PORT` | `8000` | Port to run the server on. |
| `TEMPERATURE` | `0.7` | Default sampling temperature. |
| `TOP_P` | `0.9` | Default top-p sampling parameter. |
| `MAX_TOKENS` | `2048` | Default maximum tokens to generate per completion. |
| `BATCHING_WINDOW_MS` | `25` | Window size in milliseconds to group concurrent requests. |
| `MAX_BATCH_SIZE` | `16` | Maximum number of concurrent sequences processed in one batch. |
| `API_KEY` | *(None)* | Optional Bearer authentication token. Leave blank to disable auth. |
| `ADMIN_PASSWORD` | `admin123` | Master password to protect the `/admin` dashboard and key management APIs. |
| `LOGGING_LEVEL` | `INFO` | Logger output level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `METRICS_ENABLED` | `true` | Enables or disables Prometheus telemetry `/metrics` endpoint. |

---

## API Authentication & Token Capping

The server features a multi-tenant API key tracking and capping database that persists locally in `data/keys.json`.

### 1. Security Configuration
* **Authentication Enforcement**: If `API_KEY` is set in `.env` OR if any active keys are created in the database, the server will reject requests missing a valid Bearer token (`Authorization: Bearer <key>`) with a `401 Unauthorized` status.
* **Master Password**: Protect the admin UI and CRUD key management APIs by setting the `ADMIN_PASSWORD` variable in your `.env` file.

### 2. Admin Dashboard UI (`/admin`)
Navigate to `http://127.0.0.1:8000/admin` in your browser to access the management interface:
* **Authentication**: Login with your configured `ADMIN_PASSWORD`.
* **Dynamic Keys**: Generate unique `sk-` prefix bearer tokens for different clients or projects.
* **Token Capping**: Limit key budgets (e.g., maximum 50,000 total tokens). The server automatically rejects completions with a `403 Forbidden` ("API Key usage limit exceeded") once reached.
* **Metrics Telemetry**: View total registered keys, active keys, live telemetry, and prompt/generation token breakdowns per user.
* **Actions**: Adjust budget limits, reset usage counters, or revoke/delete keys instantly.

---

## Supported Models

Any quantized 4-bit, 8-bit, or full-precision model converted to MLX format is supported. Prominent examples on Hugging Face:

* **Qwen**: `mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Qwen2.5-7B-Instruct-4bit`
* **Llama**: `mlx-community/Llama-3.2-3B-Instruct-4bit`, `mlx-community/Meta-Llama-3-8B-Instruct-4bit`
* **Gemma**: `mlx-community/gemma-3-4b-it-4bit`

---

## API Endpoints

### 1. Chat Completions (`POST /v1/chat/completions`)
Generate assistant replies matching the OpenAI schema.

**Non-Streaming Example**:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the primary colors?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "stream": false
  }'
```

**Streaming Example (`stream=true`)**:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a one-sentence tagline for a fast coding assistant."}
    ],
    "stream": true
  }'
```

---

### 2. Health Check (`GET /health`)
Returns the model status, active queue information, server uptime, and RSS memory usage (on macOS).

```bash
curl http://127.0.0.1:8000/health
```
**Example Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "mlx-community/Qwen2.5-3B-Instruct-4bit",
  "queue_length": 0,
  "active_generations": 0,
  "uptime_seconds": 120,
  "memory_usage_mb": 4200.5
}
```

---

### 3. Prometheus Metrics (`GET /metrics`)
Exposes live telemetry for dashboarding. Includes:
* `mlx_server_requests_total`: Request counter (labeled by success/error status).
* `mlx_server_queue_length`: Current queue length.
* `mlx_server_active_generations`: Active generation slots.
* `mlx_server_batch_size`: Histogram of processed batch sizes.
* `mlx_server_request_latency_seconds`: End-to-end latency histogram.
* `mlx_server_time_to_first_token_seconds`: TTFT distribution.
* `mlx_server_prompt_tokens_total` / `mlx_server_generated_tokens_total`: Token count tracking.
* `mlx_server_tokens_generated_speed_tps`: Distribution of generation speed (tokens per second).

```bash
curl http://127.0.0.1:8000/metrics
```

---

## Client Integration Examples

### Python (OpenAI SDK)

Install the OpenAI python client: `pip install openai`

```python
from openai import OpenAI

# Initialize client. Use API Key if configured, otherwise provide any string.
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-api-key-here"  # Optional
)

# Non-streaming request
response = client.chat.completions.create(
    model="mlx-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain unified memory in one sentence."}
    ],
    temperature=0.7,
    max_tokens=100
)
print("Response:", response.choices[0].message.content)

# Streaming request
stream = client.chat.completions.create(
    model="mlx-model",
    messages=[
        {"role": "user", "content": "Tell me a short story."}
    ],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

### Node.js (OpenAI SDK)

Install the OpenAI npm package: `npm install openai`

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://127.0.0.1:8000/v1',
  apiKey: 'your-api-key-here', // Optional
});

async function main() {
  const stream = await openai.chat.completions.create({
    model: 'mlx-model',
    messages: [{ role: 'user', content: 'What is Metal in Apple Silicon?' }],
    stream: true,
  });
  
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
  console.log();
}

main().catch(console.error);
```

---

## Telemetry & Benchmarking

To measure throughput and latency under concurrent stress:

1. Launch the server locally.
2. Install `hey` or use `curl` parallel triggers:
   ```bash
   # Run 100 requests with concurrency level 5
   hey -n 100 -c 5 -m POST \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Summarize Apple Silicon GPU structure."}], "max_tokens": 64}' \
     http://127.0.0.1:8000/v1/chat/completions
   ```
3. Fetch the `/metrics` or `/health` endpoint to monitor peak memory utilization and generation throughput in tokens/second.

---

## Testing

Run unit tests via `pytest` to confirm server modules:

```bash
make test
```
*All tests run completely offline by mocking MLX internals to prevent long downloads or hardware requirements in continuous integration settings.*
