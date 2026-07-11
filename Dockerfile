# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install compiler dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a separate wheels directory to speed up building
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /root/.local /root/.local
COPY app/ /app/app/
COPY .env.example /app/.env

# Update PATH to include user-installed scripts
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
