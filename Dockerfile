FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PATH=/app/.venv/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

ARG TORCH_BACKEND=cpu
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-emit-project --no-hashes | grep -vE "^(nvidia-|triton)" > requirements.txt \
    && uv venv \
    && uv pip install --torch-backend "$TORCH_BACKEND" -r requirements.txt
COPY src ./src
RUN uv pip install --no-deps -e .

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "listing_guard.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
