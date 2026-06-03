FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/workspace/wam-harness/.venv/bin:${PATH}"

WORKDIR /workspace/wam-harness

RUN python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml README.md uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev

CMD ["wam", "--help"]
