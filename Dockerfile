# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --from=builder --chown=app:app /app /app
USER app
EXPOSE 8000

CMD ["uvicorn", "document_qa.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]
