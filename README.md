# Document Q&A Service

A small, service-oriented RAG application for uploading one to three PDFs and asking grounded questions. It returns document-and-page citations and exposes the passages retrieved for every answer.

OpenAI provides embeddings and answer generation; Qdrant durably stores vectors and citation metadata. The FastAPI layer is stateless and can therefore be replicated behind a load balancer.

## Quick start

### Docker Compose (recommended)

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501) for the UI. Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Local development with uv

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
cp .env.example .env
uv sync --all-groups
uv run uvicorn document_qa.api:app --reload --port 8000
```

In a second terminal, run the web client:

```bash
uv run streamlit run src/document_qa/web.py
```

Quality checks:

```bash
uv run ruff check .
uv run pytest --cov=document_qa
```

## Architecture

```text
Streamlit web client ──HTTP──> FastAPI service ──> OpenAI embeddings/chat
                                      │
                                      ├─ PDF extraction and chunking
                                      ├─ Qdrant vector search (top 5)
                                      ├─ page-aware citation payloads
                                      └─ expiry + idempotent upload fingerprints
```

1. `POST /v1/corpora` accepts one to three PDFs, validates MIME type, file signature, per-file/combined size, extracts page text, chunks it, and indexes it with OpenAI embeddings in Qdrant.
2. A SHA-256 fingerprint prevents retries or repeat uploads from paying to create the same index twice while it remains active.
3. `POST /v1/corpora/{id}/questions` embeds the question, queries the five closest non-expired chunks from Qdrant, and asks the chat model to answer only from them.
4. The answer includes model-rendered page citations such as `[report.pdf, p. 3]`; trusted citation metadata and excerpts are returned separately in the response.
5. `DELETE /v1/corpora/{id}` deletes the corpus. Active corpus data expires after the configured TTL and the API periodically removes expired vectors during new index operations.

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness probe; no provider credentials required. |
| `GET /readyz` | Readiness probe; confirms configuration is present. |
| `GET /metrics` | Prometheus metrics; protected when API authentication is enabled. |
| `POST /v1/corpora` | Create an index from 1–3 PDF files. |
| `POST /v1/corpora/{corpus_id}/questions` | Ask a question about an indexed corpus. |
| `DELETE /v1/corpora/{corpus_id}` | Remove a temporary corpus. |

## Project layout

```text
src/document_qa/
  api.py          FastAPI routes, middleware, health checks
  service.py      application orchestration
  vector_store.py durable Qdrant persistence adapter
  rag.py          extraction, chunking, retrieval, LLM calls
  schemas.py      typed HTTP contracts
  settings.py     environment configuration
  web.py          Streamlit API client
tests/            fast unit tests
.github/          CI for linting and tests
pyproject.toml    uv dependencies and tooling
```

## Operational notes

- The containers run as a non-root user, include dependency-aware health checks, emit request IDs, and use CORS allow-listing.
- Uploads are limited to three PDFs, a configurable per-file size (`MAX_UPLOAD_MB`, default 20 MB), and a combined request limit (`MAX_TOTAL_UPLOAD_MB`, default 50 MB). MIME type and PDF magic bytes are checked before parsing.
- Qdrant uses a named Docker volume, so indexes survive API restarts. Each point includes the corpus ID, expiry, fingerprint, document/page metadata, and text required for verifiable citations.
- OpenAI clients use explicit timeouts and limited retries. Prometheus metrics cover HTTP traffic, index creation, chunk volume, and answered questions.
- `API_AUTH_TOKEN` is optional in development but required with `APP_ENV=production`; the Streamlit client forwards it using `X-API-Key`. Never commit `.env`; it is ignored by Git.

## Limitations

- Text-based PDFs only; scanned documents need an OCR stage.
- Retrieval is dense-vector only. Hybrid BM25/vector search plus reranking would improve difficult queries.
- Citations are page-level, not paragraph/bounding-box anchors.
- LLM responses must be reviewed in high-stakes settings, even when source excerpts are exposed.

## Next production steps

- Add tenant-aware metadata, per-user quotas, malware scanning, object-storage uploads, and rate limiting at the edge.
- Add OpenTelemetry traces, provider-cost attribution, and retrieval-quality evaluation fixtures.
- Add asynchronous indexing with a queue for larger files and a PDF viewer with highlighted citations.
- For highly sensitive workloads, use a private embedding/LLM endpoint and encrypt uploaded originals in object storage.
