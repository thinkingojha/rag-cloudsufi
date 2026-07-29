"""FastAPI application exposing the document question-answering service."""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from document_qa.metrics import (
    CHUNKS_INDEXED,
    CORPORA_CREATED,
    HTTP_DURATION,
    HTTP_REQUESTS,
    QUESTIONS_ANSWERED,
)
from document_qa.schemas import AnswerResponse, CorpusCreated, QuestionRequest, ServiceStatus
from document_qa.service import DependencyUnavailableError, DocumentQAService
from document_qa.settings import Settings, get_settings
from document_qa.vector_store import QdrantCorpusStore

logger = logging.getLogger("document_qa")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service = DocumentQAService(settings, QdrantCorpusStore(settings))
    service.initialize()
    app.state.service = service
    logger.info("service_started environment=%s", settings.app_env)
    yield
    logger.info("service_stopped")


app = FastAPI(
    title="Document Q&A API",
    version="0.1.0",
    description="Upload up to three PDFs and ask grounded questions with page citations.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next: object) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    started_at = time.perf_counter()
    try:
        response = await call_next(request)  # type: ignore[operator]
    except Exception:
        logger.exception("request_failed request_id=%s path=%s", request_id, request.url.path)
        raise
    route = request.scope.get("route")
    route_path = getattr(route, "path", request.url.path)
    HTTP_REQUESTS.labels(request.method, route_path, response.status_code).inc()
    HTTP_DURATION.labels(request.method, route_path).observe(time.perf_counter() - started_at)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
    )
    return response


@app.exception_handler(ValueError)
async def validation_error(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def get_service(request: Request) -> DocumentQAService:
    return request.app.state.service  # type: ignore[no-any-return]


def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_auth_token:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(supplied, settings.api_auth_token.get_secret_value()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


@app.get("/healthz", response_model=ServiceStatus, tags=["operations"])
def healthz() -> ServiceStatus:
    return ServiceStatus(status="ok")


@app.get("/readyz", response_model=ServiceStatus, tags=["operations"])
def readyz(service: DocumentQAService = Depends(get_service)) -> ServiceStatus:
    if not service.ready():
        raise HTTPException(status_code=503, detail="A required dependency is unavailable.")
    return ServiceStatus(status="ready")


@app.get("/metrics", include_in_schema=False)
def metrics(_: None = Depends(require_api_key)) -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/v1/corpora",
    response_model=CorpusCreated,
    status_code=status.HTTP_201_CREATED,
    tags=["corpora"],
)
async def create_corpus(
    files: list[UploadFile] = File(...),
    service: DocumentQAService = Depends(get_service),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_api_key),
) -> CorpusCreated:
    if not 1 <= len(files) <= 3:
        raise HTTPException(status_code=422, detail="Upload between one and three PDF files.")

    documents: list[tuple[str, bytes]] = []
    total_size = 0
    for file in files:
        if file.content_type not in {"application/pdf", "application/x-pdf"}:
            raise HTTPException(status_code=415, detail=f"'{file.filename}' is not a PDF.")
        payload = await file.read()
        if not payload.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=415,
                detail=f"'{file.filename}' is not a valid PDF file.",
            )
        if len(payload) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"'{file.filename}' exceeds the upload size limit.",
            )
        total_size += len(payload)
        if total_size > settings.max_total_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Combined upload size exceeds the limit.")
        documents.append((file.filename or "document.pdf", payload))

    try:
        stored, reused = await run_in_threadpool(service.create_corpus, documents)
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not reused:
        CORPORA_CREATED.inc()
        CHUNKS_INDEXED.inc(stored.chunk_count)
    return CorpusCreated(
        corpus_id=stored.corpus_id,
        documents=stored.document_names,
        chunk_count=stored.chunk_count,
        expires_at=stored.expires_at,
        reused=reused,
    )


@app.post("/v1/corpora/{corpus_id}/questions", response_model=AnswerResponse, tags=["questions"])
def answer_question(
    corpus_id: str,
    request: QuestionRequest,
    service: DocumentQAService = Depends(get_service),
    _: None = Depends(require_api_key),
) -> AnswerResponse:
    try:
        answer = service.ask(corpus_id, request.question)
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if answer is None:
        raise HTTPException(status_code=404, detail="Corpus not found or expired.")
    QUESTIONS_ANSWERED.inc()
    return answer


@app.delete("/v1/corpora/{corpus_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["corpora"])
def delete_corpus(
    corpus_id: str,
    service: DocumentQAService = Depends(get_service),
    _: None = Depends(require_api_key),
) -> Response:
    try:
        deleted = service.delete(corpus_id)
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Corpus not found or expired.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
