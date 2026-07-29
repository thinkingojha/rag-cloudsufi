"""Application service coordinating validation, retrieval, and model calls."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, OpenAI

from document_qa.rag import (
    Chunk,
    corpus_fingerprint,
    embed_texts,
    extract_pdf_chunks,
    generate_answer,
    group_chunks_by_source,
)
from document_qa.schemas import AnswerResponse, Citation
from document_qa.settings import Settings
from document_qa.vector_store import CorpusMetadata, QdrantCorpusStore


class DependencyUnavailableError(RuntimeError):
    """Raised when an upstream provider or vector store cannot serve a request."""


class DocumentQAService:
    def __init__(
        self,
        settings: Settings,
        store: QdrantCorpusStore,
        client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._client = client

    def initialize(self) -> None:
        try:
            self._store.initialize()
        except Exception as exc:
            raise DependencyUnavailableError("Vector store initialization failed.") from exc

    def ready(self) -> bool:
        if not self._settings.openai_api_key:
            return False
        try:
            return self._store.ready()
        except Exception:
            return False

    def create_corpus(self, documents: list[tuple[str, bytes]]) -> tuple[CorpusMetadata, bool]:
        client = self._openai_client()
        now = datetime.now(UTC)
        fingerprint = corpus_fingerprint(documents)
        try:
            self._store.purge_expired(now)
            existing = self._store.find_by_fingerprint(fingerprint, now)
        except Exception as exc:
            raise DependencyUnavailableError("Vector store is unavailable.") from exc
        if existing:
            return existing, True

        chunks = [
            chunk
            for name, payload in documents
            for chunk in extract_pdf_chunks(name, payload)
        ]
        if not chunks:
            raise ValueError("No extractable text was found in the uploaded PDFs.")
        embeddings = self._embed(client, (chunk.text for chunk in chunks))
        metadata = CorpusMetadata(
            corpus_id=str(uuid4()),
            document_names=[name for name, _ in documents],
            chunk_count=len(chunks),
            expires_at=now + timedelta(minutes=self._settings.corpus_ttl_minutes),
            fingerprint=fingerprint,
        )
        try:
            self._store.upsert(metadata, chunks, embeddings)
        except Exception as exc:
            raise DependencyUnavailableError("Vector store is unavailable.") from exc
        return metadata, False

    def ask(self, corpus_id: str, question: str) -> AnswerResponse | None:
        client = self._openai_client()
        query_vector = self._embed(client, [question])[0]
        try:
            chunks = self._store.search(corpus_id, query_vector, datetime.now(UTC))
        except Exception as exc:
            raise DependencyUnavailableError("Vector store is unavailable.") from exc
        if not chunks:
            return None
        answer = self._generate(client, question, chunks)
        citations = [
            Citation(
                document_name=source.document_name,
                page_number=source.page_number,
                excerpt="\n\n".join(excerpts)[:1_000],
            )
            for source, excerpts in group_chunks_by_source(chunks)
        ]
        return AnswerResponse(answer=answer, citations=citations)

    def delete(self, corpus_id: str) -> bool:
        try:
            return self._store.delete(corpus_id)
        except Exception as exc:
            raise DependencyUnavailableError("Vector store is unavailable.") from exc

    def _embed(self, client: OpenAI, texts: Iterable[str]) -> list[list[float]]:
        try:
            return embed_texts(client, texts, self._settings.openai_embedding_model)
        except (APIConnectionError, APIStatusError) as exc:
            raise DependencyUnavailableError("Embedding provider is unavailable.") from exc

    def _generate(self, client: OpenAI, question: str, chunks: list[Chunk]) -> str:
        try:
            return generate_answer(client, question, chunks, self._settings.openai_chat_model)
        except (APIConnectionError, APIStatusError) as exc:
            raise DependencyUnavailableError("Answer provider is unavailable.") from exc

    def _openai_client(self) -> OpenAI:
        if self._client:
            return self._client
        api_key = self._settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        return OpenAI(api_key=api_key.get_secret_value(), timeout=30.0, max_retries=2)
