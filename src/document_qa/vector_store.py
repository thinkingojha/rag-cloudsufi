"""Durable Qdrant-backed storage for chunks and their citation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from qdrant_client import QdrantClient, models

from document_qa.rag import TOP_K, Chunk
from document_qa.settings import Settings


@dataclass(frozen=True)
class CorpusMetadata:
    corpus_id: str
    document_names: list[str]
    chunk_count: int
    expires_at: datetime
    fingerprint: str


class QdrantCorpusStore:
    """Persistence adapter for the vector store.

    Chunks and corpus metadata share a point payload. This keeps all retrieval state durable
    and makes the API stateless; any replica can serve a previously indexed corpus.
    """

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        self._client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
            timeout=10,
        )

    def initialize(self) -> None:
        if self._client.collection_exists(self._settings.qdrant_collection):
            return
        self._client.create_collection(
            collection_name=self._settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=self._settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        self._client.create_payload_index(
            collection_name=self._settings.qdrant_collection,
            field_name="corpus_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._settings.qdrant_collection,
            field_name="expires_at_epoch",
            field_schema=models.PayloadSchemaType.FLOAT,
        )
        self._client.create_payload_index(
            collection_name=self._settings.qdrant_collection,
            field_name="fingerprint",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def ready(self) -> bool:
        self._client.get_collection(self._settings.qdrant_collection)
        return True

    def find_by_fingerprint(self, fingerprint: str, now: datetime) -> CorpusMetadata | None:
        records, _ = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=self._active_filter(fingerprint=fingerprint, now=now),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return self._metadata_from_payload(records[0].payload) if records else None

    def upsert(
        self,
        metadata: CorpusMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "corpus_id": metadata.corpus_id,
                    "document_names": metadata.document_names,
                    "chunk_count": metadata.chunk_count,
                    "expires_at": metadata.expires_at.isoformat(),
                    "expires_at_epoch": metadata.expires_at.timestamp(),
                    "fingerprint": metadata.fingerprint,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._client.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
            wait=True,
        )

    def search(self, corpus_id: str, query_vector: list[float], now: datetime) -> list[Chunk]:
        response = self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=query_vector,
            query_filter=self._active_filter(corpus_id=corpus_id, now=now),
            limit=TOP_K,
            with_payload=True,
        )
        return [
            Chunk(
                text=str(point.payload["text"]),
                document_name=str(point.payload["document_name"]),
                page_number=int(point.payload["page_number"]),
            )
            for point in response.points
        ]

    def delete(self, corpus_id: str) -> bool:
        records, _ = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="corpus_id",
                        match=models.MatchValue(value=corpus_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not records:
            return False
        self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="corpus_id", match=models.MatchValue(value=corpus_id)
                        )
                    ]
                )
            ),
            wait=True,
        )
        return True

    def purge_expired(self, now: datetime) -> None:
        self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="expires_at_epoch", range=models.Range(lte=now.timestamp())
                        )
                    ]
                )
            ),
            wait=False,
        )

    def _active_filter(
        self,
        *,
        now: datetime,
        corpus_id: str | None = None,
        fingerprint: str | None = None,
    ) -> models.Filter:
        conditions: list[models.FieldCondition] = [
            models.FieldCondition(key="expires_at_epoch", range=models.Range(gt=now.timestamp()))
        ]
        if corpus_id:
            conditions.append(
                models.FieldCondition(key="corpus_id", match=models.MatchValue(value=corpus_id))
            )
        if fingerprint:
            conditions.append(
                models.FieldCondition(key="fingerprint", match=models.MatchValue(value=fingerprint))
            )
        return models.Filter(must=conditions)

    @staticmethod
    def _metadata_from_payload(payload: dict[str, object] | None) -> CorpusMetadata:
        if payload is None:
            raise ValueError("Qdrant record did not include a payload.")
        return CorpusMetadata(
            corpus_id=str(payload["corpus_id"]),
            document_names=[str(name) for name in payload["document_names"]],  # type: ignore[index]
            chunk_count=int(payload["chunk_count"]),
            expires_at=datetime.fromisoformat(str(payload["expires_at"])).astimezone(UTC),
            fingerprint=str(payload["fingerprint"]),
        )
