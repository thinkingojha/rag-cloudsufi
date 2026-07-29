from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient

from document_qa.rag import Chunk
from document_qa.settings import Settings
from document_qa.vector_store import CorpusMetadata, QdrantCorpusStore


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created_collection = False
        self.indexed_fields: list[str] = []

    def collection_exists(self, _: str) -> bool:
        return False

    def create_collection(self, **_: object) -> None:
        self.created_collection = True

    def create_payload_index(self, *, field_name: str, **_: object) -> None:
        self.indexed_fields.append(field_name)


def test_store_initialization_creates_collection_and_indexes() -> None:
    client = FakeQdrantClient()
    store = QdrantCorpusStore(Settings(), client=client)  # type: ignore[arg-type]

    store.initialize()

    assert client.created_collection is True
    assert client.indexed_fields == ["corpus_id", "expires_at_epoch", "fingerprint"]


def test_active_filter_limits_results_to_unexpired_corpus() -> None:
    store = QdrantCorpusStore(Settings(), client=FakeQdrantClient())  # type: ignore[arg-type]

    query_filter = store._active_filter(  # noqa: SLF001 - deliberate unit-level contract test
        corpus_id="corpus-1",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(query_filter.must) == 2


def test_store_persists_searches_and_deletes_chunks() -> None:
    settings = Settings(qdrant_collection="test_chunks", embedding_dimensions=2)
    store = QdrantCorpusStore(settings, client=QdrantClient(":memory:"))
    metadata = CorpusMetadata(
        corpus_id="corpus-1",
        document_names=["guide.pdf"],
        chunk_count=1,
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        fingerprint="fingerprint",
    )
    with pytest.warns(UserWarning, match="Payload indexes have no effect"):
        store.initialize()
    store.upsert(metadata, [Chunk("retrieved text", "guide.pdf", 3)], [[0.1, 0.9]])

    results = store.search("corpus-1", [0.1, 0.9], datetime(2029, 1, 1, tzinfo=UTC))

    assert results == [Chunk("retrieved text", "guide.pdf", 3)]
    assert store.delete("corpus-1") is True
    assert store.search("corpus-1", [0.1, 0.9], datetime(2029, 1, 1, tzinfo=UTC)) == []
