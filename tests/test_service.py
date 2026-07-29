from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from document_qa.rag import Chunk
from document_qa.service import DocumentQAService
from document_qa.settings import Settings
from document_qa.vector_store import CorpusMetadata


class FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = SimpleNamespace(create=self.embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

    @staticmethod
    def embed(*, input: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(index), 1.0]) for index, _ in enumerate(input)]
        )

    @staticmethod
    def complete(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Grounded answer [guide.pdf, p. 2]")
                )
            ]
        )


class FakeStore:
    def __init__(self) -> None:
        self.metadata: CorpusMetadata | None = None
        self.indexed: tuple[CorpusMetadata, list[Chunk], list[list[float]]] | None = None
        self.search_results = [Chunk("relevant text", "guide.pdf", 2)]

    def purge_expired(self, _: datetime) -> None:
        return None

    def find_by_fingerprint(self, _: str, __: datetime) -> CorpusMetadata | None:
        return self.metadata

    def upsert(
        self,
        metadata: CorpusMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        self.indexed = (metadata, chunks, embeddings)

    def search(self, _: str, __: list[float], ___: datetime) -> list[Chunk]:
        return self.search_results

    def delete(self, _: str) -> bool:
        return True


def test_create_corpus_indexes_extracted_chunks(monkeypatch: object) -> None:
    store = FakeStore()
    service = DocumentQAService(Settings(), store, FakeOpenAI())  # type: ignore[arg-type]
    monkeypatch.setattr(  # type: ignore[union-attr]
        "document_qa.service.extract_pdf_chunks",
        lambda name, _: [Chunk("extracted text", name, 1)],
    )

    metadata, reused = service.create_corpus([("guide.pdf", b"%PDF-test")])

    assert reused is False
    assert metadata.chunk_count == 1
    assert store.indexed is not None
    assert store.indexed[1][0].citation == "guide.pdf, p. 1"


def test_create_corpus_reuses_active_fingerprint(monkeypatch: object) -> None:
    existing = CorpusMetadata(
        corpus_id="existing",
        document_names=["guide.pdf"],
        chunk_count=2,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        fingerprint="fingerprint",
    )
    store = FakeStore()
    store.metadata = existing
    service = DocumentQAService(Settings(), store, FakeOpenAI())  # type: ignore[arg-type]
    monkeypatch.setattr(  # type: ignore[union-attr]
        "document_qa.service.extract_pdf_chunks",
        lambda *_: (_ for _ in ()).throw(AssertionError("existing corpus must skip extraction")),
    )

    metadata, reused = service.create_corpus([("guide.pdf", b"%PDF-test")])

    assert reused is True
    assert metadata == existing
    assert store.indexed is None


def test_ask_returns_answer_with_trusted_citation_metadata() -> None:
    service = DocumentQAService(Settings(), FakeStore(), FakeOpenAI())  # type: ignore[arg-type]

    answer = service.ask("corpus-id", "What does the guide say?")

    assert answer is not None
    assert answer.answer.startswith("Grounded answer")
    assert "</>" not in answer.answer
    assert answer.citations[0].document_name == "guide.pdf"
    assert answer.citations[0].page_number == 2


def test_ask_groups_multiple_chunks_from_the_same_page_into_one_citation() -> None:
    store = FakeStore()
    store.search_results = [
        Chunk("first relevant passage", "guide.pdf", 2),
        Chunk("second relevant passage", "guide.pdf", 2),
    ]
    service = DocumentQAService(Settings(), store, FakeOpenAI())  # type: ignore[arg-type]

    answer = service.ask("corpus-id", "What does the guide say?")

    assert answer is not None
    assert len(answer.citations) == 1
    assert "first relevant passage" in answer.citations[0].excerpt
    assert "second relevant passage" in answer.citations[0].excerpt
