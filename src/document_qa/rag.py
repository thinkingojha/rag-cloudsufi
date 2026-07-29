"""The deliberately small RAG implementation behind the HTTP service."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from openai import OpenAI
from pypdf import PdfReader

CHUNK_SIZE = 1_100
CHUNK_OVERLAP = 180
TOP_K = 5
CITATION_PATTERN = re.compile(r"\[[^\]\n]+,\s*p\.\s*\d+\]")
HTML_TAG_PATTERN = re.compile(r"</?[^>\n]*>")


@dataclass(frozen=True)
class Chunk:
    text: str
    document_name: str
    page_number: int

    @property
    def citation(self) -> str:
        return f"{self.document_name}, p. {self.page_number}"


def extract_pdf_chunks(file_name: str, payload: bytes) -> list[Chunk]:
    """Extract text on a page boundary so every response can be cited."""
    try:
        reader = PdfReader(BytesIO(payload))
    except Exception as exc:  # parser exceptions vary by pypdf release
        raise ValueError(f"'{file_name}' could not be read as a PDF.") from exc

    chunks: list[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = (page.extract_text() or "").strip()
        for text in split_text(extracted_text):
            chunks.append(Chunk(text=text, document_name=file_name, page_number=page_number))
    return chunks


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on nearby whitespace while keeping enough overlap for context."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def corpus_fingerprint(documents: list[tuple[str, bytes]]) -> str:
    """Fingerprint both file name and content to make upload retries idempotent."""
    digest = sha256()
    for name, payload in sorted(documents, key=lambda document: document[0]):
        digest.update(name.encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()


def embed_texts(client: OpenAI, texts: Iterable[str], model: str) -> list[list[float]]:
    """Embed in batches to leave headroom for provider request limits."""
    values = list(texts)
    vectors: list[list[float]] = []
    for start in range(0, len(values), 100):
        response = client.embeddings.create(model=model, input=values[start : start + 100])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def group_chunks_by_source(chunks: list[Chunk]) -> list[tuple[Chunk, list[str]]]:
    """Group multiple retrieved chunks from one page under a single trusted citation."""
    grouped: dict[tuple[str, int], tuple[Chunk, list[str]]] = {}
    for chunk in chunks:
        key = (chunk.document_name, chunk.page_number)
        if key not in grouped:
            grouped[key] = (chunk, [])
        grouped[key][1].append(chunk.text)
    return list(grouped.values())


def normalize_citations(answer: str, chunks: list[Chunk]) -> str:
    """Keep only page labels that correspond to a retrieved source.

    The model writes prose, but source identifiers remain service-controlled metadata.
    """
    labels = {f"[{source.citation}]" for source, _ in group_chunks_by_source(chunks)}

    def retain_known_label(match: re.Match[str]) -> str:
        return match.group(0) if match.group(0) in labels else ""

    answer_without_unknown_citations = CITATION_PATTERN.sub(retain_known_label, answer)
    return HTML_TAG_PATTERN.sub("", answer_without_unknown_citations).strip()


def generate_answer(client: OpenAI, question: str, chunks: list[Chunk], chat_model: str) -> str:
    """Generate an answer restricted to the retrieved chunks."""
    context = "\n\n".join(
        f"[{source.citation}]\n" + "\n\n".join(excerpts)
        for source, excerpts in group_chunks_by_source(chunks)
    )
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied PDF excerpts. "
                    "If they do not answer the question, say so plainly. "
                    "Use clean Markdown only; never emit HTML or XML tags. "
                    "Keep the answer concise and do not add a generic concluding summary after "
                    "a list of supported facts. Cite each factual bullet or paragraph with the "
                    "single most relevant exact document-and-page label supplied in the excerpts. "
                    "Use more than one citation only when a claim genuinely depends on "
                    "multiple sources."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ],
    )
    answer = response.choices[0].message.content or "I could not generate an answer."
    return normalize_citations(answer, chunks)
