from document_qa.rag import (
    Chunk,
    corpus_fingerprint,
    group_chunks_by_source,
    normalize_citations,
    split_text,
)


def test_split_text_creates_overlapping_chunks() -> None:
    text = "word " * 500

    chunks = split_text(text, size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunks)
    assert chunks[0].split()[-1] in chunks[1]


def test_split_text_ignores_whitespace() -> None:
    assert split_text("   \n  ") == []


def test_corpus_fingerprint_is_order_independent() -> None:
    documents = [("a.pdf", b"one"), ("b.pdf", b"two")]

    assert corpus_fingerprint(documents) == corpus_fingerprint(list(reversed(documents)))


def test_group_chunks_by_source_keeps_all_excerpts_under_one_page_citation() -> None:
    chunks = [
        Chunk("first passage", "guide.pdf", 1),
        Chunk("second passage", "guide.pdf", 1),
        Chunk("other page", "guide.pdf", 2),
    ]

    grouped = group_chunks_by_source(chunks)

    assert [(source.citation, excerpts) for source, excerpts in grouped] == [
        ("guide.pdf, p. 1", ["first passage", "second passage"]),
        ("guide.pdf, p. 2", ["other page"]),
    ]


def test_normalize_citations_removes_unretrieved_page_labels() -> None:
    chunks = [Chunk("passage", "guide.pdf", 1)]

    answer = normalize_citations(
        "Supported [guide.pdf, p. 1]. Unsupported [other.pdf, p. 9].",
        chunks,
    )

    assert answer == "Supported [guide.pdf, p. 1]. Unsupported ."


def test_normalize_citations_removes_stray_html_markup() -> None:
    answer = normalize_citations("A clean answer.</>", [Chunk("passage", "guide.pdf", 1)])

    assert answer == "A clean answer."
