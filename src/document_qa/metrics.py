"""Low-cardinality Prometheus metrics for operating the API."""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "document_qa_http_requests_total",
    "Completed HTTP requests.",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "document_qa_http_request_duration_seconds",
    "HTTP request duration.",
    ("method", "route"),
)
CORPORA_CREATED = Counter("document_qa_corpora_created_total", "Created document corpora.")
CHUNKS_INDEXED = Counter("document_qa_chunks_indexed_total", "Indexed document chunks.")
QUESTIONS_ANSWERED = Counter("document_qa_questions_answered_total", "Answered questions.")
