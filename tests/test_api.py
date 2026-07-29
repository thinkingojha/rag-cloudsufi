import asyncio

import httpx

from document_qa.api import app


class FakeService:
    def ready(self) -> bool:
        return True

    def ask(self, _: str, __: str) -> None:
        return None


def request_api(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        app.state.service = FakeService()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_liveness_endpoint() -> None:
    response = request_api("GET", "/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_question_for_missing_corpus_returns_not_found() -> None:
    response = request_api(
        "POST",
        "/v1/corpora/missing/questions",
        json={"question": "What is this?"},
    )

    assert response.status_code == 404


def test_create_corpus_rejects_more_than_three_files() -> None:
    files = [
        ("files", (f"document-{number}.pdf", b"%PDF", "application/pdf"))
        for number in range(4)
    ]

    response = request_api("POST", "/v1/corpora", files=files)

    assert response.status_code == 422
    assert response.json()["detail"] == "Upload between one and three PDF files."
