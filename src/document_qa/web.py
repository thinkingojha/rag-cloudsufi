"""Thin Streamlit client for the document Q&A API."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Document Q&A")
st.caption("Upload up to three PDFs and ask questions with page citations.")


def api_error(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def api_headers() -> dict[str, str]:
    return {"X-API-Key": API_AUTH_TOKEN} if API_AUTH_TOKEN else {}


if "corpus_id" not in st.session_state:
    st.session_state.corpus_id = None
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Choose 1–3 PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index documents", disabled=not uploads):
        if not 1 <= len(uploads) <= 3:
            st.error("Choose between one and three PDFs.")
        else:
            files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploads]
            try:
                with st.spinner("Extracting text and creating embeddings…"):
                    response = httpx.post(
                        f"{API_BASE_URL}/v1/corpora",
                        files=files,
                        headers=api_headers(),
                        timeout=90.0,
                    )
                if response.is_success:
                    data = response.json()
                    st.session_state.corpus_id = data["corpus_id"]
                    st.session_state.messages = []
                    st.success(f"Ready: {data['chunk_count']} chunks indexed.")
                else:
                    st.error(api_error(response))
            except httpx.HTTPError:
                st.error("The API service is unavailable. Check that it is running.")
    if st.session_state.corpus_id and st.button("Clear session"):
        try:
            httpx.delete(
                f"{API_BASE_URL}/v1/corpora/{st.session_state.corpus_id}",
                headers=api_headers(),
                timeout=10.0,
            )
        except httpx.HTTPError:
            pass
        st.session_state.corpus_id = None
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Sources used"):
                for citation in message["citations"]:
                    st.markdown(f"**{citation['document_name']}, p. {citation['page_number']}**")
                    st.caption(citation["excerpt"])

question = st.chat_input("Ask about the indexed PDFs", disabled=not st.session_state.corpus_id)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant pages…"):
                response = httpx.post(
                    f"{API_BASE_URL}/v1/corpora/{st.session_state.corpus_id}/questions",
                    json={"question": question},
                    headers=api_headers(),
                    timeout=90.0,
                )
            if not response.is_success:
                st.error(api_error(response))
            else:
                data = response.json()
                st.markdown(data["answer"])
                with st.expander("Sources used"):
                    for citation in data["citations"]:
                        st.markdown(
                            f"**{citation['document_name']}, p. {citation['page_number']}**"
                        )
                        st.caption(citation["excerpt"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "citations": data["citations"]}
                )
        except httpx.HTTPError:
            st.error("The API service is unavailable. Check that it is running.")
