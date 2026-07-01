"""Retrieval-Augmented Generation over a document.

Splits the document into chunks, indexes them in an in-memory vector store,
retrieves the passages most relevant to a question, and asks Claude to answer
using only those passages. This scales to documents too large to fit in one
prompt (where the context-stuffing in ``qa.py`` would not).
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_llm
from .embeddings import get_embeddings

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the retrieved context below. "
            "If the context does not contain the answer, say so.\n\n"
            "--- CONTEXT ---\n{context}\n--- END CONTEXT ---",
        ),
        ("human", "{question}"),
    ]
)


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split ``text`` into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def build_retriever(
    text: str,
    embeddings: Embeddings | None = None,
    k: int = 3,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """Build an in-memory retriever over the chunks of ``text``."""
    embeddings = embeddings or get_embeddings()
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = InMemoryVectorStore.from_texts(chunks, embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def rag_answer(
    text: str,
    question: str,
    embeddings: Embeddings | None = None,
    model: str | None = None,
    k: int = 3,
) -> str:
    """Answer ``question`` using retrieval-augmented generation over ``text``."""
    retriever = build_retriever(text, embeddings=embeddings, k=k)
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    chain = _PROMPT | get_llm(model=model) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
