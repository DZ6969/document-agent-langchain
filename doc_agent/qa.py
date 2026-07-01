"""Question answering over a single document (context-stuffing).

For documents that fit comfortably in the context window (invoices, bank
statements, contracts) the simplest robust approach is to pass the whole text
to the model. Larger corpora would swap this for retrieval (RAG).
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import get_llm

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer questions using only the document below. "
            "If the answer is not in the document, say so plainly.\n\n"
            "--- DOCUMENT ---\n{document}\n--- END DOCUMENT ---",
        ),
        ("human", "{question}"),
    ]
)


def answer_question(document_text: str, question: str, model: str | None = None) -> str:
    """Answer a natural-language ``question`` grounded in ``document_text``."""
    llm = get_llm(model=model)
    chain = _PROMPT | llm | StrOutputParser()
    return chain.invoke({"document": document_text, "question": question})
