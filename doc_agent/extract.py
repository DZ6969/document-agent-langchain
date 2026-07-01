"""Structured extraction: turn free-form document text into an ``Invoice``.

Uses ``ChatAnthropic.with_structured_output`` so Claude returns a validated
Pydantic object instead of a string we have to parse by hand.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .config import get_llm
from .schemas import Invoice

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured data from invoices and receipts. "
            "Only use values present in the document. If a field is missing, "
            "leave it null — never invent data.",
        ),
        ("human", "Extract the invoice fields from this document:\n\n{document}"),
    ]
)


def extract_invoice(document_text: str, model: str | None = None) -> Invoice:
    """Extract an :class:`Invoice` from raw document text."""
    llm = get_llm(model=model)
    chain = _PROMPT | llm.with_structured_output(Invoice)
    return chain.invoke({"document": document_text})
