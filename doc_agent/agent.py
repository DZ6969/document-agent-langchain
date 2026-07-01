"""A LangGraph ReAct agent that reasons over a loaded document.

The agent is given tools and decides for itself when to read the raw text,
extract structured invoice fields, or answer a question — a small but real
example of an LLM-driven agent (LangChain tools + LangGraph orchestration).
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool

from .config import get_llm
from .extract import extract_invoice
from .qa import answer_question

_SYSTEM = (
    "You are a document-processing assistant. A document has been loaded. "
    "Use your tools to read it, extract structured fields, or answer questions. "
    "Prefer extract_invoice_fields for anything about amounts, taxes, totals or line items."
)


def build_agent(document_text: str, model: str | None = None):
    """Build a ReAct agent whose tools operate on ``document_text``."""
    llm = get_llm(model=model)

    @tool
    def read_document() -> str:
        """Return the full raw text of the loaded document."""
        return document_text

    @tool
    def extract_invoice_fields() -> str:
        """Extract structured invoice fields (issuer, totals, taxes, line items) as JSON."""
        return extract_invoice(document_text, model=model).model_dump_json(indent=2)

    @tool
    def answer_from_document(question: str) -> str:
        """Answer a natural-language question grounded in the loaded document."""
        return answer_question(document_text, question, model=model)

    tools = [read_document, extract_invoice_fields, answer_from_document]
    return create_agent(llm, tools, system_prompt=_SYSTEM)


def run_agent(document_text: str, task: str, model: str | None = None) -> str:
    """Run the agent on a natural-language ``task`` and return its final answer."""
    agent = build_agent(document_text, model=model)
    result = agent.invoke({"messages": [("user", task)]})
    return result["messages"][-1].content
