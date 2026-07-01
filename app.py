"""CLI for the document-processing agent.

Examples:
    python app.py extract sample_docs/sample_invoice.txt
    python app.py ask     sample_docs/sample_invoice.txt "How much VAT was charged?"
    python app.py rag     sample_docs/sample_invoice.txt "What is the grand total?"
    python app.py agent   sample_docs/sample_invoice.txt "Summarize this invoice."
"""
from __future__ import annotations

import argparse

from doc_agent.agent import run_agent
from doc_agent.extract import extract_invoice
from doc_agent.loader import load_document
from doc_agent.qa import answer_question
from doc_agent.rag import rag_answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intelligent document processing with Claude + LangChain/LangGraph."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract structured invoice fields")
    p_extract.add_argument("file")

    p_ask = sub.add_parser("ask", help="Answer a question about the document (context-stuffing)")
    p_ask.add_argument("file")
    p_ask.add_argument("question")

    p_rag = sub.add_parser("rag", help="Answer a question via retrieval-augmented generation")
    p_rag.add_argument("file")
    p_rag.add_argument("question")

    p_agent = sub.add_parser("agent", help="Let the LangGraph agent decide how to solve a task")
    p_agent.add_argument("file")
    p_agent.add_argument("task")

    for p in (p_extract, p_ask, p_rag, p_agent):
        p.add_argument("--model", default=None, help="Override the Claude model")

    args = parser.parse_args()
    text = load_document(args.file)

    if args.command == "extract":
        print(extract_invoice(text, model=args.model).model_dump_json(indent=2))
    elif args.command == "ask":
        print(answer_question(text, args.question, model=args.model))
    elif args.command == "rag":
        print(rag_answer(text, args.question, model=args.model))
    elif args.command == "agent":
        print(run_agent(text, args.task, model=args.model))


if __name__ == "__main__":
    main()
