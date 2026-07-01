# Intelligent Document Agent — LangChain · LangGraph · Claude

A small but complete **intelligent document-processing agent** built with
[LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
and Anthropic's **Claude** (via `langchain-anthropic`).

Given a document (PDF or text) it can:

- **Extract structured data** — turn an invoice into a validated `Invoice` object
  (issuer, receiver, currency, subtotal, tax, total, line items) using
  `with_structured_output` + Pydantic. No fragile regex.
- **Answer questions** grounded in the document.
- **Act as an agent** — a LangGraph ReAct agent that reasons over the document
  and decides on its own which tool to call (read / extract / answer).

> Es una demostración de procesamiento inteligente de documentos: combina OCR/parseo
> de PDFs (un flujo que ya construyo en producción) con una capa de LLM usando LangChain.

## Why this exists

I build production document-automation pipelines in Python (OCR + PDF parsing of
invoices, bank statements and credit reports). This project layers an LLM on top
of that same problem using LangChain / LangGraph and Claude.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

```bash
# Extract structured invoice fields -> JSON
python app.py extract sample_docs/sample_invoice.txt

# Ask a question about the document
python app.py ask sample_docs/sample_invoice.txt "How much VAT was charged and what is the total?"

# Let the agent decide how to solve the task
python app.py agent sample_docs/sample_invoice.txt "Summarize this invoice and list its line items."
```

## Architecture

```
app.py                 CLI (extract | ask | agent)
doc_agent/
  config.py            ChatAnthropic model (Claude via langchain-anthropic)
  loader.py            PDF / text loading
  schemas.py           Pydantic schemas (Invoice, LineItem)
  extract.py           structured extraction  (with_structured_output)
  qa.py                grounded question answering (context-stuffing)
  agent.py             LangGraph ReAct agent + tools
```

## Tech

Python · LangChain · LangGraph · langchain-anthropic (Claude) · Pydantic · pypdf

## Notes

- The default model is `claude-opus-4-8`; set `MODEL=claude-sonnet-5` (or
  `claude-haiku-4-5`) in `.env` to trade quality for cost.
- Documents are assumed to fit the context window. For large corpora, swap the
  context-stuffing in `qa.py` for retrieval (RAG).
