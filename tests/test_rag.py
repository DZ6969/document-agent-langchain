"""RAG retrieval is tested offline with HashingEmbeddings — no API key needed."""
from doc_agent.embeddings import HashingEmbeddings
from doc_agent.rag import build_retriever, split_text

TEXT = (
    "The cat sat quietly on the warm mat.\n\n"
    "Invoices must include the VAT amount and the grand total.\n\n"
    "Bananas are yellow and grow in tropical climates."
)


def test_split_text_produces_multiple_chunks():
    chunks = split_text(TEXT, chunk_size=60, chunk_overlap=0)
    assert len(chunks) >= 2


def test_retriever_returns_the_relevant_chunk():
    retriever = build_retriever(
        TEXT, embeddings=HashingEmbeddings(), k=1, chunk_size=60, chunk_overlap=0
    )
    docs = retriever.invoke("What is the VAT and the total?")
    assert docs
    assert "VAT" in docs[0].page_content
