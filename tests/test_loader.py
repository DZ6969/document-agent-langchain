import pytest

from doc_agent.loader import load_document


def test_loads_sample_invoice():
    text = load_document("sample_docs/sample_invoice.txt")
    assert "IVA" in text
    assert "15,764.40" in text


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_document("does_not_exist.txt")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(empty)
