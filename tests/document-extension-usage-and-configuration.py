import pytest
from testgen_copilot.vscode import write_usage_docs


def test_success(tmp_path):
    dest = tmp_path / "docs"
    doc_file = write_usage_docs(dest)
    assert doc_file.exists()
    text = doc_file.read_text()
    assert "Generate Tests" in text


def test_edge_case(tmp_path):
    invalid = tmp_path / "file"
    invalid.write_text("not a dir")
    with pytest.raises(NotADirectoryError):
        write_usage_docs(invalid)
