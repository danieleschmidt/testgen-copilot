import json
import pytest
from testgen_copilot.vscode import scaffold_extension


def test_success(tmp_path):
    dest = tmp_path / "ext"
    scaffold_extension(dest)
    pkg = json.loads((dest / "package.json").read_text())
    assert pkg["name"] == "testgen-copilot"
    assert (dest / "src" / "extension.js").exists()


def test_edge_case(tmp_path):
    dest = tmp_path / "ext"
    dest.mkdir()
    (dest / "package.json").write_text("{}")
    with pytest.raises(FileExistsError):
        scaffold_extension(dest)
