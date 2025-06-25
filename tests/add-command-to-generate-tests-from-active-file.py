import json
import pytest
from testgen_copilot.vscode import scaffold_extension


def test_success(tmp_path):
    dest = tmp_path / "ext"
    scaffold_extension(dest)
    pkg = json.loads((dest / "package.json").read_text())
    commands = [c["command"] for c in pkg["contributes"]["commands"]]
    assert "testgen.generateTestsFromActiveFile" in commands
    text = (dest / "src" / "extension.js").read_text()
    assert "generateTestsFromActiveFile" in text


def test_edge_case(tmp_path):
    dest = tmp_path / "ext"
    scaffold_extension(dest)
    with pytest.raises(FileExistsError):
        scaffold_extension(dest)
