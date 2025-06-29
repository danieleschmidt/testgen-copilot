import json
from testgen_copilot.vscode import scaffold_extension


def test_commands_added(tmp_path):
    dest = tmp_path / "ext"
    scaffold_extension(dest)
    pkg = json.loads((dest / "package.json").read_text())
    commands = [c["command"] for c in pkg["contributes"]["commands"]]
    assert "testgen.runSecurityScan" in commands
    assert "testgen.showCoverage" in commands
    text = (dest / "src" / "extension.js").read_text()
    assert "runSecurityScan" in text
    assert "showCoverage" in text
