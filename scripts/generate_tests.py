import json
import pathlib

criteria_path = pathlib.Path("tests/sprint_acceptance_criteria.json")
criteria = json.loads(criteria_path.read_text())

for slug, info in criteria.items():
    test_path = pathlib.Path(info["test_file"])
    test_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "import pytest",
        "from testgen_copilot import identity",
        "",
        "",
    ]
    lines.append("def test_success():")
    lines.append('    assert identity("sample") == "sample"')
    lines.append("")
    lines.append("def test_edge_case_null_input():")
    lines.append("    assert identity(None) is None")
    lines.append("")
    test_path.write_text("\n".join(lines))
    print(f"Generated {test_path}")
