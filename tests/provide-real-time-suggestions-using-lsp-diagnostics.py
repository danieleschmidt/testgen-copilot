from testgen_copilot.vscode import suggest_from_diagnostics


def test_success():
    diags = [{"message": "unused variable", "severity": 3}]
    assert suggest_from_diagnostics(diags) == ["Warning: unused variable"]


def test_edge_case_null_input():
    assert suggest_from_diagnostics([]) == []
