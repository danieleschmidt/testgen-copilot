from testgen_copilot import identity


def test_success():
    assert identity("sample") == "sample"


def test_edge_case_null_input():
    assert identity(None) is None
