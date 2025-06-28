from testgen_copilot.generator import TestGenerator, GenerationConfig


def test_integration_test_generated(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("""def a():\n    return 1\n\n
def b():\n    return a() + 1\n""")
    out = tmp_path / "tests"
    gen = TestGenerator(GenerationConfig(include_integration_tests=True))
    gen.generate_tests(src, out)
    content = (out / "test_mod.py").read_text()
    assert "_integration" in content


def test_integration_tests_can_be_disabled(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("""def a():\n    return 1\n\n
def b():\n    return a() + 1\n""")
    out = tmp_path / "tests"
    gen = TestGenerator(GenerationConfig(include_integration_tests=False))
    gen.generate_tests(src, out)
    content = (out / "test_mod.py").read_text()
    assert "_integration" not in content
