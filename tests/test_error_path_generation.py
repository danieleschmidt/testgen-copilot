from testgen_copilot.generator import TestGenerator, GenerationConfig


def test_error_path_test_generated(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("""def f(x):\n    if x < 0:\n        raise ValueError('bad')\n    return x\n""")
    out = tmp_path / "tests"
    gen = TestGenerator(GenerationConfig(include_error_paths=True))
    gen.generate_tests(src, out)
    content = (out / "test_mod.py").read_text()
    assert "pytest.raises(ValueError)" in content


def test_error_path_tests_can_be_disabled(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("""def f(x):\n    if x < 0:\n        raise ValueError('bad')\n    return x\n""")
    out = tmp_path / "tests"
    gen = TestGenerator(GenerationConfig(include_error_paths=False))
    gen.generate_tests(src, out)
    content = (out / "test_mod.py").read_text()
    assert "pytest.raises" not in content
