from testgen_copilot.generator import TestGenerator, GenerationConfig


def test_benchmark_test_generated(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("def f(x):\n    return x * 2\n")
    out = tmp_path / "tests"
    gen = TestGenerator(GenerationConfig(include_benchmarks=True))
    gen.generate_tests(src, out)
    content = (out / "test_mod.py").read_text()
    assert "benchmark(" in content


def test_benchmark_tests_can_be_disabled(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("def f(x):\n    return x * 2\n")
    out = tmp_path / "tests"
    gen = TestGenerator(GenerationConfig(include_benchmarks=False))
    gen.generate_tests(src, out)
    content = (out / "test_mod.py").read_text()
    assert "benchmark(" not in content
