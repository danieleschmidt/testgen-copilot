import json
import pytest
from pathlib import Path

from testgen_copilot.generator import TestGenerator as TG

from testgen_copilot.cli import _language_pattern, main


def test_language_pattern_defaults_to_extension():
    assert _language_pattern("python") == "*.py"
    assert _language_pattern("Py") == "*.py"
    assert _language_pattern("csharp") == "*.cs"
    assert _language_pattern("unknown") == "*unknown"


def _make_project(
    tmp_path, *, dir_name: str = "tests", covered: bool = True, quality: bool = True
):
    src = tmp_path / "sample.py"
    src.write_text("""def a():\n    pass\n\n""")
    tests_dir = tmp_path / dir_name
    tests_dir.mkdir()
    if covered:
        if quality:
            (tests_dir / "test_sample.py").write_text(
                """from sample import a

def test_a():
    assert a() is None
"""
            )
        else:
            (tests_dir / "test_sample.py").write_text(
                """from sample import a

def test_a():
    a()
"""
            )
    else:
        if quality:
            (tests_dir / "test_sample.py").write_text(
                """def test_dummy():
    assert True
"""
            )
        else:
            (tests_dir / "test_sample.py").write_text(
                """def test_dummy():
    pass
"""
            )
    return src


def test_cli_coverage_only_pass(tmp_path, capsys):
    _make_project(tmp_path, covered=True)
    main(["analyze", "--project", str(tmp_path), "--coverage-target", "100"])
    out = capsys.readouterr().out
    assert "Coverage target satisfied" in out


def test_cli_coverage_only_fail(tmp_path):
    _make_project(tmp_path, covered=False)
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--project", str(tmp_path), "--coverage-target", "100"])
    assert exc.value.code == 1


def test_cli_show_missing(tmp_path, capsys):
    _make_project(tmp_path, covered=False)
    with pytest.raises(SystemExit):
        main([
            "analyze",
            "--project",
            str(tmp_path),
            "--coverage-target",
            "100",
            "--show-missing",
        ])
    out = capsys.readouterr().out
    assert "Missing: a" in out


def test_cli_custom_tests_dir(tmp_path, capsys):
    _make_project(tmp_path, dir_name="unit", covered=True)
    main([
        "analyze",
        "--project",
        str(tmp_path),
        "--coverage-target",
        "100",
        "--tests-dir",
        "unit",
    ])
    out = capsys.readouterr().out
    assert "Coverage target satisfied" in out


def test_cli_quality_only_pass(tmp_path, capsys):
    _make_project(tmp_path, quality=True)
    main([
        "analyze",
        "--project",
        str(tmp_path),
        "--quality-target",
        "100",
    ])
    out = capsys.readouterr().out
    assert "Quality target satisfied" in out


def test_cli_quality_only_fail(tmp_path):
    _make_project(tmp_path, quality=False)
    with pytest.raises(SystemExit) as exc:
        main([
            "analyze",
            "--project",
            str(tmp_path),
            "--quality-target",
            "100",
        ])
    assert exc.value.code == 1


def test_cli_batch_generation(tmp_path):
    src1 = tmp_path / "a.py"
    src1.write_text("def foo():\n    pass\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    src2 = sub / "b.py"
    src2.write_text("def bar():\n    pass\n")
    out = tmp_path / "tests"
    main(["generate", "--project", str(tmp_path), "--output", str(out), "--batch"])
    assert (out / "test_a.py").exists()
    assert (out / "test_b.py").exists()


def test_cli_batch_requires_project(tmp_path):
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--output", str(out), "--batch"])


def test_cli_batch_requires_output(tmp_path):
    with pytest.raises(SystemExit):
        main(["generate", "--project", str(tmp_path), "--batch"])


def test_cli_batch_no_file_allowed(tmp_path):
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main([
            "generate",
            "--project",
            str(tmp_path),
            "--output",
            str(out),
            "--batch",
            "--file",
            str(tmp_path / "a.py"),
        ])


def test_cli_watch_auto_generation(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    file = src / "a.py"
    file.write_text("def foo():\n    pass\n")
    out = tmp_path / "tests"

    called = {}

    def fake_watch(directory, generator, out_dir, pattern, poll=1.0, *, auto_generate=False, max_cycles=None):
        called["auto"] = auto_generate
        called["poll"] = poll
        if auto_generate:
            generator.generate_tests(file, out_dir)

    monkeypatch.setattr("testgen_copilot.cli._watch_for_changes", fake_watch)
    main(["generate", "--watch", str(src), "--output", str(out), "--auto-generate", "--poll", "2.0"])
    assert called.get("auto") is True
    assert called.get("poll") == 2.0
    assert (out / "test_a.py").exists()


def test_cli_watch_change_report_only(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "tests"

    called = {}

    def fake_watch(directory, generator, out_dir, pattern, poll=1.0, *, auto_generate=False, max_cycles=None):
        called["auto"] = auto_generate
        called["poll"] = poll

    monkeypatch.setattr("testgen_copilot.cli._watch_for_changes", fake_watch)
    main(["generate", "--watch", str(src), "--output", str(out)])
    assert called.get("auto") is False
    assert called.get("poll") == 1.0
    assert not list(out.glob("test_*.py"))


def test_cli_watch_requires_output(tmp_path):
    with pytest.raises(SystemExit):
        main(["generate", "--watch", str(tmp_path)])


def test_cli_watch_no_file_allowed(tmp_path):
    file = tmp_path / "a.py"
    file.write_text("""
def foo():
    pass
""")
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main([
            "generate",
            "--watch",
            str(tmp_path),
            "--output",
            str(out),
            "--file",
            str(file),
        ])


def test_cli_no_edge_cases_flag(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    out = tmp_path / "tests"

    config_values = {}

    def fake_generate(self, file_path, output_dir):
        config_values["include_edge_cases"] = self.config.include_edge_cases
        return Path(output_dir) / "test_a.py"

    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out), "--no-edge-cases"])
    assert config_values["include_edge_cases"] is False


def test_cli_no_error_tests_flag(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo(x):\n    if x < 0:\n        raise ValueError\n")
    out = tmp_path / "tests"

    config_values = {}

    def fake_generate(self, file_path, output_dir):
        config_values["include_error_paths"] = self.config.include_error_paths
        return Path(output_dir) / "test_a.py"

    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out), "--no-error-tests"])
    assert config_values["include_error_paths"] is False


def test_cli_no_benchmark_tests_flag(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    return 1\n")
    out = tmp_path / "tests"

    config_values = {}

    def fake_generate(self, file_path, output_dir):
        config_values["include_benchmarks"] = self.config.include_benchmarks
        return Path(output_dir) / "test_a.py"

    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out), "--no-benchmark-tests"])
    assert config_values["include_benchmarks"] is False


def test_cli_no_integration_tests_flag(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    return 1\n")
    out = tmp_path / "tests"

    config_values = {}

    def fake_generate(self, file_path, output_dir):
        config_values["include_integration_tests"] = self.config.include_integration_tests
        return Path(output_dir) / "test_a.py"

    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out), "--no-integration-tests"])
    assert config_values["include_integration_tests"] is False


def test_cli_config_file(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    cfg = {"include_edge_cases": False}
    cfg_path = tmp_path / ".testgen.config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "tests"

    config_values = {}

    def fake_generate(self, file_path, output_dir):
        config_values["include_edge_cases"] = self.config.include_edge_cases
        return Path(output_dir) / "test_a.py"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out)])
    assert config_values["include_edge_cases"] is False


def test_cli_flag_overrides_config(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    cfg = {"include_edge_cases": True}
    cfg_path = tmp_path / ".testgen.config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "tests"

    config_values = {}

    def fake_generate(self, file_path, output_dir):
        config_values["include_edge_cases"] = self.config.include_edge_cases
        return Path(output_dir) / "test_a.py"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out), "--no-edge-cases"])
    assert config_values["include_edge_cases"] is False


def test_cli_scaffold_command(tmp_path):
    dest = tmp_path / "ext"
    main(["scaffold", str(dest)])
    assert (dest / "package.json").exists()


def test_cli_invalid_file_path(tmp_path):
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--file", str(tmp_path / "nofile.py"), "--output", str(out)])


def test_cli_invalid_project_path(tmp_path):
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--project", str(tmp_path / "missing"), "--output", str(out), "--batch"])


def test_cli_invalid_watch_path(tmp_path):
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--watch", str(tmp_path / "missing"), "--output", str(out)])


def test_cli_invalid_config_schema(tmp_path):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    cfg = {"include_edge_cases": "yes"}
    cfg_path = tmp_path / ".testgen.config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--file", str(src), "--output", str(out), "--config", str(cfg_path)])


def test_cli_invalid_config_option(tmp_path):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    cfg = {"unknown": True}
    cfg_path = tmp_path / ".testgen.config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--file", str(src), "--output", str(out), "--config", str(cfg_path)])


def test_cli_invalid_config_json(tmp_path):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    cfg_path = tmp_path / ".testgen.config.json"
    cfg_path.write_text("{invalid")
    out = tmp_path / "tests"
    with pytest.raises(SystemExit):
        main(["generate", "--file", str(src), "--output", str(out), "--config", str(cfg_path)])


def test_cli_creates_output_directory(tmp_path, monkeypatch):
    src = tmp_path / "a.py"
    src.write_text("def foo():\n    pass\n")
    out = tmp_path / "tests"

    def fake_generate(self, file_path, output_dir):
        return Path(output_dir) / "test_a.py"

    monkeypatch.setattr(TG, "generate_tests", fake_generate)
    main(["generate", "--file", str(src), "--output", str(out)])
    assert out.is_dir()

