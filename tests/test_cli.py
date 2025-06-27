import pytest

from testgen_copilot.cli import main


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
    main(["--project", str(tmp_path), "--coverage-target", "100"])
    out = capsys.readouterr().out
    assert "Coverage target satisfied" in out


def test_cli_coverage_only_fail(tmp_path):
    _make_project(tmp_path, covered=False)
    with pytest.raises(SystemExit) as exc:
        main(["--project", str(tmp_path), "--coverage-target", "100"])
    assert exc.value.code == 1


def test_cli_show_missing(tmp_path, capsys):
    _make_project(tmp_path, covered=False)
    with pytest.raises(SystemExit):
        main([
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
            "--project",
            str(tmp_path),
            "--quality-target",
            "100",
        ])
    assert exc.value.code == 1
