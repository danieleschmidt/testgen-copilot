from testgen_copilot.coverage import CoverageAnalyzer


def test_basic_coverage(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass

def b():
    pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
from sample import a

def test_a():
    a()
"""
    )

    analyzer = CoverageAnalyzer()
    percent = analyzer.analyze(src, tests_dir)
    assert percent == 50.0


def test_no_functions(tmp_path):
    src = tmp_path / "empty.py"
    src.write_text("pass\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tests_dir) == 100.0


def test_names_in_comments_not_counted(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass

def b():
    pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
from sample import a

def test_a():  # b should not be counted
    a()
"""
    )

    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tests_dir) == 50.0


def test_attribute_call_is_detected(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass

def b():
    pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
import sample

def test_a():
    sample.a()
"""
    )

    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tests_dir) == 50.0


def test_nested_test_dirs(tmp_path):
    """Files in subdirectories should be considered."""
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass
"""
    )

    sub = tmp_path / "tests" / "unit"
    sub.mkdir(parents=True)
    (sub / "test_sample.py").write_text(
        """
from sample import a

def test_a():
    a()
"""
    )

    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tmp_path / "tests") == 100.0


def test_class_methods_counted(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
class C:
    def a(self):
        pass

    def b(self):
        pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
from sample import C

def test_a():
    C().a()
"""
    )

    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tests_dir) == 50.0


def test_alias_imports_are_detected(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass

def b():
    pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
from sample import a as alias_a

def test_a():
    alias_a()
"""
    )

    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tests_dir) == 50.0


def test_module_aliases_are_detected(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass

def b():
    pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
import sample as alias_mod

def test_a():
    alias_mod.a()
"""
    )

    analyzer = CoverageAnalyzer()
    assert analyzer.analyze(src, tests_dir) == 50.0


def test_uncovered_functions_report(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        """
def a():
    pass

def b():
    pass
"""
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        """
from sample import a

def test_a():
    a()
"""
    )

    analyzer = CoverageAnalyzer()
    missing = analyzer.uncovered_functions(src, tests_dir)
    assert missing == {"b"}
