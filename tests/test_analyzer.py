import pytest
import tempfile
import os
import textwrap
from testgen.analyzer import extract_functions, extract_tested_functions, find_untested


SAMPLE = textwrap.dedent('''
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def greet(name: str) -> str:
        return f"Hello {name}"

    def _private(x):
        pass

    class MyClass:
        def method(self, value: int):
            pass
        def __init__(self):
            pass
''')


def write_temp(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_extract_functions_finds_all():
    path = write_temp(SAMPLE)
    try:
        funcs = extract_functions(open(path).read(), path)
        names = {f["name"] for f in funcs}
        assert "add" in names
        assert "greet" in names
    finally:
        os.unlink(path)


def test_extract_functions_skips_dunder():
    path = write_temp(SAMPLE)
    try:
        funcs = extract_functions(open(path).read(), path)
        names = {f["name"] for f in funcs}
        assert "__init__" not in names
    finally:
        os.unlink(path)


def test_extract_functions_annotations():
    path = write_temp(SAMPLE)
    try:
        funcs = extract_functions(open(path).read(), path)
        add = next(f for f in funcs if f["name"] == "add")
        assert add["annotations"].get("a") == "int"
        assert add["annotations"].get("b") == "int"
    finally:
        os.unlink(path)


def test_extract_tested_functions():
    test_code = textwrap.dedent('''
        def test_add():
            assert add(1, 2) == 3
        def test_greet():
            assert greet("world") == "Hello world"
    ''')
    tested = extract_tested_functions(test_code)
    assert "add" in tested
    assert "greet" in tested


def test_find_untested_all_untested():
    path = write_temp(SAMPLE)
    try:
        untested = find_untested(path)
        names = {f["name"] for f in untested}
        assert "add" in names
        assert "greet" in names
    finally:
        os.unlink(path)


def test_find_untested_with_test_file():
    path = write_temp(SAMPLE)
    test_code = "def test_add():\n    assert add(1,2)==3\n"
    tpath = write_temp(test_code)
    try:
        untested = find_untested(path, tpath)
        names = {f["name"] for f in untested}
        assert "add" not in names
        assert "greet" in names
    finally:
        os.unlink(path)
        os.unlink(tpath)
