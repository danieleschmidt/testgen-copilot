import pytest
import tempfile
import os
from testgen.vulnscan import scan_file, format_report


def write_temp(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_detects_hardcoded_secret():
    code = 'password = "super_secret_123"\n'
    path = write_temp(code)
    try:
        findings = scan_file(path)
        ids = {f["id"] for f in findings}
        assert "HARDCODED_SECRET" in ids
    finally:
        os.unlink(path)


def test_detects_eval():
    code = "result = eval(user_input)\n"
    path = write_temp(code)
    try:
        findings = scan_file(path)
        ids = {f["id"] for f in findings}
        assert "EVAL_USAGE" in ids
    finally:
        os.unlink(path)


def test_clean_file():
    code = "def add(a, b):\n    return a + b\n"
    path = write_temp(code)
    try:
        findings = scan_file(path)
        assert findings == []
    finally:
        os.unlink(path)


def test_format_report_no_findings():
    assert "No vulnerabilities" in format_report([])


def test_format_report_with_findings():
    findings = [{"file": "x.py", "line": 1, "code": "eval(x)", "id": "EVAL_USAGE", "description": "eval", "severity": "MEDIUM"}]
    report = format_report(findings)
    assert "EVAL_USAGE" in report
