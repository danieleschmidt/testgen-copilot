from testgen_copilot.security import SecurityScanner


def _scan(src, code):
    src.write_text(code)
    scanner = SecurityScanner()
    return scanner.scan_file(src)


def test_os_system_constant(tmp_path):
    src = tmp_path / "s.py"
    report = _scan(src, "import os\nos.system('ls')\n")
    assert any('os.system' in i.message for i in report.issues)


def test_shell_injection(tmp_path):
    src = tmp_path / 'inj.py'
    code = "import os\ncmd = 'ls ' + 'foo'\nos.system(cmd)\n"
    report = _scan(src, code)
    assert any('shell injection' in i.message for i in report.issues)


def test_named_temporary_file_delete_false(tmp_path):
    src = tmp_path / 'tmp.py'
    code = "import tempfile\n" "tempfile.NamedTemporaryFile(delete=False)\n"
    report = _scan(src, code)
    assert any('NamedTemporaryFile' in i.message for i in report.issues)


def test_mktemp_usage(tmp_path):
    src = tmp_path / 'mk.py'
    code = "import tempfile\n" "tempfile.mktemp()\n"
    report = _scan(src, code)
    assert any('mktemp' in i.message for i in report.issues)

