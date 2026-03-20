import re
from pathlib import Path
from typing import List, Dict


PATTERNS = [
    {
        "id": "SQL_INJECTION",
        "description": "Possible SQL injection via string formatting",
        "pattern": re.compile(
            r'(execute|query|cursor\.execute)\s*\(\s*["\'].*%[sd]|'
            r'f["\'].*SELECT.*\{|'
            r'execute\s*\(\s*f["\']',
            re.IGNORECASE,
        ),
        "severity": "HIGH",
    },
    {
        "id": "HARDCODED_SECRET",
        "description": "Possible hardcoded secret or password",
        "pattern": re.compile(
            r'(password|secret|api_key|apikey|token|passwd)\s*=\s*["\'][^"\']{4,}["\']',
            re.IGNORECASE,
        ),
        "severity": "HIGH",
    },
    {
        "id": "HARDCODED_IP",
        "description": "Hardcoded IP address",
        "pattern": re.compile(r'["\'](\d{1,3}\.){3}\d{1,3}["\']'),
        "severity": "LOW",
    },
    {
        "id": "SHELL_INJECTION",
        "description": "Possible shell injection via os.system or subprocess with shell=True",
        "pattern": re.compile(
            r'os\.system\s*\(|subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
            re.IGNORECASE,
        ),
        "severity": "HIGH",
    },
    {
        "id": "EVAL_USAGE",
        "description": "Use of eval() with potentially untrusted input",
        "pattern": re.compile(r'\beval\s*\('),
        "severity": "MEDIUM",
    },
    {
        "id": "PICKLE_USAGE",
        "description": "Use of pickle with potentially untrusted data",
        "pattern": re.compile(r'pickle\.(load|loads)\s*\('),
        "severity": "MEDIUM",
    },
    {
        "id": "DEBUG_MODE",
        "description": "Debug mode enabled in production code",
        "pattern": re.compile(r'debug\s*=\s*True', re.IGNORECASE),
        "severity": "LOW",
    },
]


def scan_file(filepath: str) -> List[Dict]:
    """Scan a file for security vulnerabilities. Returns list of findings."""
    findings = []
    source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()

    for lineno, line in enumerate(lines, 1):
        for vuln in PATTERNS:
            if vuln["pattern"].search(line):
                findings.append({
                    "file": filepath,
                    "line": lineno,
                    "code": line.strip(),
                    "id": vuln["id"],
                    "description": vuln["description"],
                    "severity": vuln["severity"],
                })

    return findings


def format_report(findings: List[Dict]) -> str:
    """Format findings as a readable report."""
    if not findings:
        return "No vulnerabilities found."

    lines = [f"Found {len(findings)} potential issue(s):\n"]
    for f in findings:
        lines.append(f"[{f['severity']}] {f['id']} — {f['file']}:{f['line']}")
        lines.append(f"  {f['description']}")
        lines.append(f"  Code: {f['code'][:120]}")
        lines.append("")
    return "\n".join(lines)
