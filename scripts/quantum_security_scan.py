#!/usr/bin/env python3
"""Comprehensive security scanning for quantum-inspired task planner."""

import os
import sys
import json
import re
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Add src to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))


@dataclass
class SecurityIssue:
    """Security issue found during scanning."""
    
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str  # "vulnerability", "hardcoding", "injection", "crypto", etc.
    file_path: str
    line_number: int
    description: str
    recommendation: str
    cwe_id: Optional[str] = None
    confidence: float = 1.0  # 0.0 - 1.0


@dataclass 
class SecurityScanResult:
    """Results of security scanning."""
    
    scan_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_files_scanned: int = 0
    total_lines_scanned: int = 0
    issues: List[SecurityIssue] = field(default_factory=list)
    security_score: float = 0.0  # 0.0 - 10.0, higher is better
    scan_duration_seconds: float = 0.0
    
    def get_issues_by_severity(self) -> Dict[str, List[SecurityIssue]]:
        """Group issues by severity level."""
        grouped = {}
        for issue in self.issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)
        return grouped
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for issue in self.issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        return counts


class QuantumSecurityScanner:
    """Comprehensive security scanner for quantum codebase."""
    
    def __init__(self, root_path: str):
        """Initialize security scanner."""
        self.root_path = Path(root_path)
        self.scan_patterns = self._initialize_patterns()
        self.excluded_paths = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".mypy_cache"
        }
    
    def _initialize_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize security scanning patterns."""
        
        return {
            "hardcoded_secrets": [
                {
                    "pattern": r"(?i)(password|pwd|passwd|secret|key|token|api_key|apikey)\s*=\s*['\"][^'\"]{8,}['\"]",
                    "severity": "high",
                    "description": "Hardcoded password or secret detected",
                    "cwe": "CWE-798"
                },
                {
                    "pattern": r"(?i)(aws_access_key_id|aws_secret_access_key)\s*=\s*['\"][^'\"]+['\"]",
                    "severity": "critical",
                    "description": "Hardcoded AWS credentials detected",
                    "cwe": "CWE-798"
                },
                {
                    "pattern": r"['\"](?:AKIA[0-9A-Z]{16}|[A-Za-z0-9/+=]{40})['\"]",
                    "severity": "critical", 
                    "description": "AWS access key or secret key pattern detected",
                    "cwe": "CWE-798"
                }
            ],
            
            "sql_injection": [
                {
                    "pattern": r"(?i)execute\s*\(\s*['\"].*%s.*['\"].*%\s*\(",
                    "severity": "high",
                    "description": "Potential SQL injection via string formatting",
                    "cwe": "CWE-89"
                },
                {
                    "pattern": r"(?i)cursor\.execute\s*\(\s*f['\"].*\{.*\}.*['\"]",
                    "severity": "high",
                    "description": "Potential SQL injection via f-string",
                    "cwe": "CWE-89"
                },
                {
                    "pattern": r"(?i)SELECT.*FROM.*WHERE.*\+.*['\"]",
                    "severity": "medium",
                    "description": "Potential SQL injection via string concatenation",
                    "cwe": "CWE-89"
                }
            ],
            
            "command_injection": [
                {
                    "pattern": r"(?i)subprocess\.(call|run|Popen)\([^)]*shell\s*=\s*True",
                    "severity": "high",
                    "description": "Command injection risk with shell=True",
                    "cwe": "CWE-78"
                },
                {
                    "pattern": r"(?i)os\.system\s*\(\s*[^)]*\+",
                    "severity": "high",
                    "description": "Command injection via os.system with concatenation",
                    "cwe": "CWE-78"
                },
                {
                    "pattern": r"(?i)eval\s*\(\s*[^)]*input",
                    "severity": "critical",
                    "description": "Code execution via eval with user input",
                    "cwe": "CWE-95"
                }
            ],
            
            "cryptographic_issues": [
                {
                    "pattern": r"(?i)hashlib\.(md5|sha1)\(",
                    "severity": "medium",
                    "description": "Use of weak cryptographic hash function",
                    "cwe": "CWE-327"
                },
                {
                    "pattern": r"(?i)random\.random\(\).*password|password.*random\.random\(\)",
                    "severity": "high",
                    "description": "Weak random number generation for security purposes",
                    "cwe": "CWE-330"
                },
                {
                    "pattern": r"ssl_verify\s*=\s*False|verify\s*=\s*False.*requests\.",
                    "severity": "medium",
                    "description": "SSL/TLS certificate verification disabled",
                    "cwe": "CWE-295"
                }
            ],
            
            "input_validation": [
                {
                    "pattern": r"(?i)pickle\.loads?\s*\(",
                    "severity": "high",
                    "description": "Unsafe deserialization with pickle",
                    "cwe": "CWE-502"
                },
                {
                    "pattern": r"(?i)yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader",
                    "severity": "medium",
                    "description": "Unsafe YAML loading",
                    "cwe": "CWE-502"
                },
                {
                    "pattern": r"(?i)open\s*\([^)]*\+.*['\"]w",
                    "severity": "low", 
                    "description": "File opened in write mode - verify path validation",
                    "cwe": "CWE-22"
                }
            ],
            
            "quantum_specific": [
                {
                    "pattern": r"(?i)quantum.*key.*=.*['\"][^'\"]+['\"]",
                    "severity": "medium",
                    "description": "Potential hardcoded quantum key or parameter",
                    "cwe": "CWE-798"
                },
                {
                    "pattern": r"(?i)coherence.*time.*=.*\d+\.?\d*$",
                    "severity": "info",
                    "description": "Hardcoded quantum coherence time - consider making configurable",
                    "cwe": None
                },
                {
                    "pattern": r"(?i)entanglement.*strength.*=.*\d+\.?\d*$",
                    "severity": "info",
                    "description": "Hardcoded entanglement strength - consider making configurable",
                    "cwe": None
                }
            ],
            
            "information_disclosure": [
                {
                    "pattern": r"(?i)print\s*\([^)]*password|print\s*\([^)]*secret|print\s*\([^)]*token",
                    "severity": "medium",
                    "description": "Potential secret information in print statement",
                    "cwe": "CWE-532"
                },
                {
                    "pattern": r"(?i)log.*\.(error|info|debug|warn).*password|log.*\.(error|info|debug|warn).*secret",
                    "severity": "medium",
                    "description": "Potential secret information in logs",
                    "cwe": "CWE-532"
                },
                {
                    "pattern": r"(?i)traceback\.print_exc\(\)",
                    "severity": "low",
                    "description": "Full stack trace disclosure - consider sanitizing in production",
                    "cwe": "CWE-209"
                }
            ]
        }
    
    def scan(self) -> SecurityScanResult:
        """Perform comprehensive security scan."""
        
        start_time = datetime.now()
        result = SecurityScanResult()
        
        python_files = list(self.root_path.rglob("*.py"))
        python_files = [f for f in python_files if not any(excluded in str(f) for excluded in self.excluded_paths)]
        
        result.total_files_scanned = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    result.total_lines_scanned += len(lines)
                    
                    file_issues = self._scan_file(file_path, content, lines)
                    result.issues.extend(file_issues)
                    
            except Exception as e:
                print(f"Warning: Could not scan {file_path}: {e}")
        
        # Calculate security score
        result.security_score = self._calculate_security_score(result)
        
        end_time = datetime.now()
        result.scan_duration_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _scan_file(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Scan individual file for security issues."""
        
        issues = []
        
        for category, patterns in self.scan_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config["pattern"]
                severity = pattern_config["severity"]
                description = pattern_config["description"]
                cwe = pattern_config.get("cwe")
                
                # Search for pattern in content
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # Skip if in comments (basic check)
                        line_stripped = line.strip()
                        if line_stripped.startswith('#') or line_stripped.startswith('"""') or line_stripped.startswith("'''"):
                            continue
                            
                        # Skip test files for some patterns
                        if "test_" in str(file_path) and severity in ["info", "low"]:
                            continue
                        
                        issue = SecurityIssue(
                            severity=severity,
                            category=category,
                            file_path=str(file_path.relative_to(self.root_path)),
                            line_number=line_num,
                            description=description,
                            recommendation=self._get_recommendation(category, pattern_config),
                            cwe_id=cwe,
                            confidence=self._calculate_confidence(line, pattern)
                        )
                        
                        issues.append(issue)
        
        return issues
    
    def _get_recommendation(self, category: str, pattern_config: Dict[str, Any]) -> str:
        """Get security recommendation for specific issue type."""
        
        recommendations = {
            "hardcoded_secrets": "Use environment variables or secure configuration management for secrets",
            "sql_injection": "Use parameterized queries or ORM methods to prevent SQL injection",
            "command_injection": "Avoid shell=True, validate inputs, use subprocess with list arguments",
            "cryptographic_issues": "Use strong cryptographic functions (SHA-256+), secure random generators",
            "input_validation": "Validate and sanitize all user inputs, use safe deserialization methods",
            "quantum_specific": "Use configuration files or environment variables for quantum parameters",
            "information_disclosure": "Avoid logging sensitive information, sanitize error messages"
        }
        
        return recommendations.get(category, "Review and remediate security issue")
    
    def _calculate_confidence(self, line: str, pattern: str) -> float:
        """Calculate confidence level for the detection."""
        
        # Simple heuristic-based confidence calculation
        confidence = 1.0
        
        # Reduce confidence if in test files
        if "test" in line.lower():
            confidence *= 0.7
        
        # Reduce confidence if pattern is very generic
        if len(pattern) < 20:
            confidence *= 0.8
        
        # Increase confidence for specific patterns
        if "password" in pattern.lower() or "secret" in pattern.lower():
            confidence *= 1.1
        
        return min(confidence, 1.0)
    
    def _calculate_security_score(self, result: SecurityScanResult) -> float:
        """Calculate overall security score (0-10, higher is better)."""
        
        if result.total_lines_scanned == 0:
            return 10.0  # No code to scan
        
        severity_weights = {
            "critical": -5.0,
            "high": -3.0,
            "medium": -1.5,
            "low": -0.5,
            "info": -0.1
        }
        
        total_deduction = 0.0
        severity_counts = result.get_severity_counts()
        
        for severity, count in severity_counts.items():
            if severity in severity_weights:
                total_deduction += severity_weights[severity] * count
        
        # Base score of 10, subtract deductions
        score = 10.0 + total_deduction
        
        # Normalize to 0-10 range
        score = max(0.0, min(10.0, score))
        
        # Bonus for large codebase with few issues
        if result.total_lines_scanned > 1000 and len(result.issues) < 10:
            score += 0.5
        
        return score


def generate_security_report(scan_result: SecurityScanResult, output_file: Optional[str] = None) -> str:
    """Generate comprehensive security report."""
    
    severity_counts = scan_result.get_severity_counts()
    issues_by_severity = scan_result.get_issues_by_severity()
    
    report = f"""
# Quantum Task Planner - Security Scan Report

**Scan Timestamp:** {scan_result.scan_timestamp}  
**Scan Duration:** {scan_result.scan_duration_seconds:.2f} seconds  
**Files Scanned:** {scan_result.total_files_scanned}  
**Lines Scanned:** {scan_result.total_lines_scanned:,}  
**Security Score:** {scan_result.security_score:.1f}/10.0  

## Summary

| Severity | Count |
|----------|-------|
| Critical | {severity_counts['critical']} |
| High     | {severity_counts['high']} |
| Medium   | {severity_counts['medium']} |
| Low      | {severity_counts['low']} |
| Info     | {severity_counts['info']} |
| **Total** | **{len(scan_result.issues)}** |

## Security Assessment

"""
    
    if scan_result.security_score >= 8.0:
        report += "🟢 **EXCELLENT** - Very few security issues detected. Good security practices followed.\n\n"
    elif scan_result.security_score >= 6.0:
        report += "🟡 **GOOD** - Some security issues detected. Address high-priority issues.\n\n"
    elif scan_result.security_score >= 4.0:
        report += "🟠 **MODERATE** - Several security issues detected. Remediation recommended.\n\n"
    else:
        report += "🔴 **POOR** - Many security issues detected. Immediate attention required.\n\n"
    
    # Detail issues by severity
    for severity in ["critical", "high", "medium", "low", "info"]:
        if severity in issues_by_severity and issues_by_severity[severity]:
            report += f"## {severity.upper()} Issues\n\n"
            
            for issue in issues_by_severity[severity]:
                report += f"### {issue.description}\n"
                report += f"**File:** `{issue.file_path}:{issue.line_number}`  \n"
                report += f"**Category:** {issue.category}  \n"
                if issue.cwe_id:
                    report += f"**CWE:** {issue.cwe_id}  \n"
                report += f"**Confidence:** {issue.confidence:.0%}  \n"
                report += f"**Recommendation:** {issue.recommendation}\n\n"
                report += "---\n\n"
    
    # Recommendations section
    report += """
## General Security Recommendations

1. **Secrets Management**
   - Use environment variables for all secrets and API keys
   - Implement proper secrets rotation
   - Consider using secret management tools (HashiCorp Vault, AWS Secrets Manager)

2. **Input Validation**
   - Validate and sanitize all user inputs
   - Use parameterized queries for database operations
   - Implement proper error handling without information disclosure

3. **Cryptography**
   - Use strong cryptographic algorithms (SHA-256+, AES-256)
   - Use cryptographically secure random number generators
   - Implement proper key management

4. **Quantum-Specific Security**
   - Protect quantum parameters and algorithms from tampering
   - Implement quantum-safe cryptographic practices
   - Monitor quantum coherence integrity

5. **Monitoring & Logging**
   - Implement security event logging
   - Monitor for suspicious patterns
   - Sanitize logs to prevent information disclosure

## Next Steps

1. Address all **CRITICAL** and **HIGH** severity issues immediately
2. Create remediation plan for **MEDIUM** severity issues  
3. Review and enhance security testing procedures
4. Consider implementing automated security scanning in CI/CD pipeline
5. Regular security code reviews and penetration testing

---

*Generated by Quantum Security Scanner*
"""
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Security report written to: {output_file}")
    
    return report


def run_security_audit() -> Dict[str, Any]:
    """Run complete security audit of the quantum system."""
    
    print("🔐 Starting Quantum Security Audit...")
    
    # Initialize scanner
    root_path = Path(__file__).parent.parent
    scanner = QuantumSecurityScanner(str(root_path))
    
    # Run scan
    scan_result = scanner.scan()
    
    # Generate report
    report_path = root_path / "security_scan_report.md"
    report_content = generate_security_report(scan_result, str(report_path))
    
    # Create JSON output for automation
    json_result = {
        "scan_timestamp": scan_result.scan_timestamp,
        "security_score": scan_result.security_score,
        "total_issues": len(scan_result.issues),
        "severity_counts": scan_result.get_severity_counts(),
        "scan_duration_seconds": scan_result.scan_duration_seconds,
        "files_scanned": scan_result.total_files_scanned,
        "lines_scanned": scan_result.total_lines_scanned,
        "issues": [
            {
                "severity": issue.severity,
                "category": issue.category,
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "description": issue.description,
                "recommendation": issue.recommendation,
                "cwe_id": issue.cwe_id,
                "confidence": issue.confidence
            }
            for issue in scan_result.issues
        ]
    }
    
    # Save JSON report
    json_path = root_path / "security_scan_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2)
    
    print(f"✅ Security scan completed!")
    print(f"📊 Security Score: {scan_result.security_score:.1f}/10.0")
    print(f"🔍 Issues Found: {len(scan_result.issues)}")
    print(f"📄 Report: {report_path}")
    print(f"📋 JSON Results: {json_path}")
    
    return json_result


if __name__ == "__main__":
    result = run_security_audit()
    
    # Exit with error code if critical issues found
    severity_counts = result["severity_counts"]
    if severity_counts["critical"] > 0:
        print(f"❌ CRITICAL SECURITY ISSUES FOUND: {severity_counts['critical']}")
        sys.exit(1)
    elif severity_counts["high"] > 5:  # More than 5 high-severity issues
        print(f"⚠️  HIGH SECURITY RISK: {severity_counts['high']} high-severity issues")
        sys.exit(1)
    else:
        print("✅ No critical security issues detected")
        sys.exit(0)