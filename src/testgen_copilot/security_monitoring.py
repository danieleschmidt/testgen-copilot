"""Advanced security monitoring and threat detection for TestGen Copilot."""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .logging_config import get_core_logger
from .resilience import circuit_breaker, retry, CircuitBreakerConfig, RetryConfig


class ThreatLevel(Enum):
    """Security threat levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatCategory(Enum):
    """Categories of security threats."""
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    SECRETS_EXPOSURE = "secrets_exposure"
    MALICIOUS_PATTERNS = "malicious_patterns"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_LEAKAGE = "data_leakage"
    SUPPLY_CHAIN = "supply_chain"
    CONFIGURATION = "configuration"


@dataclass
class SecurityThreat:
    """Detected security threat."""
    id: str
    level: ThreatLevel
    category: ThreatCategory
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False


@dataclass
class SecurityEvent:
    """Security monitoring event."""
    id: str
    event_type: str
    severity: ThreatLevel
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SecurityPatterns:
    """Advanced security patterns and detection rules."""
    
    # Code injection patterns
    CODE_INJECTION_PATTERNS = [
        (r'eval\s*\(', ThreatLevel.CRITICAL, "Direct eval() usage - code injection risk"),
        (r'exec\s*\(', ThreatLevel.CRITICAL, "Direct exec() usage - code injection risk"),
        (r'__import__\s*\(.*input\s*\(', ThreatLevel.HIGH, "Dynamic import with user input"),
        (r'compile\s*\(.*input\s*\(', ThreatLevel.HIGH, "Code compilation with user input"),
        (r'os\.system\s*\(.*input\s*\(', ThreatLevel.CRITICAL, "System command with user input"),
        (r'subprocess\.\w+\(.*shell\s*=\s*True', ThreatLevel.HIGH, "Shell command execution"),
        (r'pickle\.loads?\s*\(', ThreatLevel.HIGH, "Pickle deserialization - potential RCE"),
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        (r'\.\.[\\/]', ThreatLevel.HIGH, "Directory traversal attempt"),
        (r'os\.path\.join\(.*\.\.\/', ThreatLevel.MEDIUM, "Path join with traversal"),
        (r'open\s*\(["\']?[^"\']*\.\.\/', ThreatLevel.MEDIUM, "File access with traversal"),
        (r'/etc/passwd', ThreatLevel.HIGH, "System file access attempt"),
        (r'/proc/self', ThreatLevel.HIGH, "Process information access"),
        (r'["\']\.\.[\\/]', ThreatLevel.MEDIUM, "Relative path traversal"),
    ]
    
    # Secrets and sensitive data patterns
    SECRETS_PATTERNS = [
        (r'password\s*=\s*["\'][^"\']{3,}["\']', ThreatLevel.HIGH, "Hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']{10,}["\']', ThreatLevel.HIGH, "Hardcoded API key"),
        (r'secret_key\s*=\s*["\'][^"\']{10,}["\']', ThreatLevel.HIGH, "Hardcoded secret key"),
        (r'token\s*=\s*["\'][^"\']{20,}["\']', ThreatLevel.HIGH, "Hardcoded token"),
        (r'-----BEGIN [A-Z ]+-----', ThreatLevel.MEDIUM, "Embedded certificate/key"),
        (r'["\'][A-Za-z0-9]{40,}["\']', ThreatLevel.LOW, "Potential secret string"),
        (r'mysql://[^"\']+', ThreatLevel.MEDIUM, "Database connection string"),
        (r'postgresql://[^"\']+', ThreatLevel.MEDIUM, "PostgreSQL connection string"),
        (r'mongodb://[^"\']+', ThreatLevel.MEDIUM, "MongoDB connection string"),
    ]
    
    # Malicious patterns
    MALICIOUS_PATTERNS = [
        (r'rm\s+-rf\s+/', ThreatLevel.CRITICAL, "Dangerous file deletion command"),
        (r'curl\s+[^|]*\s*\|\s*sh', ThreatLevel.CRITICAL, "Pipe to shell execution"),
        (r'wget\s+[^|]*\s*\|\s*sh', ThreatLevel.CRITICAL, "Pipe to shell execution"),
        (r'nc\s+-l\s+-p', ThreatLevel.HIGH, "Netcat backdoor listener"),
        (r'/dev/tcp/', ThreatLevel.MEDIUM, "TCP socket redirection"),
        (r'base64\s*-d.*\|\s*sh', ThreatLevel.HIGH, "Base64 decode and execute"),
        (r'echo\s+[A-Za-z0-9+/=]{50,}\s*\|\s*base64\s*-d', ThreatLevel.HIGH, "Base64 encoded payload"),
    ]
    
    # Resource exhaustion patterns
    RESOURCE_EXHAUSTION_PATTERNS = [
        (r'while\s+True\s*:', ThreatLevel.MEDIUM, "Infinite loop detected"),
        (r'for\s+\w+\s+in\s+range\s*\(\s*\d{6,}', ThreatLevel.MEDIUM, "Large iteration range"),
        (r'[\'"]\*[\'"]\s*\*\s*\d{6,}', ThreatLevel.MEDIUM, "Memory exhaustion pattern"),
        (r'open\([^)]+,\s*["\']w["\'].*\.seek\(\d{9,}\)', ThreatLevel.MEDIUM, "Large file write"),
        (r'threading\.Thread.*target.*while', ThreatLevel.MEDIUM, "Potential thread bomb"),
    ]
    
    # Privilege escalation patterns
    PRIVILEGE_ESCALATION_PATTERNS = [
        (r'sudo\s+', ThreatLevel.MEDIUM, "Sudo usage detected"),
        (r'setuid\s*\(', ThreatLevel.HIGH, "UID manipulation"),
        (r'setgid\s*\(', ThreatLevel.HIGH, "GID manipulation"),
        (r'os\.setuid', ThreatLevel.HIGH, "Process UID change"),
        (r'os\.setgid', ThreatLevel.HIGH, "Process GID change"),
        (r'chmod\s+[47][07][07]', ThreatLevel.MEDIUM, "Executable permissions"),
    ]


class SecurityScanner:
    """Advanced security scanner with threat detection."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self.patterns = SecurityPatterns()
        self.scan_history: List[SecurityThreat] = []
        self.events_history: List[SecurityEvent] = []
        self.file_hashes: Dict[str, str] = {}
        self.scan_count = 0
        self.threat_counts = defaultdict(int)
        
    @circuit_breaker("security_scan", CircuitBreakerConfig(failure_threshold=3, timeout_duration_seconds=30))
    @retry("security_scan", RetryConfig(max_attempts=2))
    def scan_file(self, file_path: str) -> List[SecurityThreat]:
        """Perform comprehensive security scan on a file."""
        try:
            self.scan_count += 1
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                self.logger.warning("Security scan skipped - file not found", {
                    "file_path": file_path,
                    "scan_id": self.scan_count
                })
                return []
                
            # Read and hash file for integrity checking
            content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if file was modified since last scan
            if file_path in self.file_hashes and self.file_hashes[file_path] == file_hash:
                self.logger.debug("File unchanged since last scan", {
                    "file_path": file_path,
                    "file_hash": file_hash[:16]
                })
                
            self.file_hashes[file_path] = file_hash
            
            threats = []
            lines = content.split('\n')
            
            # Scan for various threat categories
            threats.extend(self._scan_code_injection(content, lines, file_path))
            threats.extend(self._scan_path_traversal(content, lines, file_path))
            threats.extend(self._scan_secrets(content, lines, file_path))
            threats.extend(self._scan_malicious_patterns(content, lines, file_path))
            threats.extend(self._scan_resource_exhaustion(content, lines, file_path))
            threats.extend(self._scan_privilege_escalation(content, lines, file_path))
            
            # Additional contextual analysis
            threats.extend(self._analyze_imports(content, lines, file_path))
            threats.extend(self._analyze_network_usage(content, lines, file_path))
            
            # Update threat statistics
            for threat in threats:
                self.threat_counts[threat.category] += 1
                
            self.scan_history.extend(threats)
            
            if threats:
                self.logger.warning("Security threats detected", {
                    "file_path": file_path,
                    "threat_count": len(threats),
                    "severity_breakdown": {
                        level.value: sum(1 for t in threats if t.level == level)
                        for level in ThreatLevel
                    },
                    "scan_id": self.scan_count
                })
            else:
                self.logger.debug("No security threats detected", {
                    "file_path": file_path,
                    "scan_id": self.scan_count
                })
                
            return threats
            
        except Exception as e:
            self._log_security_event(
                "security_scan_error",
                ThreatLevel.MEDIUM,
                f"Security scan failed: {str(e)}",
                file_path,
                {"error_type": type(e).__name__}
            )
            raise
            
    def _scan_patterns(self, patterns: List[Tuple[str, ThreatLevel, str]], 
                      content: str, lines: List[str], file_path: str,
                      category: ThreatCategory) -> List[SecurityThreat]:
        """Scan content against a list of patterns."""
        threats = []
        
        for pattern, level, description in patterns:
            try:
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        threat = SecurityThreat(
                            id=f"{category.value}_{file_path}_{line_num}_{match.start()}",
                            level=level,
                            category=category,
                            title=f"{category.value.replace('_', ' ').title()} Detected",
                            description=description,
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip()[:200],
                            recommendation=self._get_recommendation(category, pattern),
                            metadata={
                                "pattern": pattern,
                                "match_text": match.group(),
                                "match_start": match.start(),
                                "match_end": match.end()
                            }
                        )
                        threats.append(threat)
                        
            except re.error as e:
                self.logger.error("Invalid regex pattern", {
                    "pattern": pattern,
                    "category": category.value,
                    "error": str(e)
                })
                
        return threats
        
    def _scan_code_injection(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Scan for code injection vulnerabilities."""
        return self._scan_patterns(
            self.patterns.CODE_INJECTION_PATTERNS, 
            content, lines, file_path, 
            ThreatCategory.CODE_INJECTION
        )
        
    def _scan_path_traversal(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Scan for path traversal vulnerabilities."""
        return self._scan_patterns(
            self.patterns.PATH_TRAVERSAL_PATTERNS,
            content, lines, file_path,
            ThreatCategory.PATH_TRAVERSAL
        )
        
    def _scan_secrets(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Scan for exposed secrets and credentials."""
        return self._scan_patterns(
            self.patterns.SECRETS_PATTERNS,
            content, lines, file_path,
            ThreatCategory.SECRETS_EXPOSURE
        )
        
    def _scan_malicious_patterns(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Scan for malicious code patterns."""
        return self._scan_patterns(
            self.patterns.MALICIOUS_PATTERNS,
            content, lines, file_path,
            ThreatCategory.MALICIOUS_PATTERNS
        )
        
    def _scan_resource_exhaustion(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Scan for resource exhaustion vulnerabilities."""
        return self._scan_patterns(
            self.patterns.RESOURCE_EXHAUSTION_PATTERNS,
            content, lines, file_path,
            ThreatCategory.RESOURCE_EXHAUSTION
        )
        
    def _scan_privilege_escalation(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Scan for privilege escalation attempts."""
        return self._scan_patterns(
            self.patterns.PRIVILEGE_ESCALATION_PATTERNS,
            content, lines, file_path,
            ThreatCategory.PRIVILEGE_ESCALATION
        )
        
    def _analyze_imports(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Analyze imports for potentially dangerous modules."""
        threats = []
        dangerous_imports = [
            ('os', ThreatLevel.LOW, "OS module usage - review for command execution"),
            ('subprocess', ThreatLevel.MEDIUM, "Subprocess module - potential command injection"),
            ('pickle', ThreatLevel.MEDIUM, "Pickle module - deserialization risks"),
            ('marshal', ThreatLevel.MEDIUM, "Marshal module - code object risks"),
            ('__builtin__', ThreatLevel.HIGH, "Builtin module access - code injection risk"),
            ('sys', ThreatLevel.LOW, "Sys module usage - review for system manipulation"),
            ('socket', ThreatLevel.LOW, "Network socket usage - review for backdoors"),
            ('requests', ThreatLevel.LOW, "HTTP requests - review for data exfiltration"),
        ]
        
        for line_num, line in enumerate(lines, 1):
            for module, level, description in dangerous_imports:
                if re.search(rf'import\s+{module}|from\s+{module}\s+import', line, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        id=f"import_{file_path}_{line_num}_{module}",
                        level=level,
                        category=ThreatCategory.CONFIGURATION,
                        title=f"Potentially Dangerous Import: {module}",
                        description=description,
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation=f"Review usage of {module} module for security implications",
                        metadata={"imported_module": module}
                    ))
                    
        return threats
        
    def _analyze_network_usage(self, content: str, lines: List[str], file_path: str) -> List[SecurityThreat]:
        """Analyze network-related code for security issues."""
        threats = []
        network_patterns = [
            (r'socket\.socket\s*\(', ThreatLevel.MEDIUM, "Raw socket creation"),
            (r'urllib\.request\.urlopen\s*\([^)]*http://[^)]', ThreatLevel.LOW, "Insecure HTTP request"),
            (r'requests\.get\s*\([^)]*verify\s*=\s*False', ThreatLevel.HIGH, "SSL verification disabled"),
            (r'ssl\._create_unverified_context', ThreatLevel.HIGH, "SSL verification bypassed"),
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, level, description in network_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    threats.append(SecurityThreat(
                        id=f"network_{file_path}_{line_num}",
                        level=level,
                        category=ThreatCategory.CONFIGURATION,
                        title="Network Security Issue",
                        description=description,
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation="Review network security configuration",
                        metadata={"pattern_matched": pattern}
                    ))
                    
        return threats
        
    def _get_recommendation(self, category: ThreatCategory, pattern: str) -> str:
        """Get security recommendation for a threat category."""
        recommendations = {
            ThreatCategory.CODE_INJECTION: "Avoid dynamic code execution. Use safe alternatives and input validation.",
            ThreatCategory.PATH_TRAVERSAL: "Validate file paths. Use Path.resolve() and check against allowed directories.",
            ThreatCategory.SECRETS_EXPOSURE: "Move secrets to environment variables or secure key management.",
            ThreatCategory.MALICIOUS_PATTERNS: "Remove malicious code patterns. Review code purpose and legitimacy.",
            ThreatCategory.RESOURCE_EXHAUSTION: "Add resource limits and bounds checking to prevent DoS.",
            ThreatCategory.PRIVILEGE_ESCALATION: "Review privilege requirements. Use least-privilege principle.",
            ThreatCategory.DATA_LEAKAGE: "Implement data sanitization and access controls.",
            ThreatCategory.CONFIGURATION: "Review configuration for security best practices.",
        }
        
        return recommendations.get(category, "Review code for security implications.")
        
    def _log_security_event(self, event_type: str, severity: ThreatLevel, 
                           message: str, source: str, metadata: Dict[str, Any] = None) -> None:
        """Log a security monitoring event."""
        event = SecurityEvent(
            id=f"event_{int(time.time())}_{len(self.events_history)}",
            event_type=event_type,
            severity=severity,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        self.events_history.append(event)
        
        self.logger.error("Security event logged", {
            "event_id": event.id,
            "event_type": event_type,
            "severity": severity.value,
            "message": message,
            "source": source,
            "metadata": metadata
        })
        
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        now = datetime.now(timezone.utc)
        
        # Recent threats (last 24 hours)
        recent_threats = [
            t for t in self.scan_history 
            if (now - t.timestamp) < timedelta(hours=24)
        ]
        
        # Threat statistics
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for threat in recent_threats:
            severity_counts[threat.level.value] += 1
            category_counts[threat.category.value] += 1
            
        # High-risk files (files with multiple threats)
        file_threat_counts = defaultdict(int)
        for threat in recent_threats:
            file_threat_counts[threat.file_path] += 1
            
        high_risk_files = [
            {"file": file, "threat_count": count}
            for file, count in file_threat_counts.items()
            if count >= 3
        ]
        
        return {
            "report_timestamp": now.isoformat(),
            "scan_summary": {
                "total_scans": self.scan_count,
                "files_scanned": len(self.file_hashes),
                "total_threats": len(self.scan_history),
                "recent_threats": len(recent_threats),
                "unmitigated_threats": sum(1 for t in self.scan_history if not t.mitigated)
            },
            "severity_breakdown": dict(severity_counts),
            "category_breakdown": dict(category_counts),
            "high_risk_files": high_risk_files,
            "recent_threats": [
                {
                    "id": t.id,
                    "level": t.level.value,
                    "category": t.category.value,
                    "title": t.title,
                    "file_path": t.file_path,
                    "line_number": t.line_number,
                    "timestamp": t.timestamp.isoformat(),
                    "mitigated": t.mitigated
                }
                for t in recent_threats[-10:]  # Last 10 threats
            ],
            "security_trends": {
                "threat_growth": self._calculate_threat_growth(),
                "most_common_categories": sorted(
                    category_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            },
            "recommendations": self._generate_security_recommendations(recent_threats)
        }
        
    def _calculate_threat_growth(self) -> Dict[str, Any]:
        """Calculate threat growth trends."""
        now = datetime.now(timezone.utc)
        
        # Threats in last 24 hours vs previous 24 hours
        last_24h = sum(1 for t in self.scan_history 
                      if (now - t.timestamp) < timedelta(hours=24))
        prev_24h = sum(1 for t in self.scan_history 
                      if timedelta(hours=24) <= (now - t.timestamp) < timedelta(hours=48))
                      
        growth_rate = ((last_24h - prev_24h) / prev_24h * 100) if prev_24h > 0 else 0
        
        return {
            "last_24_hours": last_24h,
            "previous_24_hours": prev_24h,
            "growth_rate_percent": round(growth_rate, 2),
            "trend": "increasing" if growth_rate > 10 else "decreasing" if growth_rate < -10 else "stable"
        }
        
    def _generate_security_recommendations(self, recent_threats: List[SecurityThreat]) -> List[str]:
        """Generate actionable security recommendations."""
        recommendations = []
        
        # Critical threats
        critical_threats = [t for t in recent_threats if t.level == ThreatLevel.CRITICAL]
        if critical_threats:
            recommendations.append(
                f"URGENT: Address {len(critical_threats)} critical security threats immediately"
            )
            
        # Common categories
        category_counts = defaultdict(int)
        for threat in recent_threats:
            category_counts[threat.category] += 1
            
        if category_counts[ThreatCategory.SECRETS_EXPOSURE] > 0:
            recommendations.append(
                "Implement secrets management system to prevent credential exposure"
            )
            
        if category_counts[ThreatCategory.CODE_INJECTION] > 0:
            recommendations.append(
                "Review dynamic code execution patterns and implement input validation"
            )
            
        if category_counts[ThreatCategory.CONFIGURATION] > 2:
            recommendations.append(
                "Conduct security configuration review and implement hardening guidelines"
            )
            
        # High-risk files
        file_counts = defaultdict(int)
        for threat in recent_threats:
            file_counts[threat.file_path] += 1
            
        high_risk = [f for f, c in file_counts.items() if c >= 3]
        if high_risk:
            recommendations.append(
                f"Focus security review on {len(high_risk)} high-risk files with multiple threats"
            )
            
        if not recommendations:
            recommendations.append("Continue regular security scanning and monitoring")
            
        return recommendations
        
    def mitigate_threat(self, threat_id: str, mitigation_note: str = "") -> bool:
        """Mark a threat as mitigated."""
        for threat in self.scan_history:
            if threat.id == threat_id:
                threat.mitigated = True
                threat.metadata["mitigation_note"] = mitigation_note
                threat.metadata["mitigated_at"] = datetime.now(timezone.utc).isoformat()
                
                self.logger.info("Security threat mitigated", {
                    "threat_id": threat_id,
                    "threat_category": threat.category.value,
                    "mitigation_note": mitigation_note
                })
                return True
                
        return False


# Global security scanner instance
_security_scanner: Optional[SecurityScanner] = None

def get_security_scanner() -> SecurityScanner:
    """Get or create the global security scanner instance."""
    global _security_scanner
    if _security_scanner is None:
        _security_scanner = SecurityScanner()
    return _security_scanner