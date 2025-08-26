"""
🛡️ Robust Validation Engine v2.0
================================

Comprehensive input validation, sanitization, and security enforcement system.
Implements defense-in-depth validation with automatic error recovery.
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from datetime import datetime
import unicodedata
import ipaddress

from .logging_config import get_core_logger
from .security_monitoring import SecurityScanner

logger = get_core_logger()


class ValidationSeverity(Enum):
    """Validation error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation results"""
    VALID = "valid"
    INVALID = "invalid"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    field: str
    message: str
    severity: ValidationSeverity
    suggested_fix: Optional[str] = None
    blocked_value: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    result: ValidationResult
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class RobustValidationEngine:
    """
    🛡️ Advanced validation engine with multi-layered security controls
    
    Features:
    - Input sanitization and normalization
    - SQL injection prevention
    - XSS attack protection
    - Path traversal detection
    - Command injection blocking
    - Data type validation
    - Business rule enforcement
    - Automatic error recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.security_scanner = SecurityScanner()
        
        # Compiled regex patterns for performance
        self._sql_injection_patterns = [
            re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
            re.compile(r"(--|/\*|\*/|;|\|)", re.IGNORECASE),
            re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"(SCRIPT|JAVASCRIPT|VBSCRIPT)", re.IGNORECASE)
        ]
        
        self._xss_patterns = [
            re.compile(r"<\s*script[^>]*>.*?</\s*script\s*>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<\s*iframe[^>]*>", re.IGNORECASE),
            re.compile(r"<\s*object[^>]*>", re.IGNORECASE),
            re.compile(r"<\s*embed[^>]*>", re.IGNORECASE)
        ]
        
        self._path_traversal_patterns = [
            re.compile(r"\.\.[\\/]"),
            re.compile(r"[\\/]\.\."),
            re.compile(r"%2e%2e"),
            re.compile(r"%2f"),
            re.compile(r"%5c")
        ]
        
        self._command_injection_patterns = [
            re.compile(r"[;&|`$(){}[\]<>]"),
            re.compile(r"\b(eval|exec|system|shell_exec|passthru)\b", re.IGNORECASE),
            re.compile(r"(\||&&|;|&|\$\(|\`)")
        ]
        
        # Allowed patterns for strict validation
        self._safe_patterns = {
            "alphanumeric": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "filename": re.compile(r"^[a-zA-Z0-9._-]+$"),
            "url": re.compile(r"^https?://[^\s/$.?#].[^\s]*$"),
            "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
        }
        
        # Maximum lengths for different field types
        self._max_lengths = {
            "short_text": 100,
            "medium_text": 500,
            "long_text": 5000,
            "description": 2000,
            "filename": 255,
            "path": 4096,
            "url": 2048,
            "email": 254
        }
    
    def validate_comprehensive(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationReport:
        """
        Perform comprehensive validation with automatic sanitization
        """
        report = ValidationReport(result=ValidationResult.VALID)
        sanitized_data = {}
        
        try:
            for field_name, field_config in schema.items():
                if field_name in data:
                    field_result = self._validate_field(
                        field_name, 
                        data[field_name], 
                        field_config
                    )
                    
                    # Merge field issues into report
                    report.issues.extend(field_result.issues)
                    
                    # Update result severity
                    if field_result.result == ValidationResult.BLOCKED:
                        report.result = ValidationResult.BLOCKED
                    elif field_result.result == ValidationResult.INVALID and report.result != ValidationResult.BLOCKED:
                        report.result = ValidationResult.INVALID
                    elif field_result.result == ValidationResult.SANITIZED and report.result == ValidationResult.VALID:
                        report.result = ValidationResult.SANITIZED
                    
                    # Add sanitized value
                    if field_result.sanitized_data:
                        sanitized_data[field_name] = field_result.sanitized_data.get(field_name, data[field_name])
                    else:
                        sanitized_data[field_name] = data[field_name]
                
                elif field_config.get("required", False):
                    report.issues.append(ValidationIssue(
                        field=field_name,
                        message=f"Required field '{field_name}' is missing",
                        severity=ValidationSeverity.HIGH,
                        suggested_fix="Provide a valid value for this required field"
                    ))
                    report.result = ValidationResult.INVALID
            
            # Set sanitized data if any changes were made
            if sanitized_data != data:
                report.sanitized_data = sanitized_data
            
            # Add validation metadata
            report.metadata = {
                "validation_timestamp": datetime.now().isoformat(),
                "fields_validated": len(schema),
                "issues_found": len(report.issues),
                "critical_issues": len([i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]),
                "data_hash": self._calculate_data_hash(data)
            }
            
            # Log security events
            self._log_security_events(report)
            
        except Exception as e:
            logger.error(f"Validation engine error: {e}")
            report.result = ValidationResult.INVALID
            report.issues.append(ValidationIssue(
                field="system",
                message=f"Validation engine error: {str(e)}",
                severity=ValidationSeverity.CRITICAL
            ))
        
        return report
    
    def _validate_field(self, field_name: str, value: Any, config: Dict[str, Any]) -> ValidationReport:
        """Validate a single field with comprehensive checks"""
        report = ValidationReport(result=ValidationResult.VALID)
        sanitized_value = value
        
        # Type validation
        expected_type = config.get("type", "string")
        if not self._validate_type(value, expected_type):
            report.issues.append(ValidationIssue(
                field=field_name,
                message=f"Invalid type for field '{field_name}'. Expected {expected_type}, got {type(value).__name__}",
                severity=ValidationSeverity.HIGH,
                suggested_fix=f"Convert value to {expected_type}"
            ))
            report.result = ValidationResult.INVALID
            return report
        
        # Convert to string for pattern matching
        str_value = str(value) if value is not None else ""
        
        # Length validation
        max_length = config.get("max_length", self._max_lengths.get(expected_type, 1000))
        if len(str_value) > max_length:
            report.issues.append(ValidationIssue(
                field=field_name,
                message=f"Field '{field_name}' exceeds maximum length of {max_length}",
                severity=ValidationSeverity.MEDIUM,
                suggested_fix=f"Truncate to {max_length} characters"
            ))
            sanitized_value = str_value[:max_length]
            report.result = ValidationResult.SANITIZED
        
        # Security pattern validation
        security_issues = self._check_security_patterns(field_name, str_value)
        if security_issues:
            report.issues.extend(security_issues)
            
            # Determine if we should block or sanitize
            critical_issues = [i for i in security_issues if i.severity == ValidationSeverity.CRITICAL]
            if critical_issues:
                report.result = ValidationResult.BLOCKED
                return report
            else:
                # Attempt sanitization
                sanitized_value = self._sanitize_value(str_value, config)
                report.result = ValidationResult.SANITIZED
        
        # Business rule validation
        business_rules = config.get("business_rules", [])
        for rule in business_rules:
            if not self._validate_business_rule(sanitized_value, rule):
                severity = ValidationSeverity[rule.get("severity", "MEDIUM").upper()]
                report.issues.append(ValidationIssue(
                    field=field_name,
                    message=f"Business rule violation: {rule.get('message', 'Unknown rule')}",
                    severity=severity,
                    suggested_fix=rule.get("suggested_fix")
                ))
                if severity == ValidationSeverity.CRITICAL:
                    report.result = ValidationResult.BLOCKED
                    return report
                elif report.result == ValidationResult.VALID:
                    report.result = ValidationResult.INVALID
        
        # Pattern validation (if specified)
        pattern = config.get("pattern")
        if pattern and not re.match(pattern, str(sanitized_value)):
            report.issues.append(ValidationIssue(
                field=field_name,
                message=f"Field '{field_name}' does not match required pattern",
                severity=ValidationSeverity.MEDIUM,
                suggested_fix="Ensure value matches the required format"
            ))
            if report.result == ValidationResult.VALID:
                report.result = ValidationResult.INVALID
        
        # Set sanitized data if value was changed
        if sanitized_value != value:
            report.sanitized_data = {field_name: sanitized_value}
        
        return report
    
    def _check_security_patterns(self, field_name: str, value: str) -> List[ValidationIssue]:
        """Check for security vulnerabilities in input"""
        issues = []
        
        # SQL injection detection
        for pattern in self._sql_injection_patterns:
            if pattern.search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential SQL injection detected",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix="Remove SQL keywords and special characters",
                    blocked_value=value
                ))
                break
        
        # XSS detection
        for pattern in self._xss_patterns:
            if pattern.search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential XSS attack detected",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix="Remove HTML/JavaScript content",
                    blocked_value=value
                ))
                break
        
        # Path traversal detection
        for pattern in self._path_traversal_patterns:
            if pattern.search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential path traversal attack detected",
                    severity=ValidationSeverity.HIGH,
                    suggested_fix="Remove directory traversal sequences",
                    blocked_value=value
                ))
                break
        
        # Command injection detection
        for pattern in self._command_injection_patterns:
            if pattern.search(value):
                issues.append(ValidationIssue(
                    field=field_name,
                    message="Potential command injection detected",
                    severity=ValidationSeverity.CRITICAL,
                    suggested_fix="Remove shell metacharacters",
                    blocked_value=value
                ))
                break
        
        return issues
    
    def _sanitize_value(self, value: str, config: Dict[str, Any]) -> str:
        """Sanitize input value to remove security threats"""
        sanitized = value
        
        # Unicode normalization
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # HTML entity encoding for potential XSS
        sanitized = (sanitized
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters (except common whitespace)
        sanitized = ''.join(char for char in sanitized 
                          if unicodedata.category(char) != 'Cc' 
                          or char in '\t\n\r ')
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_validators = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "boolean": lambda x: isinstance(x, bool),
            "list": lambda x: isinstance(x, list),
            "dict": lambda x: isinstance(x, dict),
            "email": lambda x: isinstance(x, str) and self._safe_patterns["email"].match(x),
            "url": lambda x: isinstance(x, str) and self._safe_patterns["url"].match(x),
            "filename": lambda x: isinstance(x, str) and self._safe_patterns["filename"].match(x),
            "path": lambda x: isinstance(x, str) and self._is_safe_path(x),
            "ip_address": lambda x: self._is_valid_ip(x)
        }
        
        validator = type_validators.get(expected_type, lambda x: True)
        return validator(value)
    
    def _validate_business_rule(self, value: Any, rule: Dict[str, Any]) -> bool:
        """Validate against custom business rules"""
        rule_type = rule.get("type")
        
        if rule_type == "range":
            min_val, max_val = rule.get("min", float('-inf')), rule.get("max", float('inf'))
            return min_val <= float(value) <= max_val
        
        elif rule_type == "enum":
            allowed_values = rule.get("values", [])
            return value in allowed_values
        
        elif rule_type == "regex":
            pattern = rule.get("pattern")
            return bool(re.match(pattern, str(value))) if pattern else True
        
        elif rule_type == "custom":
            # Execute custom validation function
            func_name = rule.get("function")
            if func_name and hasattr(self, f"_validate_{func_name}"):
                validator = getattr(self, f"_validate_{func_name}")
                return validator(value, rule.get("params", {}))
        
        return True
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is safe (no traversal attacks)"""
        try:
            # Resolve path and check if it stays within bounds
            resolved = Path(path).resolve()
            return not any(pattern.search(str(resolved)) for pattern in self._path_traversal_patterns)
        except (OSError, ValueError):
            return False
    
    def _is_valid_ip(self, value: str) -> bool:
        """Validate IP address"""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of input data for integrity checking"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _log_security_events(self, report: ValidationReport) -> None:
        """Log security events for monitoring"""
        critical_issues = [i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]
        
        if critical_issues:
            for issue in critical_issues:
                self.security_scanner.scan_text(
                    event_type="validation_blocked",
                    severity="critical",
                    details={
                        "field": issue.field,
                        "message": issue.message,
                        "blocked_value": issue.blocked_value
                    }
                )
        
        # Log validation summary
        if report.issues:
            self.security_monitor.log_security_event(
                event_type="validation_issues",
                severity="info" if report.result != ValidationResult.BLOCKED else "warning",
                details={
                    "result": report.result.value,
                    "total_issues": len(report.issues),
                    "critical_issues": len([i for i in report.issues if i.severity == ValidationSeverity.CRITICAL])
                }
            )


# Pre-built validation schemas for common use cases
VALIDATION_SCHEMAS = {
    "user_input": {
        "username": {
            "type": "string",
            "max_length": 50,
            "pattern": "^[a-zA-Z0-9_-]+$",
            "required": True
        },
        "email": {
            "type": "email",
            "max_length": 254,
            "required": True
        },
        "password": {
            "type": "string",
            "max_length": 128,
            "business_rules": [
                {
                    "type": "regex",
                    "pattern": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
                    "message": "Password must contain uppercase, lowercase, number, and special character",
                    "severity": "HIGH"
                }
            ],
            "required": True
        }
    },
    
    "file_operations": {
        "filename": {
            "type": "filename",
            "max_length": 255,
            "required": True
        },
        "path": {
            "type": "path",
            "max_length": 4096,
            "required": True
        },
        "content": {
            "type": "string",
            "max_length": 1048576  # 1MB
        }
    },
    
    "api_request": {
        "endpoint": {
            "type": "url",
            "max_length": 2048,
            "required": True
        },
        "method": {
            "type": "string",
            "business_rules": [
                {
                    "type": "enum",
                    "values": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "message": "Invalid HTTP method",
                    "severity": "HIGH"
                }
            ],
            "required": True
        },
        "headers": {
            "type": "dict",
            "max_length": 8192
        },
        "body": {
            "type": "string",
            "max_length": 10485760  # 10MB
        }
    }
}