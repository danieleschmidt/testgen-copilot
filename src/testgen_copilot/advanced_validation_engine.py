"""Advanced Validation Engine - Generation 2 Implementation

Comprehensive input validation, sanitization, and data integrity
verification with ML-powered anomaly detection.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from .logging_config import get_logger


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Supported data types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    DATE = "date"
    UUID = "uuid"
    JSON = "json"
    BASE64 = "base64"
    HEX = "hex"


@dataclass
class ValidationRule:
    """Individual validation rule."""
    name: str
    validator: Callable[[Any], Tuple[bool, str]]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Any = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaField:
    """Schema field definition."""
    name: str
    data_type: DataType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validators: List[ValidationRule] = field(default_factory=list)


@dataclass
class ValidationSchema:
    """Complete validation schema."""
    name: str
    fields: List[SchemaField]
    strict: bool = True  # Reject unknown fields
    allow_empty: bool = False
    custom_rules: List[ValidationRule] = field(default_factory=list)


class AdvancedValidationEngine:
    """Comprehensive validation and sanitization engine."""
    
    def __init__(self):
        """Initialize validation engine."""
        self.logger = get_logger(__name__)
        
        # Validation history for anomaly detection
        self._validation_history: List[Dict[str, Any]] = []
        
        # Common validation patterns
        self._setup_patterns()
        
        # Built-in schemas
        self._schemas: Dict[str, ValidationSchema] = {}
        self._setup_builtin_schemas()
        
        # Anomaly detection parameters
        self._anomaly_threshold = 2.0  # Standard deviations
        self._min_samples_for_anomaly = 10
        
        self.logger.info("Advanced validation engine initialized")
    
    def _setup_patterns(self):
        """Setup common validation patterns."""
        self.patterns = {
            "email": re.compile(
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            ),
            "url": re.compile(
                r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$"
            ),
            "ipv4": re.compile(
                r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            ),
            "ipv6": re.compile(
                r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
            ),
            "uuid": re.compile(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
            ),
            "hex": re.compile(r"^[0-9a-fA-F]+$"),
            "base64": re.compile(r"^[A-Za-z0-9+/]*={0,2}$"),
            "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
            "alpha": re.compile(r"^[a-zA-Z]+$"),
            "numeric": re.compile(r"^[0-9]+$"),
            "phone": re.compile(r"^\+?[1-9]\d{1,14}$"),  # E.164 format
            "credit_card": re.compile(r"^[0-9]{13,19}$"),
            "social_security": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
            "safe_filename": re.compile(r"^[a-zA-Z0-9._-]+$"),
            "sql_keywords": re.compile(
                r"(?i)(select|insert|update|delete|drop|create|alter|exec|union|script)",
                re.IGNORECASE
            ),
            "html_tags": re.compile(r"<[^>]+>"),
            "javascript": re.compile(
                r"(?i)(javascript:|vbscript:|onload|onerror|onclick|onmouseover)",
                re.IGNORECASE
            )
        }
    
    def _setup_builtin_schemas(self):
        """Setup built-in validation schemas."""
        # User registration schema
        user_schema = ValidationSchema(
            name="user_registration",
            fields=[
                SchemaField(
                    name="username",
                    data_type=DataType.STRING,
                    required=True,
                    min_length=3,
                    max_length=50,
                    pattern=r"^[a-zA-Z0-9_]+$"
                ),
                SchemaField(
                    name="email",
                    data_type=DataType.EMAIL,
                    required=True
                ),
                SchemaField(
                    name="password",
                    data_type=DataType.STRING,
                    required=True,
                    min_length=8,
                    max_length=128
                ),
                SchemaField(
                    name="age",
                    data_type=DataType.INTEGER,
                    min_value=13,
                    max_value=120
                )
            ],
            strict=True
        )
        self.register_schema(user_schema)
        
        # API request schema
        api_schema = ValidationSchema(
            name="api_request",
            fields=[
                SchemaField(
                    name="endpoint",
                    data_type=DataType.STRING,
                    required=True,
                    pattern=r"^/[a-zA-Z0-9/_-]*$"
                ),
                SchemaField(
                    name="method",
                    data_type=DataType.STRING,
                    required=True,
                    allowed_values=["GET", "POST", "PUT", "DELETE", "PATCH"]
                ),
                SchemaField(
                    name="payload",
                    data_type=DataType.DICT
                )
            ]
        )
        self.register_schema(api_schema)
    
    def register_schema(self, schema: ValidationSchema):
        """Register a validation schema."""
        self._schemas[schema.name] = schema
        self.logger.debug(f"Registered schema: {schema.name}")
    
    def validate_by_schema(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """Validate data against a registered schema."""
        if schema_name not in self._schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema '{schema_name}' not found"]
            )
        
        schema = self._schemas[schema_name]
        return self._validate_against_schema(data, schema)
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: ValidationSchema) -> ValidationResult:
        """Validate data against schema."""
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                errors=["Data must be a dictionary"]
            )
        
        # Validate required fields
        for field in schema.fields:
            if field.required and field.name not in data:
                errors.append(f"Required field '{field.name}' is missing")
                continue
            
            if field.name in data:
                # Validate field
                field_result = self._validate_field(data[field.name], field)
                if not field_result.is_valid:
                    errors.extend([f"{field.name}: {error}" for error in field_result.errors])
                warnings.extend([f"{field.name}: {warning}" for warning in field_result.warnings])
                
                # Use sanitized data
                sanitized_data[field.name] = field_result.sanitized_data
        
        # Check for unknown fields in strict mode
        if schema.strict:
            schema_field_names = {field.name for field in schema.fields}
            unknown_fields = set(data.keys()) - schema_field_names
            if unknown_fields:
                errors.extend([f"Unknown field: {field}" for field in unknown_fields])
        else:
            # Include unknown fields in non-strict mode
            schema_field_names = {field.name for field in schema.fields}
            for field_name, field_value in data.items():
                if field_name not in schema_field_names:
                    sanitized_data[field_name] = field_value
        
        # Check empty data
        if not schema.allow_empty and not sanitized_data:
            errors.append("Empty data not allowed")
        
        # Apply custom schema rules
        for rule in schema.custom_rules:
            if rule.enabled:
                is_valid, message = rule.validator(sanitized_data)
                if not is_valid:
                    if rule.severity == ValidationSeverity.ERROR:
                        errors.append(f"Schema rule '{rule.name}': {message}")
                    else:
                        warnings.append(f"Schema rule '{rule.name}': {message}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data
        )
    
    def _validate_field(self, value: Any, field: SchemaField) -> ValidationResult:
        """Validate individual field."""
        errors = []
        warnings = []
        sanitized_value = value
        
        # Type validation
        type_result = self._validate_type(value, field.data_type)
        if not type_result.is_valid:
            errors.extend(type_result.errors)
            return ValidationResult(is_valid=False, errors=errors)
        
        sanitized_value = type_result.sanitized_data
        
        # Length validation for strings and lists
        if field.data_type in [DataType.STRING, DataType.LIST]:
            length = len(sanitized_value) if sanitized_value else 0
            
            if field.min_length is not None and length < field.min_length:
                errors.append(f"Length {length} is less than minimum {field.min_length}")
            
            if field.max_length is not None and length > field.max_length:
                errors.append(f"Length {length} exceeds maximum {field.max_length}")
        
        # Value range validation for numbers
        if field.data_type in [DataType.INTEGER, DataType.FLOAT] and sanitized_value is not None:
            if field.min_value is not None and sanitized_value < field.min_value:
                errors.append(f"Value {sanitized_value} is less than minimum {field.min_value}")
            
            if field.max_value is not None and sanitized_value > field.max_value:
                errors.append(f"Value {sanitized_value} exceeds maximum {field.max_value}")
        
        # Pattern validation for strings
        if field.pattern and field.data_type == DataType.STRING and sanitized_value:
            if not re.match(field.pattern, sanitized_value):
                errors.append(f"Value does not match required pattern: {field.pattern}")
        
        # Allowed values validation
        if field.allowed_values and sanitized_value not in field.allowed_values:
            errors.append(f"Value must be one of: {field.allowed_values}")
        
        # Custom field validators
        for rule in field.custom_validators:
            if rule.enabled:
                is_valid, message = rule.validator(sanitized_value)
                if not is_valid:
                    if rule.severity == ValidationSeverity.ERROR:
                        errors.append(f"Custom rule '{rule.name}': {message}")
                    else:
                        warnings.append(f"Custom rule '{rule.name}': {message}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_value
        )
    
    def _validate_type(self, value: Any, expected_type: DataType) -> ValidationResult:
        """Validate and convert data type."""
        if value is None:
            return ValidationResult(is_valid=True, sanitized_data=None)
        
        try:
            if expected_type == DataType.STRING:
                return ValidationResult(is_valid=True, sanitized_data=str(value))
            
            elif expected_type == DataType.INTEGER:
                if isinstance(value, int):
                    sanitized = value
                elif isinstance(value, str) and value.isdigit():
                    sanitized = int(value)
                elif isinstance(value, float) and value.is_integer():
                    sanitized = int(value)
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid integer"])
                return ValidationResult(is_valid=True, sanitized_data=sanitized)
            
            elif expected_type == DataType.FLOAT:
                if isinstance(value, (int, float)):
                    sanitized = float(value)
                elif isinstance(value, str):
                    sanitized = float(value)
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid float"])
                return ValidationResult(is_valid=True, sanitized_data=sanitized)
            
            elif expected_type == DataType.BOOLEAN:
                if isinstance(value, bool):
                    sanitized = value
                elif isinstance(value, str):
                    sanitized = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(value, int):
                    sanitized = bool(value)
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid boolean"])
                return ValidationResult(is_valid=True, sanitized_data=sanitized)
            
            elif expected_type == DataType.LIST:
                if isinstance(value, list):
                    sanitized = value
                elif isinstance(value, str):
                    # Try to parse as JSON array
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            sanitized = parsed
                        else:
                            return ValidationResult(is_valid=False, errors=["String is not a JSON array"])
                    except json.JSONDecodeError:
                        return ValidationResult(is_valid=False, errors=["Invalid JSON array"])
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid list"])
                return ValidationResult(is_valid=True, sanitized_data=sanitized)
            
            elif expected_type == DataType.DICT:
                if isinstance(value, dict):
                    sanitized = value
                elif isinstance(value, str):
                    # Try to parse as JSON object
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            sanitized = parsed
                        else:
                            return ValidationResult(is_valid=False, errors=["String is not a JSON object"])
                    except json.JSONDecodeError:
                        return ValidationResult(is_valid=False, errors=["Invalid JSON object"])
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid dictionary"])
                return ValidationResult(is_valid=True, sanitized_data=sanitized)
            
            elif expected_type == DataType.EMAIL:
                value_str = str(value)
                if self.patterns["email"].match(value_str):
                    return ValidationResult(is_valid=True, sanitized_data=value_str.lower())
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid email format"])
            
            elif expected_type == DataType.URL:
                value_str = str(value)
                if self.patterns["url"].match(value_str):
                    return ValidationResult(is_valid=True, sanitized_data=value_str)
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid URL format"])
            
            elif expected_type == DataType.IP_ADDRESS:
                value_str = str(value)
                if self.patterns["ipv4"].match(value_str) or self.patterns["ipv6"].match(value_str):
                    return ValidationResult(is_valid=True, sanitized_data=value_str)
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid IP address format"])
            
            elif expected_type == DataType.UUID:
                value_str = str(value)
                if self.patterns["uuid"].match(value_str):
                    return ValidationResult(is_valid=True, sanitized_data=value_str.lower())
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid UUID format"])
            
            elif expected_type == DataType.JSON:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        return ValidationResult(is_valid=True, sanitized_data=parsed)
                    except json.JSONDecodeError:
                        return ValidationResult(is_valid=False, errors=["Invalid JSON"])
                else:
                    return ValidationResult(is_valid=True, sanitized_data=value)
            
            elif expected_type == DataType.BASE64:
                value_str = str(value)
                if self.patterns["base64"].match(value_str):
                    try:
                        import base64
                        base64.b64decode(value_str, validate=True)
                        return ValidationResult(is_valid=True, sanitized_data=value_str)
                    except Exception:
                        return ValidationResult(is_valid=False, errors=["Invalid base64 encoding"])
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid base64 format"])
            
            elif expected_type == DataType.HEX:
                value_str = str(value)
                if self.patterns["hex"].match(value_str):
                    return ValidationResult(is_valid=True, sanitized_data=value_str.upper())
                else:
                    return ValidationResult(is_valid=False, errors=["Invalid hexadecimal format"])
            
            else:
                return ValidationResult(is_valid=False, errors=[f"Unsupported data type: {expected_type}"])
        
        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"Type validation error: {str(e)}"])
    
    def sanitize_string(self, value: str, 
                       remove_html: bool = True,
                       remove_javascript: bool = True,
                       remove_sql_keywords: bool = True,
                       max_length: Optional[int] = None) -> str:
        """Comprehensive string sanitization."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove HTML tags
        if remove_html:
            value = self.patterns["html_tags"].sub("", value)
        
        # Remove JavaScript
        if remove_javascript:
            value = self.patterns["javascript"].sub("", value)
        
        # Remove SQL keywords (basic protection)
        if remove_sql_keywords:
            value = self.patterns["sql_keywords"].sub("", value)
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        # Truncate if necessary
        if max_length and len(value) > max_length:
            value = value[:max_length]
        
        return value
    
    def detect_anomalies(self, data: Dict[str, Any]) -> List[str]:
        """Detect anomalies in data using statistical analysis."""
        anomalies = []
        
        if len(self._validation_history) < self._min_samples_for_anomaly:
            return anomalies
        
        try:
            # Analyze string lengths
            if isinstance(data.get('username'), str):
                username_lengths = [
                    len(entry.get('username', ''))
                    for entry in self._validation_history
                    if 'username' in entry
                ]
                if username_lengths:
                    mean_len = statistics.mean(username_lengths)
                    std_len = statistics.stdev(username_lengths) if len(username_lengths) > 1 else 0
                    current_len = len(data['username'])
                    
                    if std_len > 0 and abs(current_len - mean_len) > self._anomaly_threshold * std_len:
                        anomalies.append(f"Anomalous username length: {current_len} (expected ~{mean_len:.1f})")
            
            # Analyze data patterns
            data_keys = set(data.keys())
            historical_keys = [
                set(entry.keys())
                for entry in self._validation_history[-50:]  # Last 50 entries
            ]
            
            if historical_keys:
                # Calculate average key overlap
                overlaps = [
                    len(data_keys.intersection(hist_keys)) / len(data_keys.union(hist_keys))
                    for hist_keys in historical_keys
                ]
                avg_overlap = statistics.mean(overlaps)
                
                if avg_overlap < 0.5:  # Less than 50% overlap
                    anomalies.append(f"Unusual data structure (overlap: {avg_overlap:.2f})")
            
        except Exception as e:
            self.logger.warning(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def comprehensive_validate(self, 
                             data: Any, 
                             schema_name: Optional[str] = None,
                             enable_anomaly_detection: bool = True,
                             enable_sanitization: bool = True) -> ValidationResult:
        """Comprehensive validation with all features."""
        start_time = time.time()
        
        # Schema validation
        if schema_name:
            result = self.validate_by_schema(data, schema_name)
        else:
            # Basic validation without schema
            result = ValidationResult(
                is_valid=True,
                sanitized_data=data
            )
        
        # Anomaly detection
        anomalies = []
        if enable_anomaly_detection and isinstance(data, dict):
            anomalies = self.detect_anomalies(data)
            result.warnings.extend([f"Anomaly: {anomaly}" for anomaly in anomalies])
        
        # Additional sanitization
        if enable_sanitization and result.sanitized_data:
            result.sanitized_data = self._deep_sanitize(result.sanitized_data)
        
        # Calculate confidence score
        confidence = 1.0
        if result.errors:
            confidence -= len(result.errors) * 0.2
        if result.warnings:
            confidence -= len(result.warnings) * 0.05
        if anomalies:
            confidence -= len(anomalies) * 0.1
        
        result.confidence_score = max(0.0, min(1.0, confidence))
        
        # Record validation for learning
        if isinstance(data, dict):
            self._validation_history.append({
                **data,
                "_validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "_validation_success": result.is_valid,
                "_confidence_score": result.confidence_score
            })
            
            # Keep only recent history
            if len(self._validation_history) > 1000:
                self._validation_history = self._validation_history[-500:]
        
        # Add timing metadata
        result.metadata["validation_time_ms"] = (time.time() - start_time) * 1000
        result.metadata["anomalies_detected"] = len(anomalies)
        
        return result
    
    def _deep_sanitize(self, data: Any) -> Any:
        """Deep sanitization of nested data structures."""
        if isinstance(data, dict):
            return {
                key: self._deep_sanitize(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._deep_sanitize(item) for item in data]
        elif isinstance(data, str):
            return self.sanitize_string(data)
        else:
            return data
    
    def create_custom_validator(self, 
                               name: str,
                               validator_func: Callable[[Any], Tuple[bool, str]],
                               severity: ValidationSeverity = ValidationSeverity.ERROR) -> ValidationRule:
        """Create custom validation rule."""
        return ValidationRule(
            name=name,
            validator=validator_func,
            severity=severity
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self._validation_history:
            return {"message": "No validation history available"}
        
        successful = sum(1 for entry in self._validation_history if entry.get("_validation_success", False))
        total = len(self._validation_history)
        
        confidence_scores = [
            entry.get("_confidence_score", 0)
            for entry in self._validation_history
        ]
        
        return {
            "total_validations": total,
            "successful_validations": successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "registered_schemas": len(self._schemas),
            "anomaly_detection_threshold": self._anomaly_threshold,
            "recent_anomalies": len([
                entry for entry in self._validation_history[-100:]
                if "_anomaly_detected" in entry
            ])
        }


# Global validation engine instance
_validation_engine = None


def get_validation_engine() -> AdvancedValidationEngine:
    """Get or create global validation engine instance."""
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = AdvancedValidationEngine()
    return _validation_engine


def validate_input(data: Any, 
                  schema_name: Optional[str] = None,
                  **kwargs) -> ValidationResult:
    """Convenient function for input validation."""
    return get_validation_engine().comprehensive_validate(data, schema_name, **kwargs)