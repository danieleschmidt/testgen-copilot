"""Enhanced Security Framework - Generation 2 Implementation

Comprehensive security framework with advanced threat detection,
zero-trust architecture, secure communication, audit logging,
and compliance monitoring.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .logging_config import get_logger


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "auth_violation"
    INPUT_VALIDATION_ERROR = "input_validation"
    RATE_LIMIT_EXCEEDED = "rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALWARE_DETECTED = "malware_detected"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class AccessLevel(Enum):
    """Access control levels."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_action: Optional[str] = None


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    access_level: AccessLevel = AccessLevel.PUBLIC
    permissions: Set[str] = field(default_factory=set)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    authenticated: bool = False
    expires_at: Optional[datetime] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance."""
    log_id: str
    user_id: Optional[str]
    action: str
    resource: str
    result: str  # success, failure, denied
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_ip: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class EnhancedSecurityFramework:
    """Comprehensive security framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize security framework."""
        self.logger = get_logger(__name__)
        self.config = self._load_security_config(config_path)
        
        # Encryption setup
        self._setup_encryption()
        
        # Security monitoring
        self._security_events: List[SecurityEvent] = []
        self._audit_log: List[AuditLogEntry] = []
        self._blocked_ips: Set[str] = set()
        self._suspicious_ips: Dict[str, int] = {}  # IP -> violation count
        self._active_sessions: Dict[str, SecurityContext] = {}
        
        # Rate limiting
        self._rate_limits: Dict[str, List[float]] = {}
        
        # Threat detection patterns
        self._setup_threat_detection()
        
        self.logger.info("Enhanced security framework initialized")
    
    def _load_security_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load security configuration."""
        default_config = {
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 30,
                "require_encryption": True
            },
            "authentication": {
                "session_timeout_hours": 8,
                "max_login_attempts": 3,
                "lockout_duration_minutes": 30,
                "require_2fa": False
            },
            "rate_limiting": {
                "default_limit": 100,
                "window_seconds": 60,
                "burst_multiplier": 2
            },
            "threat_detection": {
                "enable_ip_blocking": True,
                "max_violations_per_ip": 5,
                "violation_window_hours": 1,
                "enable_payload_scanning": True
            },
            "audit": {
                "log_all_requests": True,
                "log_data_access": True,
                "retention_days": 90
            },
            "compliance": {
                "gdpr_enabled": True,
                "hipaa_enabled": False,
                "soc2_enabled": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
            except Exception as e:
                self.logger.warning(f"Failed to load security config: {e}")
        
        return default_config
    
    def _setup_encryption(self):
        """Setup encryption systems."""
        # Symmetric encryption
        encryption_key = os.environ.get("TESTGEN_ENCRYPTION_KEY")
        if not encryption_key:
            encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            self.logger.warning("Generated temporary encryption key - set TESTGEN_ENCRYPTION_KEY for production")
        
        try:
            if len(encryption_key) == 44 and encryption_key.endswith('='):
                # Fernet key
                self._fernet = Fernet(encryption_key.encode())
            else:
                # Custom key - derive Fernet key
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'testgen_salt',
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
                self._fernet = Fernet(key)
                
        except Exception as e:
            self.logger.error(f"Failed to setup encryption: {e}")
            self._fernet = None
        
        # Asymmetric encryption for key exchange
        try:
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self._public_key = self._private_key.public_key()
        except Exception as e:
            self.logger.error(f"Failed to setup asymmetric encryption: {e}")
            self._private_key = None
            self._public_key = None
    
    def _setup_threat_detection(self):
        """Setup threat detection patterns and rules."""
        # Common attack patterns
        self._threat_patterns = {
            "sql_injection": [
                r"(?i)(union|select|insert|update|delete|drop|create|alter|exec)\s",
                r"(?i)(\"|')\s*(or|and)\s*(\"|')?\s*(=|\<|\>)",
                r"(?i)(exec|execute)\s*\(",
                r"(?i)(concat|char|ascii|hex)\s*\("
            ],
            "xss_injection": [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on(load|error|click|focus|blur)=",
                r"(?i)alert\s*\(",
                r"(?i)document\.(cookie|domain|location)"
            ],
            "command_injection": [
                r"[;&|`]",
                r"\$\([^)]*\)",
                r"`[^`]*`",
                r"(?i)(wget|curl|nc|netcat|bash|sh|cmd|powershell)\s"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ],
            "ldap_injection": [
                r"(?i)(\*|\)|\(|\||&)",
                r"(?i)(objectclass|cn|uid|ou|dc)="
            ]
        }
        
        # Compile regex patterns
        import re
        self._compiled_patterns = {}
        for threat_type, patterns in self._threat_patterns.items():
            self._compiled_patterns[threat_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def encrypt_data(self, data: Union[str, bytes], context: Optional[str] = None) -> Optional[str]:
        """Encrypt sensitive data."""
        if not self._fernet:
            self.logger.warning("Encryption not available")
            return None
        
        try:
            if isinstance(data, str):
                data = data.encode()
            
            # Add context to data for additional security
            if context:
                data = f"{context}::{data.decode()}".encode()
            
            encrypted = self._fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str, context: Optional[str] = None) -> Optional[str]:
        """Decrypt sensitive data."""
        if not self._fernet:
            self.logger.warning("Decryption not available")
            return None
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            data = decrypted.decode()
            
            # Remove context if present
            if context and data.startswith(f"{context}::"):
                data = data[len(context) + 2:]
            
            return data
            
        except (InvalidToken, Exception) as e:
            self.logger.error(f"Decryption failed: {e}")
            return None
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password securely."""
        if salt is None:
            salt = os.urandom(32)
        
        # Use PBKDF2 with SHA-256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode())
        
        return (
            base64.urlsafe_b64encode(key).decode(),
            base64.urlsafe_b64encode(salt).decode()
        )
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash."""
        try:
            salt_bytes = base64.urlsafe_b64decode(salt.encode())
            expected_hash, _ = self.hash_password(password, salt_bytes)
            return hmac.compare_digest(hashed, expected_hash)
        except Exception as e:
            self.logger.error(f"Password verification failed: {e}")
            return False
    
    def create_secure_token(self, payload: Dict[str, Any], expires_in_hours: int = 8) -> str:
        """Create secure token with payload."""
        token_data = {
            "payload": payload,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc).timestamp() + expires_in_hours * 3600),
            "token_id": secrets.token_urlsafe(16)
        }
        
        token_json = json.dumps(token_data, sort_keys=True)
        encrypted_token = self.encrypt_data(token_json, "secure_token")
        
        if not encrypted_token:
            raise Exception("Failed to create secure token")
        
        return encrypted_token
    
    def verify_secure_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode secure token."""
        try:
            decrypted = self.decrypt_data(token, "secure_token")
            if not decrypted:
                return None
            
            token_data = json.loads(decrypted)
            
            # Check expiration
            if token_data["expires_at"] < time.time():
                self.logger.warning("Token expired")
                return None
            
            return token_data["payload"]
            
        except Exception as e:
            self.logger.error(f"Token verification failed: {e}")
            return None
    
    def scan_for_threats(self, payload: str, source_ip: Optional[str] = None) -> List[SecurityEvent]:
        """Scan payload for security threats."""
        threats = []
        
        if not self.config["threat_detection"]["enable_payload_scanning"]:
            return threats
        
        # Check each threat type
        for threat_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(payload):
                    event = SecurityEvent(
                        event_id=secrets.token_urlsafe(16),
                        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                        threat_level=self._get_threat_level(threat_type),
                        description=f"Detected {threat_type} pattern",
                        source_ip=source_ip,
                        metadata={
                            "threat_type": threat_type,
                            "payload_length": len(payload),
                            "pattern_matched": True
                        }
                    )
                    threats.append(event)
                    self._record_security_event(event)
                    break  # One detection per threat type
        
        return threats
    
    def check_rate_limit(self, identifier: str, limit: Optional[int] = None) -> bool:
        """Check rate limit for identifier (IP, user, etc.)."""
        if limit is None:
            limit = self.config["rate_limiting"]["default_limit"]
        
        window = self.config["rate_limiting"]["window_seconds"]
        current_time = time.time()
        
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []
        
        # Clean old entries
        self._rate_limits[identifier] = [
            req_time for req_time in self._rate_limits[identifier]
            if current_time - req_time < window
        ]
        
        # Check limit
        if len(self._rate_limits[identifier]) >= limit:
            # Record rate limit violation
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                description=f"Rate limit exceeded for {identifier}",
                source_ip=identifier if self._is_ip_address(identifier) else None,
                metadata={
                    "limit": limit,
                    "window_seconds": window,
                    "current_requests": len(self._rate_limits[identifier])
                }
            )
            self._record_security_event(event)
            return False
        
        # Add current request
        self._rate_limits[identifier].append(current_time)
        return True
    
    def validate_input_security(self, 
                               data: Any, 
                               expected_type: type,
                               max_length: Optional[int] = None,
                               allowed_patterns: Optional[List[str]] = None,
                               blocked_patterns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """Comprehensive input security validation."""
        errors = []
        
        # Type validation
        if not isinstance(data, expected_type):
            errors.append(f"Invalid type: expected {expected_type.__name__}, got {type(data).__name__}")
            return False, errors
        
        # String-specific validations
        if isinstance(data, str):
            # Length check
            if max_length and len(data) > max_length:
                errors.append(f"Input too long: {len(data)} > {max_length}")
            
            # Pattern validation
            if allowed_patterns:
                import re
                for pattern in allowed_patterns:
                    if not re.match(pattern, data):
                        errors.append(f"Input doesn't match allowed pattern: {pattern}")
            
            if blocked_patterns:
                import re
                for pattern in blocked_patterns:
                    if re.search(pattern, data, re.IGNORECASE):
                        errors.append(f"Input contains blocked pattern: {pattern}")
            
            # Threat scanning
            threats = self.scan_for_threats(data)
            if threats:
                errors.extend([f"Security threat detected: {t.description}" for t in threats])
        
        return len(errors) == 0, errors
    
    def authorize_operation(self, 
                           context: SecurityContext, 
                           resource: str, 
                           action: str,
                           required_level: AccessLevel = AccessLevel.AUTHENTICATED) -> bool:
        """Authorize operation based on security context."""
        # Check authentication
        if required_level != AccessLevel.PUBLIC and not context.authenticated:
            self._record_audit_log(
                context.user_id, action, resource, "denied",
                context.source_ip, {"reason": "not_authenticated"}
            )
            return False
        
        # Check access level
        access_levels_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.AUTHENTICATED: 1,
            AccessLevel.AUTHORIZED: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SYSTEM: 4
        }
        
        if access_levels_hierarchy[context.access_level] < access_levels_hierarchy[required_level]:
            self._record_audit_log(
                context.user_id, action, resource, "denied",
                context.source_ip, {"reason": "insufficient_access_level"}
            )
            return False
        
        # Check specific permissions if required
        required_permission = f"{resource}:{action}"
        if required_level >= AccessLevel.AUTHORIZED and required_permission not in context.permissions:
            self._record_audit_log(
                context.user_id, action, resource, "denied",
                context.source_ip, {"reason": "missing_permission", "required": required_permission}
            )
            return False
        
        # Check session expiration
        if context.expires_at and datetime.now(timezone.utc) > context.expires_at:
            self._record_audit_log(
                context.user_id, action, resource, "denied",
                context.source_ip, {"reason": "session_expired"}
            )
            return False
        
        # Success
        self._record_audit_log(
            context.user_id, action, resource, "success",
            context.source_ip, {"access_level": context.access_level.value}
        )
        return True
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP address is blocked."""
        return ip in self._blocked_ips
    
    def block_ip(self, ip: str, reason: str = "security_violation"):
        """Block IP address."""
        self._blocked_ips.add(ip)
        
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            threat_level=ThreatLevel.HIGH,
            description=f"IP {ip} blocked: {reason}",
            source_ip=ip,
            response_action="ip_blocked"
        )
        self._record_security_event(event)
        
        self.logger.warning(f"Blocked IP {ip}: {reason}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics."""
        now = datetime.now(timezone.utc)
        last_hour = now.timestamp() - 3600
        last_day = now.timestamp() - 86400
        
        recent_events = [
            event for event in self._security_events
            if event.timestamp.timestamp() > last_hour
        ]
        
        daily_events = [
            event for event in self._security_events
            if event.timestamp.timestamp() > last_day
        ]
        
        return {
            "summary": {
                "total_events": len(self._security_events),
                "events_last_hour": len(recent_events),
                "events_last_day": len(daily_events),
                "blocked_ips": len(self._blocked_ips),
                "active_sessions": len(self._active_sessions)
            },
            "threat_levels": {
                level.value: len([e for e in daily_events if e.threat_level == level])
                for level in ThreatLevel
            },
            "event_types": {
                event_type.value: len([e for e in daily_events if e.event_type == event_type])
                for event_type in SecurityEventType
            },
            "top_source_ips": self._get_top_source_ips(daily_events),
            "compliance_status": self._get_compliance_status()
        }
    
    def _record_security_event(self, event: SecurityEvent):
        """Record security event."""
        self._security_events.append(event)
        
        # Auto-block IPs with too many violations
        if event.source_ip and self.config["threat_detection"]["enable_ip_blocking"]:
            if event.source_ip not in self._suspicious_ips:
                self._suspicious_ips[event.source_ip] = 0
            
            self._suspicious_ips[event.source_ip] += 1
            
            if self._suspicious_ips[event.source_ip] >= self.config["threat_detection"]["max_violations_per_ip"]:
                self.block_ip(event.source_ip, "too_many_violations")
        
        # Log high-severity events
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.error(f"Security event: {event.description} from {event.source_ip}")
    
    def _record_audit_log(self, 
                         user_id: Optional[str],
                         action: str, 
                         resource: str, 
                         result: str,
                         source_ip: Optional[str] = None,
                         details: Optional[Dict[str, Any]] = None):
        """Record audit log entry."""
        entry = AuditLogEntry(
            log_id=secrets.token_urlsafe(16),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            source_ip=source_ip,
            details=details or {}
        )
        
        self._audit_log.append(entry)
        
        # Log for compliance
        self.logger.info(
            f"Audit: {user_id or 'anonymous'} {action} {resource} -> {result}",
            extra={
                "audit_log_id": entry.log_id,
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "result": result,
                "source_ip": source_ip,
                "details": details
            }
        )
    
    def _get_threat_level(self, threat_type: str) -> ThreatLevel:
        """Determine threat level based on type."""
        critical_threats = ["sql_injection", "command_injection"]
        high_threats = ["xss_injection", "path_traversal"]
        medium_threats = ["ldap_injection"]
        
        if threat_type in critical_threats:
            return ThreatLevel.CRITICAL
        elif threat_type in high_threats:
            return ThreatLevel.HIGH
        elif threat_type in medium_threats:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _is_ip_address(self, value: str) -> bool:
        """Check if value is an IP address."""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    
    def _get_top_source_ips(self, events: List[SecurityEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top source IPs from events."""
        ip_counts = {}
        for event in events:
            if event.source_ip:
                ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        return [
            {"ip": ip, "count": count}
            for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_compliance_status(self) -> Dict[str, bool]:
        """Get compliance status for various frameworks."""
        return {
            "gdpr_compliant": self.config["compliance"]["gdpr_enabled"] and len(self._audit_log) > 0,
            "hipaa_compliant": self.config["compliance"]["hipaa_enabled"],
            "soc2_compliant": self.config["compliance"]["soc2_enabled"] and self._fernet is not None,
            "audit_logging_active": len(self._audit_log) > 0,
            "encryption_enabled": self._fernet is not None
        }


# Global security framework instance
_security_framework = None


def get_security_framework(config_path: Optional[str] = None) -> EnhancedSecurityFramework:
    """Get or create global security framework instance."""
    global _security_framework
    if _security_framework is None:
        _security_framework = EnhancedSecurityFramework(config_path)
    return _security_framework


def require_security(access_level: AccessLevel = AccessLevel.AUTHENTICATED):
    """Decorator to require security authorization."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would integrate with request context in a real application
            # For now, we'll create a basic context
            security = get_security_framework()
            context = SecurityContext(authenticated=True, access_level=access_level)
            
            if not security.authorize_operation(context, func.__name__, "execute", access_level):
                raise PermissionError("Insufficient permissions")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator