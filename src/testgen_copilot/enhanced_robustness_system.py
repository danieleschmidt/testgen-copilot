"""Enhanced Robustness System - Generation 2 Implementation

This module implements comprehensive robustness enhancements including:
- Advanced error handling with context preservation
- Comprehensive input validation and sanitization
- Distributed logging with structured telemetry
- Health monitoring with predictive alerts
- Security hardening with threat detection
- Graceful degradation and auto-recovery
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse

import psutil
import yaml
from cryptography.fernet import Fernet

from .logging_config import get_logger
from .monitoring import ApplicationMetrics, HealthMonitor
from .resilience import CircuitBreaker, RetryMechanism


F = TypeVar("F", bound=Callable[..., Any])


class SecurityThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemHealthStatus(Enum):
    """System health status indicators."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ThreatDetection:
    """Security threat detection result."""
    threat_id: str
    threat_type: str
    level: SecurityThreatLevel
    description: str
    source_ip: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mitigated: bool = False


@dataclass
class SystemHealth:
    """System health assessment."""
    status: SystemHealthStatus
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    error_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alerts: List[str] = field(default_factory=list)


class EnhancedRobustnessSystem:
    """Comprehensive robustness and reliability system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize robustness system with configuration."""
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.health_monitor = HealthMonitor()
        self.metrics = ApplicationMetrics()
        
        # Security components
        self._cipher_suite = self._init_encryption()
        self._threat_patterns = self._load_threat_patterns()
        self._blocked_ips: Set[str] = set()
        self._rate_limits: Dict[str, List[float]] = {}
        
        # System monitoring
        self._system_alerts: List[str] = []
        self._performance_baseline = self._establish_baseline()
        
        # Error tracking
        self._error_contexts: Dict[str, Dict[str, Any]] = {}
        self._recovery_strategies: Dict[str, Callable] = {}
        
        # Graceful shutdown
        self._shutdown_handlers: List[Callable] = []
        self._setup_signal_handlers()
        
        self.logger.info("Enhanced robustness system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load robustness configuration."""
        default_config = {
            "security": {
                "max_request_size": 10485760,  # 10MB
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
                "encryption_enabled": True,
                "threat_detection_enabled": True
            },
            "monitoring": {
                "health_check_interval": 30,
                "alert_thresholds": {
                    "cpu_usage": 80.0,
                    "memory_usage": 85.0,
                    "disk_usage": 90.0,
                    "error_rate": 5.0
                }
            },
            "resilience": {
                "circuit_breaker_enabled": True,
                "retry_enabled": True,
                "graceful_degradation": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _init_encryption(self) -> Optional[Fernet]:
        """Initialize encryption system."""
        if not self.config["security"]["encryption_enabled"]:
            return None
            
        try:
            key = os.environ.get("TESTGEN_ENCRYPTION_KEY")
            if not key:
                key = Fernet.generate_key()
                self.logger.warning("Using generated encryption key - set TESTGEN_ENCRYPTION_KEY for production")
            else:
                key = key.encode()
            
            return Fernet(key)
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            return None
    
    def _load_threat_patterns(self) -> Dict[str, re.Pattern]:
        """Load security threat detection patterns."""
        patterns = {
            "sql_injection": re.compile(
                r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|script|javascript|<script)",
                re.IGNORECASE
            ),
            "xss_attempt": re.compile(
                r"(?i)(<script|javascript:|vbscript:|onload=|onerror=|alert\()",
                re.IGNORECASE
            ),
            "path_traversal": re.compile(
                r"(\.\.\/|\.\.\\|\.\.\%2f|\.\.\%5c)",
                re.IGNORECASE
            ),
            "command_injection": re.compile(
                r"(?i)(;|&&|\|\||`|\$\(|\${)",
                re.IGNORECASE
            )
        }
        return patterns
    
    def _establish_baseline(self) -> Dict[str, float]:
        """Establish performance baseline metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_baseline": cpu_percent,
                "memory_baseline": memory.percent,
                "disk_baseline": disk.percent,
                "response_time_baseline": 0.1  # 100ms baseline
            }
        except Exception as e:
            self.logger.warning(f"Failed to establish baseline: {e}")
            return {
                "cpu_baseline": 20.0,
                "memory_baseline": 50.0,
                "disk_baseline": 30.0,
                "response_time_baseline": 0.1
            }
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def shutdown_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
    
    @contextmanager
    def enhanced_error_context(self, operation: str, **context):
        """Enhanced error context manager with automatic recovery."""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            self._error_contexts[operation_id] = {
                "operation": operation,
                "start_time": start_time,
                **context
            }
            
            self.logger.debug(f"Starting operation {operation} [{operation_id}]", extra=context)
            yield operation_id
            
        except Exception as e:
            duration = time.time() - start_time
            error_context = self._error_contexts.get(operation_id, {})
            
            self.logger.error(
                f"Operation {operation} failed after {duration:.2f}s: {str(e)}",
                extra={
                    "operation_id": operation_id,
                    "error_type": type(e).__name__,
                    "duration": duration,
                    "stack_trace": traceback.format_exc(),
                    **error_context
                },
                exc_info=True
            )
            
            # Attempt recovery if strategy exists
            if operation in self._recovery_strategies:
                try:
                    self.logger.info(f"Attempting recovery for {operation}")
                    self._recovery_strategies[operation](e, error_context)
                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed: {recovery_error}")
            
            raise
        
        finally:
            duration = time.time() - start_time
            self.metrics.record_operation(operation, duration)
            self._error_contexts.pop(operation_id, None)
    
    def robust_decorator(self, 
                        operation_name: str = None,
                        circuit_breaker: bool = True,
                        retry_attempts: int = 3,
                        timeout_seconds: float = 30.0) -> Callable[[F], F]:
        """Comprehensive robustness decorator."""
        def decorator(func: F) -> F:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Setup circuit breaker
            cb = None
            if circuit_breaker:
                cb = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60,
                    expected_exception=Exception
                )
            
            # Setup retry mechanism
            retry = RetryMechanism(max_attempts=retry_attempts)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.enhanced_error_context(op_name, args=args, kwargs=kwargs):
                    
                    # Apply circuit breaker
                    if cb:
                        cb.call(lambda: None)  # Test circuit state
                    
                    # Apply timeout
                    try:
                        return await asyncio.wait_for(
                            func(*args, **kwargs) if asyncio.iscoroutinefunction(func)
                            else asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs),
                            timeout=timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Operation {op_name} timed out after {timeout_seconds}s")
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.enhanced_error_context(op_name, args=args, kwargs=kwargs):
                    
                    # Apply circuit breaker
                    if cb:
                        return cb.call(lambda: func(*args, **kwargs))
                    
                    # Apply retry
                    return retry.execute(lambda: func(*args, **kwargs))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def validate_input(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive input validation."""
        errors = []
        
        try:
            # Type validation
            if "type" in schema:
                expected_type = schema["type"]
                if expected_type == "string" and not isinstance(data, str):
                    errors.append(f"Expected string, got {type(data).__name__}")
                elif expected_type == "integer" and not isinstance(data, int):
                    errors.append(f"Expected integer, got {type(data).__name__}")
                elif expected_type == "list" and not isinstance(data, list):
                    errors.append(f"Expected list, got {type(data).__name__}")
                elif expected_type == "dict" and not isinstance(data, dict):
                    errors.append(f"Expected dict, got {type(data).__name__}")
            
            # String validation
            if isinstance(data, str):
                if "max_length" in schema and len(data) > schema["max_length"]:
                    errors.append(f"String too long: {len(data)} > {schema['max_length']}")
                
                if "pattern" in schema:
                    pattern = re.compile(schema["pattern"])
                    if not pattern.match(data):
                        errors.append(f"String does not match required pattern")
                
                # Security validation
                if "no_html" in schema and schema["no_html"]:
                    if re.search(r'<[^>]*>', data):
                        errors.append("HTML tags not allowed")
                
                if "no_script" in schema and schema["no_script"]:
                    if re.search(r'(?i)<script|javascript:', data):
                        errors.append("Script content not allowed")
            
            # Numeric validation
            if isinstance(data, (int, float)):
                if "min_value" in schema and data < schema["min_value"]:
                    errors.append(f"Value too small: {data} < {schema['min_value']}")
                
                if "max_value" in schema and data > schema["max_value"]:
                    errors.append(f"Value too large: {data} > {schema['max_value']}")
            
            # List validation
            if isinstance(data, list):
                if "max_items" in schema and len(data) > schema["max_items"]:
                    errors.append(f"Too many items: {len(data)} > {schema['max_items']}")
                
                if "item_schema" in schema:
                    for i, item in enumerate(data):
                        item_valid, item_errors = self.validate_input(item, schema["item_schema"])
                        if not item_valid:
                            errors.extend([f"Item {i}: {error}" for error in item_errors])
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def sanitize_input(self, data: str) -> str:
        """Sanitize potentially dangerous input."""
        if not isinstance(data, str):
            return str(data)
        
        # Remove potential script content
        data = re.sub(r'<script[^>]*>.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
        data = re.sub(r'javascript:', '', data, flags=re.IGNORECASE)
        
        # Remove potential SQL injection patterns
        dangerous_patterns = [
            r'(?i)union\s+select',
            r'(?i)insert\s+into',
            r'(?i)delete\s+from',
            r'(?i)drop\s+table',
            r'(?i)exec\s*\(',
            r';\s*--',
            r'/\*.*?\*/'
        ]
        
        for pattern in dangerous_patterns:
            data = re.sub(pattern, '', data, flags=re.IGNORECASE)
        
        return data.strip()
    
    def detect_security_threats(self, 
                               payload: str, 
                               source_ip: Optional[str] = None) -> List[ThreatDetection]:
        """Detect security threats in input."""
        threats = []
        
        if not self.config["security"]["threat_detection_enabled"]:
            return threats
        
        for threat_type, pattern in self._threat_patterns.items():
            if pattern.search(payload):
                threat = ThreatDetection(
                    threat_id=str(uuid.uuid4()),
                    threat_type=threat_type,
                    level=SecurityThreatLevel.HIGH,
                    description=f"Detected {threat_type} pattern in payload",
                    source_ip=source_ip,
                    payload={"content": payload[:100]}  # First 100 chars only
                )
                threats.append(threat)
                
                self.logger.warning(
                    f"Security threat detected: {threat_type}",
                    extra={
                        "threat_id": threat.threat_id,
                        "source_ip": source_ip,
                        "threat_type": threat_type
                    }
                )
        
        return threats
    
    def check_rate_limits(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        window = self.config["security"]["rate_limit_window"]
        max_requests = self.config["security"]["rate_limit_requests"]
        
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []
        
        # Clean old requests outside window
        self._rate_limits[client_id] = [
            req_time for req_time in self._rate_limits[client_id]
            if current_time - req_time < window
        ]
        
        # Check if within limits
        if len(self._rate_limits[client_id]) >= max_requests:
            return False
        
        # Add current request
        self._rate_limits[client_id].append(current_time)
        return True
    
    def assess_system_health(self) -> SystemHealth:
        """Comprehensive system health assessment."""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network metrics (simplified)
            network_latency = self._measure_network_latency()
            
            # Connection metrics
            try:
                active_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                active_connections = 0
            
            # Error rate (from application metrics)
            error_rate = self.metrics.get_error_rate()
            
            # Determine overall status
            status = SystemHealthStatus.HEALTHY
            alerts = []
            
            thresholds = self.config["monitoring"]["alert_thresholds"]
            
            if cpu_usage > thresholds["cpu_usage"]:
                status = SystemHealthStatus.DEGRADED
                alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if memory_usage > thresholds["memory_usage"]:
                status = SystemHealthStatus.DEGRADED
                alerts.append(f"High memory usage: {memory_usage:.1f}%")
            
            if disk_usage > thresholds["disk_usage"]:
                status = SystemHealthStatus.UNHEALTHY
                alerts.append(f"High disk usage: {disk_usage:.1f}%")
            
            if error_rate > thresholds["error_rate"]:
                status = SystemHealthStatus.UNHEALTHY
                alerts.append(f"High error rate: {error_rate:.1f}%")
            
            # Critical status if multiple severe issues
            if len([a for a in alerts if "High disk usage" in a or "High error rate" in a]) > 1:
                status = SystemHealthStatus.CRITICAL
            
            return SystemHealth(
                status=status,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                active_connections=active_connections,
                error_rate=error_rate,
                alerts=alerts
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess system health: {e}")
            return SystemHealth(
                status=SystemHealthStatus.CRITICAL,
                cpu_usage=0,
                memory_usage=0,
                disk_usage=0,
                network_latency=0,
                active_connections=0,
                error_rate=100.0,
                alerts=[f"Health assessment failed: {str(e)}"]
            )
    
    def _measure_network_latency(self) -> float:
        """Measure basic network latency."""
        try:
            import socket
            start = time.time()
            socket.gethostbyname("google.com")
            return (time.time() - start) * 1000  # ms
        except Exception:
            return 0.0
    
    def encrypt_sensitive_data(self, data: str) -> Optional[str]:
        """Encrypt sensitive data."""
        if not self._cipher_suite:
            return None
        
        try:
            return self._cipher_suite.encrypt(data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data."""
        if not self._cipher_suite:
            return None
        
        try:
            return self._cipher_suite.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None
    
    def register_recovery_strategy(self, operation: str, strategy: Callable):
        """Register error recovery strategy for operation."""
        self._recovery_strategies[operation] = strategy
        self.logger.debug(f"Registered recovery strategy for {operation}")
    
    def register_shutdown_handler(self, handler: Callable):
        """Register graceful shutdown handler."""
        self._shutdown_handlers.append(handler)
    
    def shutdown(self):
        """Execute graceful shutdown."""
        self.logger.info("Initiating graceful shutdown")
        
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Shutdown handler error: {e}")
        
        self.logger.info("Graceful shutdown completed")


# Global instance
_robustness_system = None


def get_robustness_system(config_path: Optional[str] = None) -> EnhancedRobustnessSystem:
    """Get or create global robustness system instance."""
    global _robustness_system
    if _robustness_system is None:
        _robustness_system = EnhancedRobustnessSystem(config_path)
    return _robustness_system


def robust(operation_name: str = None, **kwargs):
    """Decorator for robust operations."""
    return get_robustness_system().robust_decorator(operation_name, **kwargs)


def validate_and_sanitize(data: Any, schema: Dict[str, Any]) -> Tuple[Any, bool, List[str]]:
    """Validate and sanitize input data."""
    robustness = get_robustness_system()
    
    # Sanitize if string
    if isinstance(data, str):
        data = robustness.sanitize_input(data)
    
    # Validate
    is_valid, errors = robustness.validate_input(data, schema)
    
    return data, is_valid, errors