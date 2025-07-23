"""Resource limits and validation for TestGen Copilot operations."""

from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

try:
    import psutil
except ImportError:
    psutil = None

from .logging_config import get_generator_logger


@dataclass
class ResourceLimits:
    """Configuration for resource limits and validation."""
    
    max_file_size_mb: int = 10
    ast_timeout_seconds: int = 10
    max_batch_size: int = 1000
    memory_threshold_percent: float = 90.0
    
    @classmethod
    def from_environment(cls) -> ResourceLimits:
        """Create ResourceLimits from environment variables."""
        return cls(
            max_file_size_mb=int(os.getenv('TESTGEN_MAX_FILE_SIZE_MB', 10)),
            ast_timeout_seconds=int(os.getenv('TESTGEN_AST_TIMEOUT_SECONDS', 10)),
            max_batch_size=int(os.getenv('TESTGEN_MAX_BATCH_SIZE', 1000)),
            memory_threshold_percent=float(os.getenv('TESTGEN_MEMORY_THRESHOLD_PERCENT', 90.0))
        )


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits.from_environment()
        self.logger = get_generator_logger()
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        if psutil is None:
            # If psutil not available, assume memory is OK
            return True
            
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.limits.memory_threshold_percent:
                self.logger.warning("High memory usage detected", {
                    "memory_percent": memory.percent,
                    "threshold": self.limits.memory_threshold_percent,
                    "available_mb": memory.available // (1024 * 1024)
                })
                return False
            return True
        except Exception as e:
            self.logger.warning("Failed to check memory usage", {
                "error_message": str(e),
                "fallback": "assuming memory OK"
            })
            return True
    
    def safe_ast_parse(self, content: str, file_path: Optional[Path] = None) -> ast.AST:
        """Parse AST with timeout protection."""
        start_time = time.time()
        
        def check_timeout():
            if time.time() - start_time > self.limits.ast_timeout_seconds:
                raise TimeoutError(f"AST parsing exceeded {self.limits.ast_timeout_seconds} seconds")
        
        try:
            # Check timeout before starting
            check_timeout()
            
            # Parse with periodic timeout checks for large files
            tree = ast.parse(content)
            
            # Check timeout after parsing
            check_timeout()
            
            self.logger.debug("AST parsing completed", {
                "file_path": str(file_path) if file_path else "unknown",
                "parse_time_seconds": time.time() - start_time,
                "content_length": len(content),
                "ast_nodes": len(list(ast.walk(tree)))
            })
            
            return tree
            
        except TimeoutError:
            self.logger.error("AST parsing timeout", {
                "file_path": str(file_path) if file_path else "unknown",
                "timeout_seconds": self.limits.ast_timeout_seconds,
                "content_length": len(content)
            })
            raise
        except SyntaxError as e:
            self.logger.error("AST parsing syntax error", {
                "file_path": str(file_path) if file_path else "unknown",
                "line_number": e.lineno,
                "error_message": e.msg
            })
            raise
        except Exception as e:
            self.logger.error("AST parsing failed", {
                "file_path": str(file_path) if file_path else "unknown",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise
    
    def validate_batch_size(self, file_count: int) -> None:
        """Validate that batch size is within limits."""
        if file_count > self.limits.max_batch_size:
            self.logger.error("Batch size exceeds limit", {
                "file_count": file_count,
                "max_batch_size": self.limits.max_batch_size
            })
            raise ValueError(f"Batch size {file_count} exceeds limit of {self.limits.max_batch_size}")
    
    def validate_test_content(self, content: str, file_path: Optional[Path] = None) -> None:
        """Validate generated test content before writing to disk."""
        if not content.strip():
            raise ValueError("Generated test content is empty")
        
        # Check for basic Python syntax validity
        try:
            ast.parse(content)
        except SyntaxError as e:
            self.logger.error("Generated test content has invalid syntax", {
                "file_path": str(file_path) if file_path else "unknown",
                "line_number": e.lineno,
                "error_message": e.msg,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            })
            raise ValueError(f"Generated test content has invalid Python syntax: {e.msg}")
        
        # Check for potential security issues in generated content
        if any(dangerous in content for dangerous in ['eval(', 'exec(', '__import__']):
            self.logger.error("Generated test content contains dangerous patterns", {
                "file_path": str(file_path) if file_path else "unknown",
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            })
            raise ValueError("Generated test content contains potentially dangerous code")
        
        self.logger.debug("Test content validation passed", {
            "file_path": str(file_path) if file_path else "unknown",
            "content_length": len(content),
            "line_count": content.count('\n') + 1
        })


class CircuitBreaker:
    """Circuit breaker pattern for resource protection."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self.logger = get_generator_logger()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "half-open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise RuntimeError("Circuit breaker is open - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("Circuit breaker reset to closed")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.error("Circuit breaker opened due to failures", {
                    "failure_count": self.failure_count,
                    "threshold": self.failure_threshold,
                    "timeout_seconds": self.timeout_seconds
                })
            
            raise