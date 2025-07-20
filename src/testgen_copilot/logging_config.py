"""Centralized structured logging configuration for TestGen Copilot."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def __init__(self, use_json: bool = True):
        super().__init__()
        self.use_json = use_json
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        # Extract structured data from record
        structured_data = getattr(record, 'structured_data', {})
        
        # Build base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'line': record.lineno,
            'function': record.funcName
        }
        
        # Add structured data
        if structured_data:
            log_entry.update(structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add correlation context if available
        correlation_id = getattr(record, 'correlation_id', None)
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        operation_name = getattr(record, 'operation_name', None)
        if operation_name:
            log_entry['operation'] = operation_name
        
        if self.use_json:
            return json.dumps(log_entry, ensure_ascii=False)
        else:
            # Human-readable format with structured data
            base_msg = f"{log_entry['timestamp']} [{log_entry['level']}] {log_entry['module']}: {log_entry['message']}"
            
            if structured_data:
                structured_str = ' '.join(f"{k}={v}" for k, v in structured_data.items())
                base_msg += f" | {structured_str}"
            
            if correlation_id:
                base_msg += f" | correlation_id={correlation_id}"
                
            if operation_name:
                base_msg += f" | operation={operation_name}"
            
            return base_msg


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, use_json: bool = False):
        self.logger = logging.getLogger(name)
        self.use_json = use_json
        self._context_stack = []
    
    def _log_with_structure(
        self, 
        level: int, 
        message: str, 
        structured_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log message with structured data."""
        if structured_data is None:
            structured_data = {}
        
        # Create log record with structured data
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            __file__,
            0,
            message,
            (),
            None,
            func=kwargs.get('func', 'unknown'),
            extra={'structured_data': structured_data},
            sinfo=None
        )
        
        # Add context from stack
        if self._context_stack:
            current_context = self._context_stack[-1]
            record.correlation_id = current_context.get('correlation_id')
            record.operation_name = current_context.get('operation_name')
            # Merge context data
            context_data = {k: v for k, v in current_context.items() 
                          if k not in ['correlation_id', 'operation_name']}
            structured_data.update(context_data)
            record.structured_data = structured_data
        
        self.logger.handle(record)
    
    def debug(self, message: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with structured data."""
        self._log_with_structure(logging.DEBUG, message, structured_data, **kwargs)
    
    def info(self, message: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with structured data."""
        self._log_with_structure(logging.INFO, message, structured_data, **kwargs)
    
    def warning(self, message: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with structured data."""
        self._log_with_structure(logging.WARNING, message, structured_data, **kwargs)
    
    def error(self, message: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message with structured data."""
        self._log_with_structure(logging.ERROR, message, structured_data, **kwargs)
    
    def critical(self, message: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message with structured data."""
        self._log_with_structure(logging.CRITICAL, message, structured_data, **kwargs)
    
    def audit(self, message: str, structured_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Log audit message with structured data."""
        if structured_data is None:
            structured_data = {}
        structured_data['log_type'] = 'audit'
        self._log_with_structure(logging.INFO, message, structured_data, **kwargs)
    
    def metrics(self, event_name: str, metrics_data: Dict[str, Any], **kwargs):
        """Log metrics with structured data."""
        structured_data = {
            'log_type': 'metrics',
            'event_name': event_name,
            **metrics_data
        }
        self._log_with_structure(logging.INFO, f"Metrics: {event_name}", structured_data, **kwargs)
    
    @contextmanager
    def time_operation(self, operation_name: str, additional_context: Optional[Dict[str, Any]] = None):
        """Context manager to time operations and log performance metrics."""
        start_time = time.time()
        context_data = {'operation': operation_name}
        if additional_context:
            context_data.update(additional_context)
        
        self.debug(f"Starting operation: {operation_name}", context_data)
        
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            self.info(
                f"Completed operation: {operation_name}",
                {
                    'operation': operation_name,
                    'duration_ms': round(duration_ms, 2),
                    'log_type': 'performance',
                    **context_data
                }
            )


class LogContext:
    """Context manager for maintaining operation context across log calls."""
    
    def __init__(
        self, 
        logger: StructuredLogger, 
        operation_name: str, 
        context_data: Optional[Dict[str, Any]] = None
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.context_data = context_data or {}
        self.correlation_id = self.context_data.get('correlation_id', str(uuid.uuid4()))
    
    def __enter__(self):
        """Enter context and push to logger's context stack."""
        context = {
            'operation_name': self.operation_name,
            'correlation_id': self.correlation_id,
            **self.context_data
        }
        self.logger._context_stack.append(context)
        
        self.logger.info(
            f"Starting operation: {self.operation_name}",
            {
                'operation': self.operation_name,
                'correlation_id': self.correlation_id,
                'log_type': 'operation_start'
            }
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and pop from logger's context stack."""
        try:
            if exc_type is not None:
                self.logger.error(
                    f"Operation failed: {self.operation_name}",
                    {
                        'operation': self.operation_name,
                        'correlation_id': self.correlation_id,
                        'error_type': exc_type.__name__ if exc_type else None,
                        'error_message': str(exc_val) if exc_val else None,
                        'log_type': 'operation_error'
                    },
                    exc_info=True
                )
            else:
                self.logger.info(
                    f"Completed operation: {self.operation_name}",
                    {
                        'operation': self.operation_name,
                        'correlation_id': self.correlation_id,
                        'log_type': 'operation_complete'
                    }
                )
        finally:
            if self.logger._context_stack:
                self.logger._context_stack.pop()


def configure_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    max_log_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """Configure centralized logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json', 'structured', or 'simple')
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
        max_log_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
    """
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create formatter
    use_json = format_type.lower() == 'json'
    formatter = StructuredFormatter(use_json=use_json)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_log_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, use_json: bool = False) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        use_json: Whether to use JSON formatting
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, use_json=use_json)


# Global logger instances for common modules
def get_generator_logger() -> StructuredLogger:
    """Get logger for test generator operations."""
    return get_logger('testgen_copilot.generator')


def get_security_logger() -> StructuredLogger:
    """Get logger for security scanning operations."""
    return get_logger('testgen_copilot.security')


def get_coverage_logger() -> StructuredLogger:
    """Get logger for coverage analysis operations."""
    return get_logger('testgen_copilot.coverage')


def get_quality_logger() -> StructuredLogger:
    """Get logger for quality scoring operations."""
    return get_logger('testgen_copilot.quality')


def get_cli_logger() -> StructuredLogger:
    """Get logger for CLI operations."""
    return get_logger('testgen_copilot.cli')