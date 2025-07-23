"""Resource limits and validation for TestGen Copilot."""

import ast
import os
import re
import signal
import sys
import resource
import threading
from pathlib import Path
from typing import List, Optional, Union
import time

from .logging_config import get_generator_logger

# Try to import psutil for better memory monitoring, fall back to resource module
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# Configuration constants
AST_PARSE_TIMEOUT = 30  # seconds
MAX_PROJECT_FILES = 1000  # maximum files to process in a single batch
MAX_MEMORY_MB = 512  # maximum memory usage in MB
MIN_MEMORY_AVAILABLE_MB = 100  # minimum available memory required


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""
    pass


class ResourceMemoryError(Exception):
    """Raised when memory usage exceeds limits."""
    pass


class TimeoutHandler:
    """Context manager for handling operation timeouts."""
    
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.old_handler = None
        
    def __enter__(self):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
        
        self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel the alarm
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)


class CrossPlatformTimeoutHandler:
    """Cross-platform context manager for handling operation timeouts.
    
    Uses Unix signals when available (Linux/macOS), falls back to threading
    for Windows and other platforms where signals are not available.
    """
    
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.old_handler = None
        self.timer = None
        self.timed_out = False
        self.use_signals = hasattr(signal, 'SIGALRM')
        
    def __enter__(self):
        if self.use_signals:
            # Use signal-based timeout (Unix/Linux/macOS)
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
            
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
        else:
            # Use thread-based timeout (Windows and other platforms)
            def timeout_callback():
                self.timed_out = True
            
            self.timer = threading.Timer(self.timeout_seconds, timeout_callback)
            self.timer.start()
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_signals:
            # Clean up signal-based timeout
            signal.alarm(0)  # Cancel the alarm
            if self.old_handler is not None:
                signal.signal(signal.SIGALRM, self.old_handler)
        else:
            # Clean up thread-based timeout
            if self.timer is not None:
                self.timer.cancel()
            
            # Check if we timed out during the operation
            if self.timed_out:
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
                
    def check_timeout(self):
        """Check if timeout has occurred (for thread-based timeout only)."""
        if not self.use_signals and self.timed_out:
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")


class MemoryMonitor:
    """Monitor memory usage and implement circuit breaker patterns."""
    
    def __init__(self, max_memory_mb: int = MAX_MEMORY_MB):
        self.max_memory_mb = max_memory_mb
        self.logger = get_generator_logger()
        
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            if HAS_PSUTIL:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB
                return round(memory_mb, 2)
            else:
                # Fall back to resource module (Unix-only)
                if hasattr(resource, 'RUSAGE_SELF'):
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
                    if sys.platform == 'darwin':
                        memory_mb = usage.ru_maxrss / 1024 / 1024
                    else:
                        memory_mb = usage.ru_maxrss / 1024
                    return round(memory_mb, 2)
                else:
                    return 0.0  # Unable to get memory info on this platform
        except Exception as e:
            self.logger.warning("Failed to get memory usage", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "has_psutil": HAS_PSUTIL
            })
            return 0.0
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            if HAS_PSUTIL:
                memory = psutil.virtual_memory()
                available_mb = memory.available / 1024 / 1024
                return round(available_mb, 2)
            else:
                # Fall back to reading /proc/meminfo on Linux
                if os.path.exists('/proc/meminfo'):
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemAvailable:'):
                                available_kb = int(line.split()[1])
                                return round(available_kb / 1024, 2)
                    # If MemAvailable not found, try MemFree
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemFree:'):
                                free_kb = int(line.split()[1])
                                return round(free_kb / 1024, 2)
                
                # Conservative fallback
                return 1000.0
        except Exception as e:
            self.logger.warning("Failed to get available memory", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "has_psutil": HAS_PSUTIL
            })
            return 1000.0  # Conservative fallback
    
    def is_memory_exceeded(self) -> bool:
        """Check if current memory usage exceeds limits."""
        current_memory = self.get_current_memory_mb()
        available_memory = self.get_available_memory_mb()
        
        # Check if we're using too much memory
        if current_memory > self.max_memory_mb:
            self.logger.warning("Memory usage exceeded limit", {
                "current_memory_mb": current_memory,
                "max_memory_mb": self.max_memory_mb,
                "available_memory_mb": available_memory
            })
            return True
        
        # Check if system is running low on memory
        if available_memory < MIN_MEMORY_AVAILABLE_MB:
            self.logger.warning("System memory running low", {
                "available_memory_mb": available_memory,
                "min_required_mb": MIN_MEMORY_AVAILABLE_MB,
                "current_usage_mb": current_memory
            })
            return True
            
        return False
    
    def check_memory_and_raise(self):
        """Check memory usage and raise MemoryError if exceeded."""
        if self.is_memory_exceeded():
            current_memory = self.get_current_memory_mb()
            available_memory = self.get_available_memory_mb()
            
            error_msg = f"Memory limit exceeded: {current_memory}MB used (max: {self.max_memory_mb}MB), {available_memory}MB available"
            
            self.logger.error("Memory limit exceeded", {
                "current_memory_mb": current_memory,
                "max_memory_mb": self.max_memory_mb,
                "available_memory_mb": available_memory,
                "error_type": "memory_limit_exceeded"
            })
            
            raise ResourceMemoryError(error_msg)


class BatchProcessor:
    """Process files in batches with resource limits."""
    
    def __init__(self, max_files: int = MAX_PROJECT_FILES):
        self.max_files = max_files
        self.logger = get_generator_logger()
        self.memory_monitor = MemoryMonitor()
    
    def process_files(self, file_paths: List[Path]) -> List[Path]:
        """Process files in batches respecting resource limits."""
        # Limit total files to process
        limited_files = file_paths[:self.max_files]
        
        self.logger.info("Starting batch file processing", {
            "total_files_requested": len(file_paths),
            "files_to_process": len(limited_files),
            "max_files_limit": self.max_files,
            "files_skipped": len(file_paths) - len(limited_files)
        })
        
        processed_files = []
        batch_size = min(100, self.max_files)  # Process in smaller batches of 100
        
        # Process files in smaller batches
        for i in range(0, len(limited_files), batch_size):
            batch = limited_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.logger.debug("Processing file batch", {
                "batch_number": batch_num,
                "batch_size": len(batch),
                "start_index": i,
                "end_index": min(i + batch_size, len(limited_files))
            })
            
            # Check memory before processing batch
            self.memory_monitor.check_memory_and_raise()
            
            # Process this batch
            for file_path in batch:
                try:
                    # Check memory for each file to catch memory leaks early
                    if len(processed_files) % 100 == 0:  # Check every 100 files
                        self.memory_monitor.check_memory_and_raise()
                    
                    # Add file to processed list (actual processing would happen here)
                    processed_files.append(file_path)
                    
                except ResourceMemoryError:
                    self.logger.error("Memory limit reached during batch processing", {
                        "files_processed": len(processed_files),
                        "current_batch": batch_num,
                        "file_path": str(file_path)
                    })
                    break  # Stop processing this batch
                except Exception as e:
                    self.logger.error("Error processing file in batch", {
                        "file_path": str(file_path),
                        "batch_number": batch_num,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    continue  # Skip this file, continue with others
        
        self.logger.info("Batch processing completed", {
            "total_files_requested": len(file_paths),
            "files_processed": len(processed_files),
            "processing_rate": len(processed_files) / len(file_paths) if file_paths else 0
        })
        
        return processed_files


def safe_parse_ast_with_timeout(
    content: str, 
    filename: str, 
    timeout_seconds: int = AST_PARSE_TIMEOUT
) -> ast.AST:
    """Safely parse AST with timeout protection."""
    logger = get_generator_logger()
    
    logger.debug("Starting AST parsing with timeout", {
        "filename": filename,
        "content_length": len(content),
        "timeout_seconds": timeout_seconds
    })
    
    try:
        # Only use timeout on Unix-like systems (signal.SIGALRM not available on Windows)
        if hasattr(signal, 'SIGALRM'):
            with TimeoutHandler(timeout_seconds):
                start_time = time.time()
                tree = ast.parse(content, filename=filename)
                parse_time = time.time() - start_time
                
                logger.debug("AST parsing completed successfully", {
                    "filename": filename,
                    "parse_time_ms": round(parse_time * 1000, 2),
                    "ast_node_count": len(list(ast.walk(tree)))
                })
                
                return tree
        else:
            # Fallback for Windows - no timeout protection
            logger.debug("Timeout protection not available on this platform", {
                "platform": sys.platform,
                "filename": filename
            })
            start_time = time.time()
            tree = ast.parse(content, filename=filename)
            parse_time = time.time() - start_time
            
            logger.debug("AST parsing completed (no timeout)", {
                "filename": filename,
                "parse_time_ms": round(parse_time * 1000, 2)
            })
            
            return tree
            
    except TimeoutError as e:
        logger.error("AST parsing timed out", {
            "filename": filename,
            "timeout_seconds": timeout_seconds,
            "content_length": len(content),
            "error_type": "parse_timeout"
        })
        raise
        
    except SyntaxError as e:
        logger.error("Syntax error during AST parsing", {
            "filename": filename,
            "line_number": e.lineno,
            "error_message": e.msg,
            "error_type": "syntax_error"
        })
        raise
        
    except Exception as e:
        logger.error("Unexpected error during AST parsing", {
            "filename": filename,
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        raise


def validate_test_content(content: str) -> bool:
    """Validate generated test content for safety and correctness."""
    logger = get_generator_logger()
    
    logger.debug("Validating test content", {
        "content_length": len(content),
        "operation": "validate_test_content"
    })
    
    # Check for basic test patterns
    test_patterns = [
        r'def test_\w+\(',          # Test functions
        r'class Test\w+\(',         # Test classes
        r'@pytest\.mark\.',         # Pytest markers
        r'unittest\.TestCase',      # Unittest classes
        r'assert\s+',               # Assertions
        r'expect\(\w+\)',           # JavaScript-style expectations
    ]
    
    has_test_pattern = any(re.search(pattern, content, re.IGNORECASE) for pattern in test_patterns)
    
    if not has_test_pattern:
        logger.warning("Content does not contain recognizable test patterns", {
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "validation_result": "no_test_patterns"
        })
        return False
    
    # Check for potentially dangerous patterns
    dangerous_patterns = [
        r'os\.system\s*\(',         # System command execution
        r'subprocess\.\w+\(',       # Subprocess execution
        r'eval\s*\(',               # eval() calls
        r'exec\s*\(',               # exec() calls
        r'__import__\s*\(',         # Dynamic imports
        r'open\s*\(\s*[\'"][/\\]',  # Absolute path file access
        r'rm\s+-rf',                # Dangerous shell commands
        r'del\s+[/\\]',             # File deletion
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            logger.error("Dangerous pattern detected in test content", {
                "pattern": pattern,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "validation_result": "dangerous_pattern_detected"
            })
            return False
    
    # Additional validation for code structure
    try:
        # Try to parse as Python code to ensure it's syntactically valid
        ast.parse(content)
        
        logger.debug("Test content validation passed", {
            "content_length": len(content),
            "validation_result": "valid"
        })
        
        return True
        
    except SyntaxError as e:
        logger.warning("Test content has syntax errors", {
            "line_number": e.lineno,
            "error_message": e.msg,
            "validation_result": "syntax_error"
        })
        return False
    except Exception as e:
        logger.warning("Unexpected error during test content validation", {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "validation_result": "validation_error"
        })
        return False