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

# Try to import psutil for better memory monitoring, fall back to platform-specific methods
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Windows-specific memory monitoring
try:
    if sys.platform.startswith('win'):
        import ctypes
        from ctypes import wintypes
        HAS_WINDOWS_API = True
    else:
        HAS_WINDOWS_API = False
except ImportError:
    HAS_WINDOWS_API = False


# Configuration constants
AST_PARSE_TIMEOUT = 30  # seconds
MAX_PROJECT_FILES = 1000  # maximum files to process in a single batch
MAX_MEMORY_MB = 512  # maximum memory usage in MB
MIN_MEMORY_AVAILABLE_MB = 100  # minimum available memory required


# Use built-in TimeoutError class from Python 3.3+


class ResourceMemoryError(Exception):
    """Raised when memory usage exceeds limits."""
    pass


class ResourceLimits:
    """Configuration class for resource limits and thresholds."""
    
    def __init__(
        self,
        memory_threshold_percent: int = 80,
        max_batch_size: int = MAX_PROJECT_FILES,
        max_memory_mb: int = MAX_MEMORY_MB,
        min_memory_available_mb: int = MIN_MEMORY_AVAILABLE_MB,
        ast_parse_timeout: int = AST_PARSE_TIMEOUT
    ):
        self.memory_threshold_percent = memory_threshold_percent
        self.max_batch_size = max_batch_size
        self.max_memory_mb = max_memory_mb
        self.min_memory_available_mb = min_memory_available_mb
        self.ast_parse_timeout = ast_parse_timeout


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
                unit = "second" if self.timeout_seconds == 1 else "seconds"
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} {unit}")
            
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
        else:
            # Use thread-based timeout (Windows and other platforms)
            # Note: This approach is limited - it can only check for timeout
            # at specific points, not interrupt arbitrary operations
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
                unit = "second" if self.timeout_seconds == 1 else "seconds"
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} {unit}")
                
    def check_timeout(self):
        """Check if timeout has occurred (for thread-based timeout only)."""
        if not self.use_signals and self.timed_out:
            unit = "second" if self.timeout_seconds == 1 else "seconds"
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} {unit}")


class MemoryMonitor:
    """Monitor memory usage and implement circuit breaker patterns."""
    
    def __init__(self, max_memory_mb: int = MAX_MEMORY_MB):
        self.max_memory_mb = max_memory_mb
        self.logger = get_generator_logger()
        
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB with cross-platform support."""
        try:
            # First try psutil (most reliable across platforms)
            if HAS_PSUTIL:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB
                self.logger.debug("Memory usage retrieved via psutil", {
                    "memory_mb": round(memory_mb, 2),
                    "method": "psutil"
                })
                return round(memory_mb, 2)
            
            # Windows-specific fallback using Windows API
            elif HAS_WINDOWS_API and sys.platform.startswith('win'):
                return self._get_windows_memory_usage()
            
            # Unix/Linux fallback using resource module
            elif hasattr(resource, 'RUSAGE_SELF'):
                usage = resource.getrusage(resource.RUSAGE_SELF)
                # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
                if sys.platform == 'darwin':
                    memory_mb = usage.ru_maxrss / 1024 / 1024
                else:
                    memory_mb = usage.ru_maxrss / 1024
                
                self.logger.debug("Memory usage retrieved via resource module", {
                    "memory_mb": round(memory_mb, 2),
                    "method": "resource_module",
                    "platform": sys.platform
                })
                return round(memory_mb, 2)
            
            else:
                self.logger.warning("No memory monitoring method available for this platform", {
                    "platform": sys.platform,
                    "has_psutil": HAS_PSUTIL,
                    "has_windows_api": HAS_WINDOWS_API
                })
                return 0.0  # Unable to get memory info on this platform
                
        except Exception as e:
            self.logger.warning("Failed to get memory usage", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "has_psutil": HAS_PSUTIL,
                "platform": sys.platform
            })
            return 0.0
    
    def _get_windows_memory_usage(self) -> float:
        """Get current memory usage on Windows using Windows API."""
        try:
            # Define Windows API structures and functions
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            
            # Get handle to current process
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            
            current_process = kernel32.GetCurrentProcess()
            process_memory = PROCESS_MEMORY_COUNTERS()
            process_memory.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            
            # Get memory information
            if psapi.GetProcessMemoryInfo(current_process, ctypes.byref(process_memory), process_memory.cb):
                memory_mb = process_memory.WorkingSetSize / 1024 / 1024
                self.logger.debug("Memory usage retrieved via Windows API", {
                    "memory_mb": round(memory_mb, 2),
                    "method": "windows_api"
                })
                return round(memory_mb, 2)
            else:
                self.logger.warning("Windows API call failed to get process memory info")
                return 0.0
                
        except Exception as e:
            self.logger.warning("Failed to get Windows memory usage", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return 0.0
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB with cross-platform support."""
        try:
            # First try psutil (most reliable across platforms)
            if HAS_PSUTIL:
                memory = psutil.virtual_memory()
                available_mb = memory.available / 1024 / 1024
                self.logger.debug("Available memory retrieved via psutil", {
                    "available_mb": round(available_mb, 2),
                    "method": "psutil"
                })
                return round(available_mb, 2)
            
            # Windows-specific fallback using Windows API
            elif HAS_WINDOWS_API and sys.platform.startswith('win'):
                return self._get_windows_available_memory()
            
            # Linux fallback using /proc/meminfo
            elif os.path.exists('/proc/meminfo'):
                return self._get_linux_available_memory()
            
            # macOS fallback using vm_stat
            elif sys.platform == 'darwin':
                return self._get_macos_available_memory()
            
            else:
                self.logger.warning("No available memory monitoring method for this platform", {
                    "platform": sys.platform,
                    "has_psutil": HAS_PSUTIL,
                    "has_windows_api": HAS_WINDOWS_API
                })
                return 1000.0  # Conservative fallback
                
        except Exception as e:
            self.logger.warning("Failed to get available memory", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "has_psutil": HAS_PSUTIL,
                "platform": sys.platform
            })
            return 1000.0  # Conservative fallback
    
    def _get_windows_available_memory(self) -> float:
        """Get available memory on Windows using Windows API."""
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]
            
            kernel32 = ctypes.windll.kernel32
            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
                available_mb = memory_status.ullAvailPhys / 1024 / 1024
                self.logger.debug("Available memory retrieved via Windows API", {
                    "available_mb": round(available_mb, 2),
                    "method": "windows_api"
                })
                return round(available_mb, 2)
            else:
                self.logger.warning("Windows API call failed to get memory status")
                return 1000.0
                
        except Exception as e:
            self.logger.warning("Failed to get Windows available memory", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return 1000.0
    
    def _get_linux_available_memory(self) -> float:
        """Get available memory on Linux using /proc/meminfo."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        available_kb = int(line.split()[1])
                        available_mb = available_kb / 1024
                        self.logger.debug("Available memory retrieved via /proc/meminfo", {
                            "available_mb": round(available_mb, 2),
                            "method": "proc_meminfo"
                        })
                        return round(available_mb, 2)
                
                # If MemAvailable not found, try MemFree (older kernels)
                f.seek(0)
                for line in f:
                    if line.startswith('MemFree:'):
                        free_kb = int(line.split()[1])
                        free_mb = free_kb / 1024
                        self.logger.debug("Free memory retrieved via /proc/meminfo (fallback)", {
                            "available_mb": round(free_mb, 2),
                            "method": "proc_meminfo_free"
                        })
                        return round(free_mb, 2)
                        
            return 1000.0  # Fallback if parsing fails
            
        except Exception as e:
            self.logger.warning("Failed to get Linux available memory", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return 1000.0
    
    def _get_macos_available_memory(self) -> float:
        """Get available memory on macOS using vm_stat."""
        try:
            import subprocess
            
            # Run vm_stat command
            result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                page_size = 4096  # Default page size on macOS
                free_pages = 0
                
                for line in lines:
                    if 'page size of' in line:
                        # Extract page size if specified
                        try:
                            page_size = int(line.split()[-2])
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('Pages free:'):
                        try:
                            free_pages = int(line.split()[-1].rstrip('.'))
                        except (ValueError, IndexError):
                            pass
                
                if free_pages > 0:
                    available_mb = (free_pages * page_size) / 1024 / 1024
                    self.logger.debug("Available memory retrieved via vm_stat", {
                        "available_mb": round(available_mb, 2),
                        "method": "vm_stat",
                        "free_pages": free_pages,
                        "page_size": page_size
                    })
                    return round(available_mb, 2)
            
            return 1000.0  # Fallback
            
        except Exception as e:
            self.logger.warning("Failed to get macOS available memory", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return 1000.0
    
    def get_current_usage_percent(self) -> Optional[float]:
        """Get current memory usage as percentage of available system memory."""
        try:
            if HAS_PSUTIL:
                # Use psutil for most accurate percentage calculation
                memory = psutil.virtual_memory()
                percent = memory.percent
                self.logger.debug("Memory usage percentage retrieved via psutil", {
                    "usage_percent": percent,
                    "method": "psutil"
                })
                return percent
            else:
                # Calculate manually using our cross-platform methods
                current_mb = self.get_current_memory_mb()
                available_mb = self.get_available_memory_mb()
                
                if current_mb > 0 and available_mb > 0:
                    # Estimate total memory (this is approximate)
                    total_mb = available_mb + current_mb
                    percent = (current_mb / total_mb) * 100
                    
                    self.logger.debug("Memory usage percentage calculated manually", {
                        "usage_percent": round(percent, 2),
                        "current_mb": current_mb,
                        "total_mb": total_mb,
                        "method": "manual_calculation"
                    })
                    return round(percent, 2)
                else:
                    self.logger.warning("Cannot calculate memory percentage - insufficient data")
                    return None
                    
        except Exception as e:
            self.logger.warning("Failed to get memory usage percentage", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return None
    
    def is_memory_exceeded(self) -> bool:
        """Check if current memory usage exceeds limits."""
        try:
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
            
        except Exception as e:
            self.logger.error("Failed to check memory limits", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "max_memory_mb": self.max_memory_mb
            })
            # Return False on error to avoid false positives
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


def _parse_ast_worker(content: str, filename: str, result_queue) -> None:
    """Worker function for multiprocessing AST parsing."""
    try:
        tree = ast.parse(content, filename=filename)
        result_queue.put(("success", tree))
    except Exception as e:
        result_queue.put(("error", e))


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
        # Use signal-based timeout on Unix platforms
        if hasattr(signal, 'SIGALRM'):
            with CrossPlatformTimeoutHandler(timeout_seconds):
                start_time = time.time()
                tree = ast.parse(content, filename=filename)
                parse_time = time.time() - start_time
                
                logger.debug("AST parsing completed successfully", {
                    "filename": filename,
                    "parse_time_ms": round(parse_time * 1000, 2),
                    "ast_node_count": len(list(ast.walk(tree))),
                "platform": sys.platform,
                "timeout_method": "signal"
            })
            
            return tree
        else:
            # Use multiprocessing-based timeout on Windows and other platforms
            import multiprocessing
            
            start_time = time.time()
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_parse_ast_worker, 
                args=(content, filename, result_queue)
            )
            
            process.start()
            process.join(timeout=timeout_seconds)
            
            if process.is_alive():
                # Process timed out
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                
                unit = "second" if timeout_seconds == 1 else "seconds"
                logger.error("AST parsing timed out", {
                    "filename": filename,
                    "timeout_seconds": timeout_seconds,
                    "content_length": len(content),
                    "error_type": "parse_timeout",
                    "platform": sys.platform,
                    "timeout_method": "multiprocessing"
                })
                raise TimeoutError(f"AST parsing timed out after {timeout_seconds} {unit}")
            
            # Get result from queue
            if not result_queue.empty():
                result_type, result = result_queue.get()
                if result_type == "success":
                    parse_time = time.time() - start_time
                    logger.debug("AST parsing completed successfully", {
                        "filename": filename,
                        "parse_time_ms": round(parse_time * 1000, 2),
                        "ast_node_count": len(list(ast.walk(result))),
                        "platform": sys.platform,
                        "timeout_method": "multiprocessing"
                    })
                    return result
                elif result_type == "error":
                    raise result
            else:
                # Process completed but no result - likely an error
                raise RuntimeError(f"AST parsing process completed without result for {filename}")
            
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


class ResourceMonitor:
    """Comprehensive resource monitoring and validation class."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.memory_monitor = MemoryMonitor()
        self.batch_processor = BatchProcessor()
        self.logger = get_generator_logger()
    
    def check_memory_usage(self) -> bool:
        """Check if current memory usage is within limits."""
        try:
            # Use memory monitor to check against thresholds
            current_usage = self.memory_monitor.get_current_usage_percent()
            
            if current_usage is None:
                # If we can't determine usage, assume it's okay but log warning
                self.logger.warning("Could not determine memory usage, proceeding with caution")
                return True
            
            # Check against our threshold
            within_limits = current_usage < self.limits.memory_threshold_percent
            
            self.logger.debug("Memory usage check", {
                "current_usage_percent": current_usage,
                "threshold_percent": self.limits.memory_threshold_percent,
                "within_limits": within_limits
            })
            
            if not within_limits:
                self.logger.warning("Memory usage exceeds threshold", {
                    "current_usage_percent": current_usage,
                    "threshold_percent": self.limits.memory_threshold_percent
                })
            
            return within_limits
            
        except Exception as e:
            self.logger.error("Error checking memory usage", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            # If we can't check, assume it's okay but proceed with caution
            return True
    
    def validate_test_content(self, content: str, filename: str) -> bool:
        """Validate generated test content for safety and correctness."""
        self.logger.debug("Validating test content", {
            "filename": filename,
            "content_length": len(content)
        })
        
        # Use the existing validate_test_content function
        return validate_test_content(content)
    
    def validate_batch_size(self, batch_size: int) -> bool:
        """Validate if batch size is within acceptable limits."""
        within_limits = batch_size <= self.limits.max_batch_size
        
        self.logger.debug("Batch size validation", {
            "batch_size": batch_size,
            "max_batch_size": self.limits.max_batch_size,
            "within_limits": within_limits
        })
        
        if not within_limits:
            self.logger.warning("Batch size exceeds maximum limit", {
                "batch_size": batch_size,
                "max_batch_size": self.limits.max_batch_size
            })
        
        return within_limits
