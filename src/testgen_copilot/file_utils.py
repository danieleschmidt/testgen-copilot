"""Safe file I/O utilities for TestGen Copilot."""

import ast
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Tuple
import os

from .logging_config import get_generator_logger
from .cache import file_content_cache, ast_cache


class FileSizeError(Exception):
    """Raised when a file exceeds the maximum allowed size."""
    pass


class SyntaxErrorStrategy(Enum):
    """Strategy for handling syntax errors during AST parsing."""
    RAISE = "raise"
    WARN_AND_SKIP = "warn_and_skip"
    RETURN_ERROR = "return_error"


def safe_read_file(
    path: Union[str, Path], 
    max_size_mb: int = 10
) -> str:
    """Safely read a file with size limits and comprehensive error handling.
    
    Args:
        path: Path to the file to read (string or Path object)
        max_size_mb: Maximum file size in megabytes (default: 10MB)
        
    Returns:
        str: The content of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file cannot be read due to permissions
        ValueError: If the file contains invalid text encoding
        FileSizeError: If the file exceeds the size limit
        OSError: For other file I/O errors
    """
    logger = get_generator_logger()
    file_path = Path(path)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Create cache key that includes size limit for safety
    cache_key = f"read_file_{max_size_mb}mb"
    
    # Try to get from cache first
    cached_content = file_content_cache.get(file_path, cache_key)
    if cached_content is not None:
        logger.debug("File content retrieved from cache", {
            "file_path": str(file_path),
            "cache_hit": True,
            "operation": "safe_read_file"
        })
        return cached_content
    
    # Log the operation
    logger.debug("Reading file with safety checks", {
        "file_path": str(file_path),
        "max_size_mb": max_size_mb,
        "cache_hit": False,
        "operation": "safe_read_file"
    })
    
    # Check if file exists
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error("File not found", {
            "file_path": str(file_path),
            "error_type": "file_not_found"
        })
        raise FileNotFoundError(error_msg)
        
    # Check if it's actually a file
    if not file_path.is_file():
        error_msg = f"Path is not a file: {file_path}"
        logger.error("Path is not a file", {
            "file_path": str(file_path),
            "error_type": "invalid_file_type"
        })
        raise ValueError(error_msg)
    
    # Check file size before reading
    try:
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            error_msg = f"File {file_path} is too large ({file_size / 1024 / 1024:.1f}MB, max: {max_size_mb}MB)"
            logger.error("File exceeds size limit", {
                "file_path": str(file_path),
                "file_size_mb": file_size / 1024 / 1024,
                "max_size_mb": max_size_mb,
                "error_type": "file_too_large"
            })
            raise FileSizeError(error_msg)
            
    except OSError as e:
        error_msg = f"Cannot access file {file_path}: {e}"
        logger.error("Cannot access file for size check", {
            "file_path": str(file_path),
            "error_type": "access_error",
            "error_message": str(e)
        })
        raise OSError(f"File I/O error reading {file_path}: {e}") from e
    
    # Read file content with comprehensive error handling
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Cache the content for future reads
        file_content_cache.put(file_path, content, cache_key)
        
        logger.debug("File read successfully", {
            "file_path": str(file_path),
            "content_length": len(content),
            "file_size_bytes": file_size,
            "cached": True
        })
        
        return content
        
    except PermissionError as e:
        error_msg = f"Permission denied reading {file_path}: {e}"
        logger.error("Permission denied reading file", {
            "file_path": str(file_path),
            "error_type": "permission_error",
            "error_message": str(e)
        })
        raise PermissionError(error_msg) from e
        
    except UnicodeDecodeError as e:
        error_msg = f"Invalid text encoding in {file_path}: {e}"
        logger.error("Unicode decode error", {
            "file_path": str(file_path),
            "error_type": "encoding_error",
            "encoding": e.encoding,
            "error_message": str(e)
        })
        raise ValueError(error_msg) from e
        
    except OSError as e:
        error_msg = f"File I/O error reading {file_path}: {e}"
        logger.error("File I/O error", {
            "file_path": str(file_path),
            "error_type": "io_error",
            "error_message": str(e)
        })
        raise OSError(error_msg) from e


def safe_parse_ast(
    path: Union[str, Path],
    content: Optional[str] = None,
    max_size_mb: int = 10,
    timeout_seconds: Optional[int] = None,
    raise_on_syntax_error: bool = True
) -> Union[Tuple[ast.AST, str], None]:
    """Safely parse Python AST from file with consistent error handling.
    
    Args:
        path: Path to the file to parse (string or Path object)
        content: Optional file content (if provided, file won't be read)
        max_size_mb: Maximum file size in megabytes (default: 10MB)
        timeout_seconds: Optional timeout for AST parsing operations
        raise_on_syntax_error: Whether to raise SyntaxError or return None
        
    Returns:
        Tuple of (AST tree, file content) on success, None if syntax error and not raising
        
    Raises:
        FileNotFoundError: If the file doesn't exist (when content not provided)
        PermissionError: If the file cannot be read due to permissions
        ValueError: If the file contains invalid text encoding
        FileSizeError: If the file exceeds the size limit
        SyntaxError: If syntax error and raise_on_syntax_error=True
        OSError: For other file I/O errors
    """
    logger = get_generator_logger()
    file_path = Path(path)
    original_content_provided = content is not None
    
    # Create cache key based on parsing parameters
    cache_key = f"ast_parse_{max_size_mb}mb_{timeout_seconds}s_{raise_on_syntax_error}"
    
    # Try to get from cache first (only if content not provided)
    if content is None:
        cached_result = ast_cache.get(file_path, cache_key)
        if cached_result is not None:
            logger.debug("AST parsing result retrieved from cache", {
                "file_path": str(file_path),
                "cache_hit": True,
                "operation": "safe_parse_ast"
            })
            return cached_result
    
    logger.debug("Starting safe AST parsing", {
        "file_path": str(file_path),
        "content_provided": content is not None,
        "max_size_mb": max_size_mb,
        "timeout_seconds": timeout_seconds,
        "raise_on_syntax_error": raise_on_syntax_error,
        "cache_hit": False
    })
    
    # Get file content
    if content is None:
        try:
            content = safe_read_file(path, max_size_mb=max_size_mb)
        except (FileNotFoundError, PermissionError, ValueError, FileSizeError, OSError) as e:
            # File reading errors are already logged by safe_read_file
            raise
    
    # Parse AST with optional timeout
    try:
        if timeout_seconds is not None:
            # Use timeout protection if available
            try:
                from .resource_limits import safe_parse_ast_with_timeout
                tree = safe_parse_ast_with_timeout(content, str(file_path), timeout_seconds)
            except ImportError:
                # Fallback if resource_limits not available
                logger.debug("Timeout protection not available, parsing without timeout", {
                    "file_path": str(file_path)
                })
                tree = ast.parse(content, filename=str(file_path))
        else:
            tree = ast.parse(content, filename=str(file_path))
        
        result = (tree, content)
        
        # Cache the result for future parses (only if we read the file ourselves)
        if not original_content_provided:
            ast_cache.put(file_path, result, cache_key)
        
        logger.debug("AST parsing completed successfully", {
            "file_path": str(file_path),
            "content_length": len(content),
            "ast_node_count": len(list(ast.walk(tree))),
            "cached": content is None
        })
        
        return result
        
    except SyntaxError as e:
        logger.error("Syntax error during AST parsing", {
            "file_path": str(file_path),
            "line_number": e.lineno,
            "column_number": e.offset,
            "error_message": e.msg,
            "error_type": "syntax_error"
        })
        
        if raise_on_syntax_error:
            # Enhance the error message with file context while preserving original attributes
            enhanced_msg = f"Cannot parse {file_path}: syntax error at line {e.lineno}: {e.msg}"
            enhanced_error = SyntaxError(enhanced_msg)
            enhanced_error.lineno = e.lineno
            enhanced_error.offset = e.offset
            enhanced_error.filename = str(file_path)
            enhanced_error.text = e.text
            raise enhanced_error from e
        else:
            logger.warning("Returning None due to syntax error (raise_on_syntax_error=False)", {
                "file_path": str(file_path),
                "line_number": e.lineno
            })
            return None
            
    except Exception as e:
        logger.error("Unexpected error during AST parsing", {
            "file_path": str(file_path),
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        raise