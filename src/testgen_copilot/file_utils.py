"""Safe file I/O utilities for TestGen Copilot."""

from pathlib import Path
from typing import Union
import os

from .logging_config import get_generator_logger


class FileSizeError(Exception):
    """Raised when a file exceeds the maximum allowed size."""
    pass


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
    
    # Log the operation
    logger.debug("Reading file with safety checks", {
        "file_path": str(file_path),
        "max_size_mb": max_size_mb,
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
        
        logger.debug("File read successfully", {
            "file_path": str(file_path),
            "content_length": len(content),
            "file_size_bytes": file_size
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