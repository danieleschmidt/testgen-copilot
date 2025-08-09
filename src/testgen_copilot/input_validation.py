"""Enhanced input validation and sanitization for robust security."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from .logging_config import get_generator_logger

logger = get_generator_logger()


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class SecurityValidationError(ValidationError):
    """Raised when security validation fails."""
    pass


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate and sanitize file paths to prevent directory traversal attacks.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Path: Validated and resolved path
        
    Raises:
        ValidationError: If path is invalid or insecure
    """
    try:
        path = Path(file_path).resolve()
        
        # Check for directory traversal attempts
        if ".." in str(path):
            raise SecurityValidationError(f"Directory traversal attempt detected: {file_path}")
            
        # Ensure path is absolute to prevent relative path attacks
        if not path.is_absolute():
            raise SecurityValidationError(f"Path must be absolute: {file_path}")
            
        # Check if file exists when required
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
            
        # Validate file extension for security
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.vbs', '.js'}
        if path.suffix.lower() in dangerous_extensions:
            raise SecurityValidationError(f"Dangerous file extension detected: {path.suffix}")
            
        return path
        
    except (OSError, ValueError) as e:
        raise ValidationError(f"Invalid file path: {file_path}") from e


def validate_project_directory(project_path: Union[str, Path]) -> Path:
    """Validate project directory for safety and accessibility.
    
    Args:
        project_path: Directory path to validate
        
    Returns:
        Path: Validated directory path
        
    Raises:
        ValidationError: If directory is invalid
    """
    path = validate_file_path(project_path, must_exist=True)
    
    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {project_path}")
        
    # Check read permissions
    if not os.access(path, os.R_OK):
        raise ValidationError(f"Directory is not readable: {project_path}")
        
    # Warn about sensitive directories
    sensitive_dirs = {'/etc', '/boot', '/sys', '/proc', '/dev'}
    if str(path).startswith(tuple(sensitive_dirs)):
        logger.warning(f"Accessing sensitive system directory: {path}")
        
    return path


def validate_file_patterns(patterns: List[str]) -> List[str]:
    """Validate file glob patterns for safety.
    
    Args:
        patterns: List of glob patterns
        
    Returns:
        List[str]: Validated patterns
        
    Raises:
        ValidationError: If patterns are invalid or unsafe
    """
    if not patterns:
        raise ValidationError("File patterns cannot be empty")
        
    validated = []
    dangerous_patterns = {'**/.*', '**/.ssh/*', '**/id_rsa*', '**/password*'}
    
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise ValidationError(f"Pattern must be string: {pattern}")
            
        # Check for dangerous patterns
        if pattern.lower() in dangerous_patterns:
            raise SecurityValidationError(f"Dangerous file pattern detected: {pattern}")
            
        # Validate glob pattern syntax
        if not re.match(r'^[a-zA-Z0-9_\-\.\*/\[\]]+$', pattern):
            raise ValidationError(f"Invalid glob pattern: {pattern}")
            
        validated.append(pattern)
        
    return validated


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary for security and correctness.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")
        
    validated = {}
    
    # Validate language setting
    if 'language' in config:
        lang = config['language']
        supported_languages = {'python', 'javascript', 'typescript', 'java', 'csharp', 'go', 'rust'}
        if lang not in supported_languages:
            raise ValidationError(f"Unsupported language: {lang}")
        validated['language'] = lang
        
    # Validate coverage target
    if 'coverage_target' in config:
        target = config['coverage_target']
        if not isinstance(target, (int, float)) or not 0 <= target <= 100:
            raise ValidationError("Coverage target must be between 0 and 100")
        validated['coverage_target'] = float(target)
        
    # Validate security rules
    if 'security_rules' in config:
        rules = config['security_rules']
        if not isinstance(rules, dict):
            raise ValidationError("Security rules must be a dictionary")
        validated['security_rules'] = rules
        
    return validated


def validate_url(url: str) -> str:
    """Validate URL for security and format.
    
    Args:
        url: URL to validate
        
    Returns:
        str: Validated URL
        
    Raises:
        ValidationError: If URL is invalid or unsafe
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")
        
    try:
        parsed = urlparse(url)
        
        # Ensure scheme is safe
        if parsed.scheme not in {'http', 'https'}:
            raise SecurityValidationError(f"Unsafe URL scheme: {parsed.scheme}")
            
        # Validate hostname
        if not parsed.netloc:
            raise ValidationError("URL must have a valid hostname")
            
        # Check for suspicious patterns
        suspicious_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '[::]']
        if any(pattern in parsed.netloc.lower() for pattern in suspicious_patterns):
            logger.warning(f"Potentially unsafe URL detected: {url}")
            
        return url
        
    except Exception as e:
        raise ValidationError(f"Invalid URL: {url}") from e


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize text input to prevent injection attacks.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
        
    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")
        
    if len(text) > max_length:
        raise ValidationError(f"Input too long: {len(text)} > {max_length}")
        
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Log if sanitization changed the input
    if sanitized != text:
        logger.warning("Input was sanitized to remove control characters")
        
    return sanitized


def validate_resource_limits(
    cpu_limit: Optional[float] = None,
    memory_limit: Optional[int] = None,
    timeout: Optional[float] = None
) -> Dict[str, Union[float, int]]:
    """Validate resource limit parameters.
    
    Args:
        cpu_limit: CPU usage limit (cores)
        memory_limit: Memory limit (MB) 
        timeout: Timeout limit (seconds)
        
    Returns:
        Dict: Validated resource limits
        
    Raises:
        ValidationError: If limits are invalid
    """
    validated = {}
    
    if cpu_limit is not None:
        if not isinstance(cpu_limit, (int, float)) or cpu_limit <= 0:
            raise ValidationError("CPU limit must be positive number")
        if cpu_limit > 64:  # Reasonable upper limit
            raise ValidationError("CPU limit too high (max 64 cores)")
        validated['cpu_limit'] = float(cpu_limit)
        
    if memory_limit is not None:
        if not isinstance(memory_limit, int) or memory_limit <= 0:
            raise ValidationError("Memory limit must be positive integer")
        if memory_limit > 32768:  # 32GB limit
            raise ValidationError("Memory limit too high (max 32GB)")
        validated['memory_limit'] = memory_limit
        
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError("Timeout must be positive number")
        if timeout > 3600:  # 1 hour limit
            raise ValidationError("Timeout too high (max 1 hour)")
        validated['timeout'] = float(timeout)
        
    return validated