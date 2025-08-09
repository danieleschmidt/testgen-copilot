"""Advanced error recovery and fault tolerance mechanisms."""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager

from .logging_config import get_generator_logger

logger = get_generator_logger()


class RecoveryError(Exception):
    """Base exception for error recovery failures."""
    pass


class RetryExhaustedError(RecoveryError):
    """Raised when all retry attempts have been exhausted."""
    pass


class CircuitBreakerOpenError(RecoveryError):
    """Raised when circuit breaker is open and rejecting calls."""
    pass


class FallbackExecutor:
    """Executes fallback strategies when primary operations fail."""
    
    def __init__(self):
        self.fallback_registry: Dict[str, Callable] = {}
        
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for a specific operation.
        
        Args:
            operation_name: Name of the operation
            fallback_func: Function to call when primary operation fails
        """
        self.fallback_registry[operation_name] = fallback_func
        logger.info(f"Registered fallback for operation: {operation_name}")
        
    def execute_with_fallback(self, 
                            operation_name: str,
                            primary_func: Callable,
                            *args,
                            **kwargs) -> Any:
        """Execute primary function with fallback on failure.
        
        Args:
            operation_name: Name of the operation
            primary_func: Primary function to execute
            *args: Arguments to pass to functions
            **kwargs: Keyword arguments to pass to functions
            
        Returns:
            Result from primary function or fallback
            
        Raises:
            RecoveryError: If both primary and fallback fail
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary operation {operation_name} failed: {e}")
            
            if operation_name in self.fallback_registry:
                try:
                    fallback_func = self.fallback_registry[operation_name]
                    result = fallback_func(*args, **kwargs)
                    logger.info(f"Fallback executed successfully for: {operation_name}")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for {operation_name}: {fallback_error}")
                    raise RecoveryError(f"Both primary and fallback failed") from fallback_error
            else:
                logger.error(f"No fallback registered for: {operation_name}")
                raise RecoveryError(f"Operation failed and no fallback available") from e


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise RetryExhaustedError(
                            f"Function {func.__name__} failed after {max_attempts} attempts"
                        ) from e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
                    
            raise last_exception
            
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting to close circuit
            expected_exception: Exception type that triggers circuit opening
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func: Callable) -> Callable:
        """Decorate function with circuit breaker logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
        
    def _call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                logger.info(f"Circuit breaker entering HALF_OPEN state for {func.__name__}")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for {func.__name__}"
                )
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
        
    def _on_success(self):
        """Handle successful function execution."""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            logger.info("Circuit breaker reset to CLOSED state")
            
    def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


@contextmanager
def error_boundary(operation_name: str, 
                  suppress_errors: bool = False,
                  fallback_value: Any = None):
    """Context manager for isolating errors within operations.
    
    Args:
        operation_name: Name of the operation for logging
        suppress_errors: Whether to suppress errors and continue
        fallback_value: Value to return if error is suppressed
        
    Yields:
        None
        
    Raises:
        Exception: Re-raises exception if not suppressed
    """
    try:
        logger.debug(f"Starting error boundary for: {operation_name}")
        yield
        logger.debug(f"Completed error boundary for: {operation_name}")
        
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}", exc_info=True)
        
        if suppress_errors:
            logger.warning(f"Suppressing error in {operation_name}, returning fallback")
            return fallback_value
        else:
            raise


class GracefulDegradation:
    """Manages graceful degradation of service capabilities."""
    
    def __init__(self):
        self.feature_states: Dict[str, bool] = {}
        self.degraded_implementations: Dict[str, Callable] = {}
        
    def register_feature(self, feature_name: str, 
                        degraded_impl: Optional[Callable] = None):
        """Register a feature with optional degraded implementation.
        
        Args:
            feature_name: Name of the feature
            degraded_impl: Function to use when feature is degraded
        """
        self.feature_states[feature_name] = True
        if degraded_impl:
            self.degraded_implementations[feature_name] = degraded_impl
            
        logger.info(f"Registered feature: {feature_name}")
        
    def degrade_feature(self, feature_name: str, reason: str = ""):
        """Degrade a specific feature.
        
        Args:
            feature_name: Name of the feature to degrade
            reason: Reason for degradation
        """
        if feature_name in self.feature_states:
            self.feature_states[feature_name] = False
            logger.warning(f"Degraded feature {feature_name}: {reason}")
        
    def is_available(self, feature_name: str) -> bool:
        """Check if a feature is available (not degraded).
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            bool: True if feature is available
        """
        return self.feature_states.get(feature_name, False)
        
    def execute_feature(self, feature_name: str, 
                       primary_func: Callable,
                       *args, **kwargs) -> Any:
        """Execute a feature with graceful degradation.
        
        Args:
            feature_name: Name of the feature
            primary_func: Primary implementation
            *args: Arguments to pass to functions
            **kwargs: Keyword arguments
            
        Returns:
            Result from primary or degraded implementation
            
        Raises:
            RecoveryError: If feature is degraded and no fallback exists
        """
        if self.is_available(feature_name):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary implementation failed for {feature_name}: {e}")
                self.degrade_feature(feature_name, str(e))
                # Fall through to degraded implementation
                
        # Use degraded implementation
        if feature_name in self.degraded_implementations:
            degraded_func = self.degraded_implementations[feature_name]
            logger.info(f"Using degraded implementation for: {feature_name}")
            return degraded_func(*args, **kwargs)
        else:
            raise RecoveryError(f"Feature {feature_name} is degraded and no fallback available")


# Global instances for easy access
fallback_executor = FallbackExecutor()
graceful_degradation = GracefulDegradation()


def safe_execute(func: Callable, 
                default_value: Any = None,
                log_errors: bool = True) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_value: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Function result or default value on error
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default_value


async def async_retry_with_backoff(
    async_func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    *args, **kwargs
) -> Any:
    """Asynchronous retry with exponential backoff.
    
    Args:
        async_func: Async function to retry
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        RetryExhaustedError: If all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await async_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                raise RetryExhaustedError(
                    f"Async function {async_func.__name__} failed after {max_attempts} attempts"
                ) from e
                
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Async attempt {attempt + 1}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )
            
            await asyncio.sleep(delay)
            
    raise last_exception