"""Enhanced error handling and recovery mechanisms for robust operation."""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, TypeVar, Union

from .logging_config import get_core_logger

T = TypeVar('T')


class RetryStrategy(Enum):
    """Different retry strategies for error recovery."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Configuration for retry operations."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retryable_exceptions: tuple = (Exception,)
    ignore_exceptions: tuple = (KeyboardInterrupt, SystemExit)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker pattern implementation for robust error handling."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.logger = get_core_logger()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker logic."""
        current_time = time.time()
        
        # Check if circuit should be closed after recovery timeout
        if (self.state == CircuitBreakerState.OPEN and 
            current_time - self.last_failure_time > self.config.recovery_timeout):
            self.state = CircuitBreakerState.HALF_OPEN
            self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
        
        # Reject requests if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            self.logger.warning("Circuit breaker is OPEN, rejecting request")
            raise RuntimeError("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count and close circuit
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker reset to CLOSED state after successful call")
            
            return result
            
        except self.config.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            self.logger.warning(f"Circuit breaker recorded failure {self.failure_count}/{self.config.failure_threshold}")
            
            # Open circuit if failure threshold exceeded
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.error("Circuit breaker opened due to excessive failures")
            
            raise


class RobustErrorHandler:
    """Comprehensive error handling with multiple recovery strategies."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def retry_with_backoff(
        self,
        func: Callable[..., T],
        config: RetryConfig = None,
        *args,
        **kwargs
    ) -> T:
        """Execute function with configurable retry and backoff."""
        config = config or RetryConfig()
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                self.logger.debug(f"Attempting operation (attempt {attempt + 1}/{config.max_attempts})")
                return func(*args, **kwargs)
                
            except config.ignore_exceptions:
                self.logger.info("Operation interrupted by user")
                raise
                
            except config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == config.max_attempts - 1:
                    self.logger.error(f"Operation failed after {config.max_attempts} attempts", {
                        "final_error": str(e),
                        "error_type": type(e).__name__
                    })
                    break
                
                # Calculate delay based on strategy
                delay = self._calculate_delay(attempt, config)
                
                self.logger.warning(f"Operation failed, retrying in {delay:.2f}s", {
                    "attempt": attempt + 1,
                    "max_attempts": config.max_attempts,
                    "error": str(e),
                    "delay": delay
                })
                
                time.sleep(delay)
        
        # Re-raise the last exception if all attempts failed
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Unexpected retry loop completion")
    
    async def async_retry_with_backoff(
        self,
        func: Callable[..., T],
        config: RetryConfig = None,
        *args,
        **kwargs
    ) -> T:
        """Async version of retry with backoff."""
        config = config or RetryConfig()
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                self.logger.debug(f"Attempting async operation (attempt {attempt + 1}/{config.max_attempts})")
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except config.ignore_exceptions:
                self.logger.info("Async operation interrupted")
                raise
                
            except config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == config.max_attempts - 1:
                    self.logger.error(f"Async operation failed after {config.max_attempts} attempts", {
                        "final_error": str(e),
                        "error_type": type(e).__name__
                    })
                    break
                
                delay = self._calculate_delay(attempt, config)
                
                self.logger.warning(f"Async operation failed, retrying in {delay:.2f}s", {
                    "attempt": attempt + 1,
                    "max_attempts": config.max_attempts,
                    "error": str(e),
                    "delay": delay
                })
                
                await asyncio.sleep(delay)
        
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Unexpected async retry loop completion")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a named operation."""
        if name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config)
            self.logger.info(f"Created circuit breaker for '{name}'")
        
        return self.circuit_breakers[name]
    
    @contextmanager
    def safe_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for safe operation execution with comprehensive error handling."""
        self.logger.debug(f"Starting safe operation: {operation_name}")
        
        try:
            yield
            self.logger.debug(f"Completed safe operation: {operation_name}")
            
        except KeyboardInterrupt:
            self.logger.info(f"Operation '{operation_name}' interrupted by user")
            raise
            
        except Exception as e:
            self.logger.error(f"Safe operation '{operation_name}' failed", {
                "operation": operation_name,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # Attempt to provide helpful error context
            self._log_error_context(e, operation_name)
            raise
    
    @asynccontextmanager
    async def async_safe_operation(self, operation_name: str) -> AsyncGenerator[None, None]:
        """Async context manager for safe operation execution."""
        self.logger.debug(f"Starting async safe operation: {operation_name}")
        
        try:
            yield
            self.logger.debug(f"Completed async safe operation: {operation_name}")
            
        except KeyboardInterrupt:
            self.logger.info(f"Async operation '{operation_name}' interrupted by user")
            raise
            
        except Exception as e:
            self.logger.error(f"Async safe operation '{operation_name}' failed", {
                "operation": operation_name,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            self._log_error_context(e, operation_name)
            raise
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt based on strategy."""
        if config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.initial_delay * (config.backoff_multiplier ** attempt)
        elif config.strategy == RetryStrategy.FIBONACCI:
            delay = config.initial_delay * self._fibonacci(attempt + 1)
        else:  # CUSTOM or fallback
            delay = config.initial_delay
        
        return min(delay, config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number for backoff strategy."""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def _log_error_context(self, error: Exception, operation_name: str) -> None:
        """Log additional context information for debugging."""
        context = {
            "operation": operation_name,
            "error_type": type(error).__name__,
            "error_args": str(error.args) if error.args else "None"
        }
        
        # Add file-related context if it's a file operation error
        if isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            if hasattr(error, 'filename') and error.filename:
                context["filename"] = str(error.filename)
            if hasattr(error, 'errno') and error.errno:
                context["errno"] = error.errno
        
        self.logger.error("Error context information", context)


# Global error handler instance
robust_error_handler = RobustErrorHandler()

# Convenience decorators
def retry_on_failure(config: RetryConfig = None):
    """Decorator for automatic retry with backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return robust_error_handler.retry_with_backoff(func, config, *args, **kwargs)
        return wrapper
    return decorator


def async_retry_on_failure(config: RetryConfig = None):
    """Decorator for automatic async retry with backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await robust_error_handler.async_retry_with_backoff(func, config, *args, **kwargs)
        return wrapper
    return decorator


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to apply circuit breaker pattern."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = robust_error_handler.get_circuit_breaker(name, config)
        return breaker(func)
    return decorator


# Example usage patterns
if __name__ == "__main__":
    # Example of using retry decorator
    @retry_on_failure(RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL))
    def example_flaky_operation():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ValueError("Simulated failure")
        return "Success!"
    
    # Example of using circuit breaker
    @circuit_breaker("external_service", CircuitBreakerConfig(failure_threshold=3))
    def call_external_service():
        # Simulated external service call
        import random
        if random.random() < 0.6:  # 60% chance of failure
            raise ConnectionError("Service unavailable")
        return "Service response"
    
    print("Testing robust error handling patterns...")
    try:
        result = example_flaky_operation()
        print(f"Flaky operation result: {result}")
    except Exception as e:
        print(f"Flaky operation failed: {e}")
    
    # Test circuit breaker
    for i in range(10):
        try:
            result = call_external_service()
            print(f"Service call {i+1}: {result}")
        except Exception as e:
            print(f"Service call {i+1} failed: {e}")