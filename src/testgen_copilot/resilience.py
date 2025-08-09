"""Resilience patterns for TestGen Copilot - circuit breakers, retries, bulkheads."""

from __future__ import annotations

import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from .logging_config import get_core_logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_duration_seconds: float = 60.0
    call_timeout_seconds: float = 10.0
    expected_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    retryable_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [ConnectionError, TimeoutError, OSError]
    )


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent_calls: int = 10
    max_queue_size: int = 100
    queue_timeout_seconds: float = 30.0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


class BulkheadFullError(Exception):
    """Raised when bulkhead capacity is exceeded."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.logger = get_core_logger()

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.total_calls = 0
        self.successful_calls = 0

        self._lock = threading.Lock()

    @contextmanager
    def call(self):
        """Context manager for making calls through circuit breaker."""
        if not self._can_proceed():
            raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")

        start_time = time.time()
        self.total_calls += 1

        try:
            # Set call timeout if configured
            if self.config.call_timeout_seconds > 0:
                # Note: This is a simplified timeout - in production you'd want
                # more sophisticated timeout handling
                pass

            yield

            # Call succeeded
            self._record_success()

        except Exception as e:
            # Check if this exception should trigger circuit breaker
            if any(isinstance(e, exc_type) for exc_type in self.config.expected_exceptions):
                self._record_failure()

            raise

    def _can_proceed(self) -> bool:
        """Check if call can proceed based on circuit state."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if (self.last_failure_time and
                    datetime.now(timezone.utc) - self.last_failure_time >
                    timedelta(seconds=self.config.timeout_duration_seconds)):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN", {
                        "circuit_breaker": self.name,
                        "state_transition": "OPEN -> HALF_OPEN"
                    })
                    return True
                return False

            elif self.state == CircuitState.HALF_OPEN:
                return True

        return False

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.successful_calls += 1

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info("Circuit breaker transitioning to CLOSED", {
                        "circuit_breaker": self.name,
                        "state_transition": "HALF_OPEN -> CLOSED",
                        "success_count": self.success_count
                    })

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.logger.error("Circuit breaker transitioning to OPEN", {
                        "circuit_breaker": self.name,
                        "state_transition": "CLOSED -> OPEN",
                        "failure_count": self.failure_count,
                        "failure_threshold": self.config.failure_threshold
                    })

            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state goes back to open
                self.state = CircuitState.OPEN
                self.logger.error("Circuit breaker transitioning to OPEN from HALF_OPEN", {
                    "circuit_breaker": self.name,
                    "state_transition": "HALF_OPEN -> OPEN",
                    "failure_in_test": True
                })

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            success_rate = (
                (self.successful_calls / self.total_calls * 100)
                if self.total_calls > 0 else 0
            )

            return {
                "name": self.name,
                "state": self.state.value,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "success_rate_percent": success_rate,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_duration_seconds": self.config.timeout_duration_seconds
                }
            }


class RetryMechanism:
    """Retry mechanism with various backoff strategies."""

    def __init__(self, name: str, config: RetryConfig):
        self.name = name
        self.config = config
        self.logger = get_core_logger()

    def __call__(self, func: Callable) -> Callable:
        """Decorator for adding retry logic to functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, *args, **kwargs)
        return wrapper

    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)

                if attempt > 0:
                    self.logger.info("Retry successful", {
                        "retry_mechanism": self.name,
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "total_attempts": self.config.max_attempts
                    })

                return result

            except Exception as e:
                last_exception = e

                # Check if this exception is retryable
                is_retryable = any(
                    isinstance(e, exc_type)
                    for exc_type in self.config.retryable_exceptions
                )

                if not is_retryable or attempt == self.config.max_attempts - 1:
                    self.logger.error("Retry mechanism exhausted or non-retryable error", {
                        "retry_mechanism": self.name,
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "total_attempts": self.config.max_attempts,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "is_retryable": is_retryable
                    })
                    break

                # Calculate delay before next attempt
                delay = self._calculate_delay(attempt)

                self.logger.warning("Retry attempt failed, retrying", {
                    "retry_mechanism": self.name,
                    "function": func.__name__,
                    "attempt": attempt + 1,
                    "total_attempts": self.config.max_attempts,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "retry_delay_seconds": delay
                })

                time.sleep(delay)

        # All attempts failed
        raise RetryExhaustedError(
            f"All retry attempts failed for {func.__name__} in {self.name}"
        ) from last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay_seconds

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay_seconds * (attempt + 1)

        elif self.config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)
            jitter = base_delay * self.config.jitter_range * (random.random() - 0.5)
            delay = base_delay + jitter

        else:
            delay = self.config.base_delay_seconds

        # Ensure delay doesn't exceed maximum
        return min(delay, self.config.max_delay_seconds)


class Bulkhead:
    """Bulkhead pattern for resource isolation."""

    def __init__(self, name: str, config: BulkheadConfig):
        self.name = name
        self.config = config
        self.logger = get_core_logger()

        # Semaphore for limiting concurrent calls
        self.semaphore = threading.Semaphore(config.max_concurrent_calls)
        self.active_calls = 0
        self.total_calls = 0
        self.rejected_calls = 0
        self.queue_size = 0

        self._lock = threading.Lock()

    @contextmanager
    def call(self):
        """Context manager for making calls through bulkhead."""
        # Try to acquire semaphore with timeout
        acquired = self.semaphore.acquire(timeout=self.config.queue_timeout_seconds)

        if not acquired:
            with self._lock:
                self.rejected_calls += 1

            self.logger.warning("Bulkhead capacity exceeded", {
                "bulkhead": self.name,
                "active_calls": self.active_calls,
                "max_concurrent": self.config.max_concurrent_calls,
                "rejected_calls": self.rejected_calls
            })

            raise BulkheadFullError(f"Bulkhead '{self.name}' at capacity")

        try:
            with self._lock:
                self.active_calls += 1
                self.total_calls += 1

            yield

        finally:
            with self._lock:
                self.active_calls -= 1
            self.semaphore.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            utilization_percent = (
                (self.active_calls / self.config.max_concurrent_calls * 100)
                if self.config.max_concurrent_calls > 0 else 0
            )

            rejection_rate = (
                (self.rejected_calls / self.total_calls * 100)
                if self.total_calls > 0 else 0
            )

            return {
                "name": self.name,
                "active_calls": self.active_calls,
                "total_calls": self.total_calls,
                "rejected_calls": self.rejected_calls,
                "utilization_percent": utilization_percent,
                "rejection_rate_percent": rejection_rate,
                "config": {
                    "max_concurrent_calls": self.config.max_concurrent_calls,
                    "max_queue_size": self.config.max_queue_size,
                    "queue_timeout_seconds": self.config.queue_timeout_seconds
                }
            }


class ResilienceManager:
    """Central manager for resilience patterns."""

    def __init__(self):
        self.logger = get_core_logger()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}

    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create a new circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig()

        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker

        self.logger.info("Circuit breaker created", {
            "circuit_breaker": name,
            "config": {
                "failure_threshold": config.failure_threshold,
                "success_threshold": config.success_threshold,
                "timeout_duration_seconds": config.timeout_duration_seconds
            }
        })

        return circuit_breaker

    def create_retry_mechanism(self, name: str, config: Optional[RetryConfig] = None) -> RetryMechanism:
        """Create a new retry mechanism."""
        if config is None:
            config = RetryConfig()

        retry_mechanism = RetryMechanism(name, config)
        self.retry_mechanisms[name] = retry_mechanism

        self.logger.info("Retry mechanism created", {
            "retry_mechanism": name,
            "config": {
                "max_attempts": config.max_attempts,
                "strategy": config.strategy.value,
                "base_delay_seconds": config.base_delay_seconds
            }
        })

        return retry_mechanism

    def create_bulkhead(self, name: str, config: Optional[BulkheadConfig] = None) -> Bulkhead:
        """Create a new bulkhead."""
        if config is None:
            config = BulkheadConfig()

        bulkhead = Bulkhead(name, config)
        self.bulkheads[name] = bulkhead

        self.logger.info("Bulkhead created", {
            "bulkhead": name,
            "config": {
                "max_concurrent_calls": config.max_concurrent_calls,
                "max_queue_size": config.max_queue_size,
                "queue_timeout_seconds": config.queue_timeout_seconds
            }
        })

        return bulkhead

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def get_retry_mechanism(self, name: str) -> Optional[RetryMechanism]:
        """Get existing retry mechanism by name."""
        return self.retry_mechanisms.get(name)

    def get_bulkhead(self, name: str) -> Optional[Bulkhead]:
        """Get existing bulkhead by name."""
        return self.bulkheads.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all resilience components."""
        return {
            "circuit_breakers": {
                name: cb.get_stats() for name, cb in self.circuit_breakers.items()
            },
            "bulkheads": {
                name: b.get_stats() for name, b in self.bulkheads.items()
            },
            "summary": {
                "total_circuit_breakers": len(self.circuit_breakers),
                "total_retry_mechanisms": len(self.retry_mechanisms),
                "total_bulkheads": len(self.bulkheads),
                "open_circuit_breakers": [
                    name for name, cb in self.circuit_breakers.items()
                    if cb.state == CircuitState.OPEN
                ]
            }
        }


# Global resilience manager
_resilience_manager: Optional[ResilienceManager] = None

def get_resilience_manager() -> ResilienceManager:
    """Get or create the global resilience manager."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


# Convenience decorators
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker pattern to a function."""
    def decorator(func: Callable) -> Callable:
        manager = get_resilience_manager()
        cb = manager.create_circuit_breaker(name, config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            with cb.call():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def retry(name: str, config: Optional[RetryConfig] = None):
    """Decorator to apply retry pattern to a function."""
    manager = get_resilience_manager()
    retry_mechanism = manager.create_retry_mechanism(name, config)
    return retry_mechanism


def bulkhead(name: str, config: Optional[BulkheadConfig] = None):
    """Decorator to apply bulkhead pattern to a function."""
    def decorator(func: Callable) -> Callable:
        manager = get_resilience_manager()
        bh = manager.create_bulkhead(name, config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            with bh.call():
                return func(*args, **kwargs)
        return wrapper
    return decorator
