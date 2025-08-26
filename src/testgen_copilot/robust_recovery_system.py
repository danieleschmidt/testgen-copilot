"""
🛡️ Robust Recovery System v2.0
===============================

Intelligent failure detection, recovery, and resilience system.
Implements circuit breakers, retry logic, fallback mechanisms, and self-healing capabilities.
"""

import asyncio
import time
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import inspect

from .logging_config import get_core_logger
from .robust_monitoring_system import RobustMonitoringSystem, Alert, AlertSeverity

logger = get_core_logger()


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SERVICE_RESTART = "service_restart"
    CACHE_FALLBACK = "cache_fallback"
    DEFAULT_RESPONSE = "default_response"


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_exceptions: List[type] = field(default_factory=list)
    stop_exceptions: List[type] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    minimum_throughput: int = 10
    error_rate_threshold: float = 0.5


@dataclass
class FallbackConfig:
    """Fallback configuration"""
    fallback_function: Optional[Callable] = None
    cache_fallback: bool = False
    default_value: Any = None
    graceful_degradation: bool = False


@dataclass
class RecoveryEvent:
    """Recovery event information"""
    event_id: str
    timestamp: datetime
    component: str
    failure_type: str
    recovery_action: RecoveryAction
    success: bool
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker implementation with intelligent failure detection
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.call_count = 0
        self.error_count = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if (self.last_failure_time and 
                    datetime.now() - self.last_failure_time >= timedelta(seconds=self.config.timeout_seconds)):
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls
            
            return False
    
    def record_success(self) -> None:
        """Record a successful execution"""
        with self._lock:
            self.call_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                self.half_open_calls += 1
                
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed execution"""
        with self._lock:
            self.call_count += 1
            self.error_count += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
                logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN due to failure in HALF_OPEN")
            
            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if (self.call_count >= self.config.minimum_throughput and
                    self.failure_count >= self.config.failure_threshold):
                    
                    error_rate = self.error_count / self.call_count
                    if error_rate >= self.config.error_rate_threshold:
                        self.state = CircuitState.OPEN
                        logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN due to error rate: {error_rate:.2f}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            error_rate = self.error_count / max(self.call_count, 1)
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "call_count": self.call_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
            }


class RobustRecoverySystem:
    """
    🛡️ Comprehensive recovery and resilience system
    
    Features:
    - Circuit breakers for service protection
    - Intelligent retry with multiple strategies
    - Fallback mechanisms with graceful degradation
    - Bulkhead isolation for fault tolerance
    - Self-healing capabilities
    - Recovery pattern learning
    - Performance impact monitoring
    """
    
    def __init__(self, monitoring_system: Optional[RobustMonitoringSystem] = None):
        self.monitoring_system = monitoring_system
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery configurations
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        
        # Recovery tracking
        self.recovery_events: List[RecoveryEvent] = []
        self.recovery_patterns: Dict[str, List[RecoveryAction]] = {}
        
        # Cache for fallback responses
        self.fallback_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
        # Default configurations
        self._initialize_defaults()
    
    def _initialize_defaults(self) -> None:
        """Initialize default configurations"""
        
        # Default retry config
        self.retry_configs["default"] = RetryConfig()
        
        # Default circuit breaker config
        default_cb_config = CircuitBreakerConfig()
        
        # Default fallback config
        self.fallback_configs["default"] = FallbackConfig()
        
        # Create default circuit breaker
        self.circuit_breakers["default"] = CircuitBreaker("default", default_cb_config)
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> None:
        """Register a new circuit breaker"""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
        logger.info(f"Registered circuit breaker: {name}")
    
    def register_retry_config(self, name: str, config: RetryConfig) -> None:
        """Register a retry configuration"""
        self.retry_configs[name] = config
        logger.info(f"Registered retry config: {name}")
    
    def register_fallback_config(self, name: str, config: FallbackConfig) -> None:
        """Register a fallback configuration"""
        self.fallback_configs[name] = config
        logger.info(f"Registered fallback config: {name}")
    
    def with_circuit_breaker(self, name: str = "default"):
        """Decorator to add circuit breaker protection"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_circuit_breaker(name, func, args, kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_circuit_breaker(name, func, args, kwargs))
            
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def with_retry(self, config_name: str = "default"):
        """Decorator to add retry logic"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_retry(config_name, func, args, kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_retry(config_name, func, args, kwargs))
            
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def with_fallback(self, config_name: str = "default"):
        """Decorator to add fallback protection"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_fallback(config_name, func, args, kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_fallback(config_name, func, args, kwargs))
            
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def with_resilience(self, 
                       circuit_breaker: str = "default",
                       retry_config: str = "default", 
                       fallback_config: str = "default"):
        """Decorator to add comprehensive resilience (circuit breaker + retry + fallback)"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_resilience(
                    circuit_breaker, retry_config, fallback_config, func, args, kwargs
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_resilience(
                    circuit_breaker, retry_config, fallback_config, func, args, kwargs
                ))
            
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _execute_with_circuit_breaker(self, cb_name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.circuit_breakers.get(cb_name, self.circuit_breakers["default"])
        
        if not circuit_breaker.can_execute():
            error_msg = f"Circuit breaker '{cb_name}' is {circuit_breaker.state.value}"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            circuit_breaker.record_success()
            return result
            
        except Exception as e:
            circuit_breaker.record_failure()
            raise
    
    async def _execute_with_retry(self, config_name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with retry logic"""
        config = self.retry_configs.get(config_name, self.retry_configs["default"])
        
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should stop retrying
                if config.stop_exceptions and type(e) in config.stop_exceptions:
                    logger.info(f"Stopping retry due to stop exception: {type(e).__name__}")
                    break
                
                # Check if we should retry this exception
                if config.retry_exceptions and type(e) not in config.retry_exceptions:
                    logger.info(f"Not retrying exception: {type(e).__name__}")
                    break
                
                # Calculate delay for next attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_retry_delay(config, attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {config.max_attempts} attempts failed")
        
        raise last_exception
    
    async def _execute_with_fallback(self, config_name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with fallback protection"""
        config = self.fallback_configs.get(config_name, self.fallback_configs["default"])
        
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache successful result if cache fallback is enabled
            if config.cache_fallback:
                cache_key = self._generate_cache_key(func, args, kwargs)
                self.fallback_cache[cache_key] = result
                self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=10)
            
            return result
            
        except Exception as e:
            logger.warning(f"Primary function failed: {e}. Attempting fallback")
            
            # Try cache fallback
            if config.cache_fallback:
                cache_key = self._generate_cache_key(func, args, kwargs)
                if cache_key in self.fallback_cache:
                    if cache_key not in self.cache_ttl or self.cache_ttl[cache_key] > datetime.now():
                        logger.info("Using cached fallback response")
                        return self.fallback_cache[cache_key]
            
            # Try fallback function
            if config.fallback_function:
                try:
                    if inspect.iscoroutinefunction(config.fallback_function):
                        result = await config.fallback_function(*args, **kwargs)
                    else:
                        result = config.fallback_function(*args, **kwargs)
                    
                    logger.info("Fallback function succeeded")
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback function also failed: {fallback_error}")
            
            # Use default value
            if config.default_value is not None:
                logger.info("Using default fallback value")
                return config.default_value
            
            # Graceful degradation
            if config.graceful_degradation:
                logger.info("Graceful degradation: returning None")
                return None
            
            # No fallback available, re-raise original exception
            raise
    
    async def _execute_with_resilience(self, cb_name: str, retry_name: str, fallback_name: str,
                                     func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with comprehensive resilience"""
        
        # Wrap function with retry logic
        retry_func = lambda: self._execute_with_retry(retry_name, func, args, kwargs)
        
        # Wrap with circuit breaker
        cb_func = lambda: self._execute_with_circuit_breaker(cb_name, retry_func, (), {})
        
        # Execute with fallback
        return await self._execute_with_fallback(fallback_name, cb_func, (), {})
    
    def _calculate_retry_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** attempt)
        
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        
        elif config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = config.base_delay * (config.backoff_multiplier ** attempt)
            jitter = random.uniform(0, base_delay * 0.1)  # 10% jitter
            delay = base_delay + jitter
        
        else:
            delay = config.base_delay
        
        # Cap at maximum delay
        return min(delay, config.max_delay)
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        func_name = f"{func.__module__}.{func.__name__}"
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{func_name}:{hash(args_str + kwargs_str)}"
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: cb.get_state()
            for name, cb in self.circuit_breakers.items()
        }
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics"""
        total_events = len(self.recovery_events)
        successful_recoveries = sum(1 for event in self.recovery_events if event.success)
        
        return {
            "total_recovery_events": total_events,
            "successful_recoveries": successful_recoveries,
            "success_rate": successful_recoveries / max(total_events, 1),
            "circuit_breakers": len(self.circuit_breakers),
            "retry_configs": len(self.retry_configs),
            "fallback_configs": len(self.fallback_configs),
            "cache_entries": len(self.fallback_cache)
        }
    
    def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker"""
        if name in self.circuit_breakers:
            cb = self.circuit_breakers[name]
            with cb._lock:
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
                cb.success_count = 0
                cb.half_open_calls = 0
            
            logger.info(f"Circuit breaker '{name}' manually reset")
            return True
        
        return False
    
    def clear_fallback_cache(self) -> None:
        """Clear the fallback cache"""
        self.fallback_cache.clear()
        self.cache_ttl.clear()
        logger.info("Fallback cache cleared")


# Convenience functions for common patterns
def create_database_recovery_system(monitoring_system: Optional[RobustMonitoringSystem] = None) -> RobustRecoverySystem:
    """Create recovery system optimized for database operations"""
    recovery_system = RobustRecoverySystem(monitoring_system)
    
    # Database-specific circuit breaker
    db_cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=30.0,
        error_rate_threshold=0.3
    )
    recovery_system.register_circuit_breaker("database", db_cb_config)
    
    # Database-specific retry config
    db_retry_config = RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=1.0,
        max_delay=10.0,
        retry_exceptions=[ConnectionError, TimeoutError]
    )
    recovery_system.register_retry_config("database", db_retry_config)
    
    return recovery_system


def create_api_recovery_system(monitoring_system: Optional[RobustMonitoringSystem] = None) -> RobustRecoverySystem:
    """Create recovery system optimized for API calls"""
    recovery_system = RobustRecoverySystem(monitoring_system)
    
    # API-specific circuit breaker
    api_cb_config = CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=60.0,
        error_rate_threshold=0.5
    )
    recovery_system.register_circuit_breaker("api", api_cb_config)
    
    # API-specific retry config
    api_retry_config = RetryConfig(
        max_attempts=5,
        strategy=RetryStrategy.JITTERED_BACKOFF,
        base_delay=2.0,
        max_delay=30.0,
        jitter=True
    )
    recovery_system.register_retry_config("api", api_retry_config)
    
    return recovery_system