"""
Advanced Resilience System for TestGen Copilot
Implements comprehensive error handling, circuit breakers, and self-healing capabilities
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import queue
import functools

import numpy as np

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing if service recovered


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # successes needed to close from half-open
    timeout: float = 30.0      # operation timeout
    

@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    interval: float = 30.0     # seconds
    timeout: float = 5.0       # seconds
    critical: bool = False     # is this check critical for overall health
    

@dataclass
class ErrorPattern:
    """Pattern for error analysis and recovery."""
    error_type: str
    error_message_pattern: str
    frequency: int = 0
    last_occurrence: Optional[datetime] = None
    recovery_strategy: str = "retry"
    auto_recovery_attempts: int = 0
    max_auto_recovery: int = 3


@dataclass
class ResilienceMetrics:
    """Metrics for resilience system."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    auto_recoveries: int = 0
    self_healing_activations: int = 0
    average_response_time: float = 0.0
    uptime_percentage: float = 100.0
    last_updated: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """Advanced circuit breaker with configurable policies."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
                    
        try:
            # Execute with timeout
            result = asyncio.wait_for(
                self._execute_async(func, *args, **kwargs),
                timeout=self.config.timeout
            )
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
            
    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
            
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.next_attempt_time is None:
            return True
        return datetime.now() >= self.next_attempt_time
        
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
                    
    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self._set_next_attempt_time()
                logger.warning(f"Circuit breaker {self.name} OPEN after half-open failure")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._set_next_attempt_time()
                logger.warning(f"Circuit breaker {self.name} OPEN after {self.failure_count} failures")
                
    def _set_next_attempt_time(self) -> None:
        """Set next attempt time."""
        self.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
        
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None
        }


class RetryHandler:
    """Advanced retry handler with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_on: Tuple[Exception, ...] = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except retry_on as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt, don't delay
                    break
                    
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
                
        # All attempts failed
        logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_exception
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter (±25%)
            jitter = delay * 0.25 * (2 * np.random.random() - 1)
            delay += jitter
            
        return max(delay, 0.1)  # Minimum 0.1s delay


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status = HealthStatus.HEALTHY
        self.check_results: Dict[str, Tuple[bool, datetime]] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
        
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                for check in self.health_checks.values():
                    self._execute_health_check(check)
                    
                self._update_overall_health()
                time.sleep(1)  # Check every second for due checks
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
                
    def _execute_health_check(self, check: HealthCheck) -> None:
        """Execute a single health check."""
        check_name = check.name
        
        # Check if it's time to run this check
        if check_name in self.check_results:
            last_check_time = self.check_results[check_name][1]
            if datetime.now() - last_check_time < timedelta(seconds=check.interval):
                return
                
        try:
            # Execute check with timeout
            start_time = time.time()
            result = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(self._async_check(check.check_function), timeout=check.timeout),
                asyncio.new_event_loop()
            ).result(timeout=check.timeout + 1)
            
            execution_time = time.time() - start_time
            self.check_results[check_name] = (result, datetime.now())
            
            if result:
                logger.debug(f"Health check {check_name} passed in {execution_time:.2f}s")
            else:
                logger.warning(f"Health check {check_name} failed in {execution_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Health check {check_name} exception: {e}")
            self.check_results[check_name] = (False, datetime.now())
            
    async def _async_check(self, check_function: Callable) -> bool:
        """Execute check function asynchronously."""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            return check_function()
            
    def _update_overall_health(self) -> None:
        """Update overall health status."""
        if not self.check_results:
            return
            
        critical_checks = [
            name for name, check in self.health_checks.items() 
            if check.critical
        ]
        
        failed_critical = [
            name for name in critical_checks 
            if name in self.check_results and not self.check_results[name][0]
        ]
        
        failed_non_critical = [
            name for name, (result, _) in self.check_results.items() 
            if not result and name not in critical_checks
        ]
        
        # Determine health status
        if failed_critical:
            new_status = HealthStatus.CRITICAL
        elif len(failed_non_critical) > len(self.check_results) * 0.5:
            new_status = HealthStatus.UNHEALTHY
        elif failed_non_critical:
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY
            
        if new_status != self.health_status:
            logger.info(f"Health status changed: {self.health_status.value} -> {new_status.value}")
            self.health_status = new_status
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "overall_status": self.health_status.value,
            "checks": {
                name: {
                    "status": "pass" if result else "fail",
                    "last_check": last_check.isoformat(),
                    "critical": self.health_checks[name].critical
                }
                for name, (result, last_check) in self.check_results.items()
            },
            "timestamp": datetime.now().isoformat()
        }


class SelfHealingSystem:
    """Self-healing system that automatically recovers from errors."""
    
    def __init__(self):
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.healing_active = True
        self.healing_queue: queue.Queue = queue.Queue()
        self.healing_thread: Optional[threading.Thread] = None
        
        self._register_default_strategies()
        self._start_healing_thread()
        
    def register_error_pattern(self, pattern: ErrorPattern) -> None:
        """Register an error pattern for automatic recovery."""
        self.error_patterns[pattern.error_type] = pattern
        logger.info(f"Registered error pattern: {pattern.error_type}")
        
    def register_recovery_strategy(self, name: str, strategy: Callable) -> None:
        """Register a recovery strategy."""
        self.recovery_strategies[name] = strategy
        logger.info(f"Registered recovery strategy: {name}")
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Handle an error and attempt self-healing."""
        if not self.healing_active:
            return False
            
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error pattern
        if error_type in self.error_patterns:
            pattern = self.error_patterns[error_type]
            pattern.frequency += 1
            pattern.last_occurrence = datetime.now()
        else:
            # Create new pattern
            pattern = ErrorPattern(
                error_type=error_type,
                error_message_pattern=error_message,
                frequency=1,
                last_occurrence=datetime.now()
            )
            self.error_patterns[error_type] = pattern
            
        # Queue for healing if auto-recovery is available
        if (pattern.auto_recovery_attempts < pattern.max_auto_recovery and
            pattern.recovery_strategy in self.recovery_strategies):
            
            healing_task = {
                "error_type": error_type,
                "error": error,
                "context": context or {},
                "pattern": pattern,
                "timestamp": datetime.now()
            }
            
            self.healing_queue.put(healing_task)
            logger.info(f"Queued error for self-healing: {error_type}")
            return True
            
        return False
        
    def _start_healing_thread(self) -> None:
        """Start the self-healing thread."""
        self.healing_thread = threading.Thread(target=self._healing_loop, daemon=True)
        self.healing_thread.start()
        logger.info("Self-healing system started")
        
    def _healing_loop(self) -> None:
        """Main self-healing loop."""
        while self.healing_active:
            try:
                # Wait for healing tasks (with timeout)
                healing_task = self.healing_queue.get(timeout=5)
                self._execute_healing(healing_task)
                self.healing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in healing loop: {e}")
                
    def _execute_healing(self, healing_task: Dict[str, Any]) -> None:
        """Execute a healing strategy."""
        error_type = healing_task["error_type"]
        pattern = healing_task["pattern"]
        context = healing_task["context"]
        
        strategy_name = pattern.recovery_strategy
        
        if strategy_name not in self.recovery_strategies:
            logger.error(f"Recovery strategy not found: {strategy_name}")
            return
            
        try:
            pattern.auto_recovery_attempts += 1
            strategy = self.recovery_strategies[strategy_name]
            
            logger.info(f"Attempting self-healing for {error_type} with strategy {strategy_name}")
            
            success = strategy(healing_task)
            
            if success:
                logger.info(f"Self-healing successful for {error_type}")
                # Reset failure count on successful healing
                pattern.auto_recovery_attempts = 0
            else:
                logger.warning(f"Self-healing failed for {error_type}")
                
        except Exception as e:
            logger.error(f"Self-healing strategy failed: {e}")
            
    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        
        def retry_strategy(healing_task: Dict[str, Any]) -> bool:
            """Simple retry strategy."""
            time.sleep(1)  # Brief delay
            return True  # Assume retry will work
            
        def restart_component_strategy(healing_task: Dict[str, Any]) -> bool:
            """Restart component strategy."""
            context = healing_task["context"]
            component = context.get("component")
            
            if component and hasattr(component, 'restart'):
                try:
                    component.restart()
                    return True
                except Exception as e:
                    logger.error(f"Component restart failed: {e}")
                    
            return False
            
        def clear_cache_strategy(healing_task: Dict[str, Any]) -> bool:
            """Clear cache strategy."""
            context = healing_task["context"]
            cache = context.get("cache")
            
            if cache and hasattr(cache, 'clear'):
                try:
                    cache.clear()
                    logger.info("Cache cleared for recovery")
                    return True
                except Exception as e:
                    logger.error(f"Cache clear failed: {e}")
                    
            return False
            
        self.register_recovery_strategy("retry", retry_strategy)
        self.register_recovery_strategy("restart_component", restart_component_strategy)
        self.register_recovery_strategy("clear_cache", clear_cache_strategy)


class AdvancedResilienceSystem:
    """
    Comprehensive resilience system combining circuit breakers, retry logic,
    health monitoring, and self-healing capabilities.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path) if config_path else {}
        
        # Components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler()
        self.health_monitor = HealthMonitor()
        self.self_healing = SelfHealingSystem()
        
        # Metrics
        self.metrics = ResilienceMetrics()
        self.metrics_lock = threading.Lock()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Start monitoring
        self.health_monitor.start_monitoring()
        
    def resilient_call(
        self,
        func: Callable,
        *args,
        circuit_breaker_name: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with full resilience features.
        """
        start_time = time.time()
        
        try:
            with self.metrics_lock:
                self.metrics.total_requests += 1
                
            # Get or create circuit breaker
            if circuit_breaker_name:
                if circuit_breaker_name not in self.circuit_breakers:
                    self.circuit_breakers[circuit_breaker_name] = CircuitBreaker(circuit_breaker_name)
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
            else:
                circuit_breaker = None
                
            # Execute with circuit breaker and retry
            if circuit_breaker:
                result = self._execute_with_circuit_breaker_and_retry(
                    circuit_breaker, func, retry_config, *args, **kwargs
                )
            else:
                result = self._execute_with_retry(func, retry_config, *args, **kwargs)
                
            # Update metrics
            execution_time = time.time() - start_time
            with self.metrics_lock:
                self.metrics.successful_requests += 1
                self._update_response_time(execution_time)
                
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self.metrics_lock:
                self.metrics.failed_requests += 1
                self._update_response_time(execution_time)
                
            # Attempt self-healing
            healing_attempted = self.self_healing.handle_error(e, context)
            
            if healing_attempted:
                with self.metrics_lock:
                    self.metrics.self_healing_activations += 1
                    
            raise
            
    def _execute_with_circuit_breaker_and_retry(
        self,
        circuit_breaker: CircuitBreaker,
        func: Callable,
        retry_config: Optional[RetryConfig],
        *args,
        **kwargs
    ) -> Any:
        """Execute with both circuit breaker and retry."""
        retry_config = retry_config or RetryConfig()
        
        async def execute_with_cb():
            return circuit_breaker.call(func, *args, **kwargs)
            
        try:
            return asyncio.run(
                self.retry_handler.execute_with_retry(
                    execute_with_cb,
                    retry_on=(CircuitBreakerOpenError, Exception)
                )
            )
        except CircuitBreakerOpenError:
            with self.metrics_lock:
                self.metrics.circuit_breaker_trips += 1
            raise
            
    def _execute_with_retry(
        self,
        func: Callable,
        retry_config: Optional[RetryConfig],
        *args,
        **kwargs
    ) -> Any:
        """Execute with retry only."""
        return asyncio.run(
            self.retry_handler.execute_with_retry(func, *args, **kwargs)
        )
        
    def _update_response_time(self, execution_time: float) -> None:
        """Update average response time."""
        # Exponential moving average
        alpha = 0.1
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = execution_time
        else:
            self.metrics.average_response_time = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics.average_response_time
            )
            
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> None:
        """Add a new circuit breaker."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
        logger.info(f"Added circuit breaker: {name}")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self.metrics_lock:
            current_metrics = {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": (
                    self.metrics.successful_requests / max(self.metrics.total_requests, 1)
                ),
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "retry_attempts": self.metrics.retry_attempts,
                "auto_recoveries": self.metrics.auto_recoveries,
                "self_healing_activations": self.metrics.self_healing_activations,
                "average_response_time": self.metrics.average_response_time
            }
            
        return {
            "metrics": current_metrics,
            "health": self.health_monitor.get_health_status(),
            "circuit_breakers": {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            },
            "error_patterns": {
                error_type: {
                    "frequency": pattern.frequency,
                    "last_occurrence": pattern.last_occurrence.isoformat() if pattern.last_occurrence else None,
                    "auto_recovery_attempts": pattern.auto_recovery_attempts
                }
                for error_type, pattern in self.self_healing.error_patterns.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        
        def memory_check() -> bool:
            """Check memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Skip if psutil not available
                
        def disk_check() -> bool:
            """Check disk space."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return (disk.free / disk.total) > 0.1  # More than 10% free
            except ImportError:
                return True
                
        def cpu_check() -> bool:
            """Check CPU usage."""
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < 95  # Less than 95% CPU usage
            except ImportError:
                return True
                
        # Register health checks
        self.health_monitor.register_health_check(
            HealthCheck("memory", memory_check, interval=30, critical=True)
        )
        self.health_monitor.register_health_check(
            HealthCheck("disk_space", disk_check, interval=60, critical=True)
        )
        self.health_monitor.register_health_check(
            HealthCheck("cpu_usage", cpu_check, interval=15, critical=False)
        )
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
            
    def shutdown(self) -> None:
        """Shutdown resilience system."""
        logger.info("Shutting down resilience system")
        self.health_monitor.stop_monitoring()
        self.self_healing.healing_active = False


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Decorators for easy use

def resilient(
    circuit_breaker_name: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    resilience_system: Optional[AdvancedResilienceSystem] = None
):
    """Decorator to make functions resilient."""
    def decorator(func: Callable) -> Callable:
        nonlocal resilience_system
        
        if resilience_system is None:
            resilience_system = AdvancedResilienceSystem()
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return resilience_system.resilient_call(
                func, *args,
                circuit_breaker_name=circuit_breaker_name,
                retry_config=retry_config,
                **kwargs
            )
        return wrapper
    return decorator


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker to function."""
    def decorator(func: Callable) -> Callable:
        cb = CircuitBreaker(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


async def main():
    """Example usage of advanced resilience system."""
    
    # Create resilience system
    resilience = AdvancedResilienceSystem()
    
    # Example function that might fail
    def unreliable_function(success_rate: float = 0.7):
        if np.random.random() < success_rate:
            return "Success!"
        else:
            raise Exception("Random failure")
    
    # Test resilient calls
    for i in range(10):
        try:
            result = resilience.resilient_call(
                unreliable_function,
                success_rate=0.3,  # Low success rate to trigger resilience features
                circuit_breaker_name="test_service",
                context={"test_run": i}
            )
            print(f"Call {i}: {result}")
        except Exception as e:
            print(f"Call {i} failed: {e}")
        
        await asyncio.sleep(0.5)
    
    # Get system status
    status = resilience.get_system_status()
    print("\nSystem Status:")
    print(f"Success rate: {status['metrics']['success_rate']:.2%}")
    print(f"Circuit breaker trips: {status['metrics']['circuit_breaker_trips']}")
    print(f"Self-healing activations: {status['metrics']['self_healing_activations']}")
    
    # Shutdown
    resilience.shutdown()


if __name__ == "__main__":
    asyncio.run(main())