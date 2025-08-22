"""Advanced performance optimization and scaling mechanisms."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import hashlib
import multiprocessing
import pickle
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from .logging_config import get_core_logger

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance measurement and tracking."""
    operation_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000


class AdvancedCache:
    """High-performance caching with TTL, LRU eviction, and statistics."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self.access_order: deque = deque()  # For LRU tracking
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        self.logger = get_core_logger()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU updating."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                value, expiry_time = self.cache[key]
                
                # Check if expired
                if current_time > expiry_time:
                    del self.cache[key]
                    self._remove_from_access_order(key)
                    self.misses += 1
                    return None
                
                # Update LRU order
                self._update_access_order(key)
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        with self.lock:
            ttl = ttl or self.default_ttl
            expiry_time = time.time() + ttl
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_from_access_order(key)
            
            # Evict if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (value, expiry_time)
            self.access_order.append(key)
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self._remove_from_access_order(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate_percent": hit_rate,
                "total_requests": total_requests
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
                self.evictions += 1
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order for key."""
        self._remove_from_access_order(key)
        self.access_order.append(key)
    
    def _remove_from_access_order(self, key: str) -> None:
        """Remove key from access order tracking."""
        try:
            self.access_order.remove(key)
        except ValueError:
            pass  # Key not in access order


class AsyncBatchProcessor:
    """Asynchronous batch processing with configurable concurrency."""
    
    def __init__(self, max_concurrent: int = None, batch_size: int = 10):
        self.max_concurrent = max_concurrent or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.batch_size = batch_size
        self.logger = get_core_logger()
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items in batches with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []
        
        async def process_item_with_semaphore(item: Any, index: int) -> Tuple[int, Any]:
            async with semaphore:
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(item)
                else:
                    # Run CPU-bound function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, processor_func, item)
                
                if progress_callback:
                    progress_callback(index + 1, len(items))
                
                return index, result
        
        # Create tasks for all items
        tasks = [
            asyncio.create_task(process_item_with_semaphore(item, i))
            for i, item in enumerate(items)
        ]
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by original index and extract values
        successful_results = []
        errors = []
        
        for result in completed_results:
            if isinstance(result, Exception):
                errors.append(result)
                successful_results.append(None)
            else:
                index, value = result
                successful_results.append((index, value))
        
        # Sort by index and extract values
        successful_results.sort(key=lambda x: x[0] if x else float('inf'))
        final_results = [value for index, value in successful_results if value is not None]
        
        if errors:
            self.logger.warning(f"Batch processing had {len(errors)} errors", {
                "error_count": len(errors),
                "success_count": len(final_results),
                "total_items": len(items)
            })
        
        return final_results


class ConnectionPool:
    """Generic connection pool for managing expensive resources."""
    
    def __init__(
        self,
        factory_func: Callable[[], Any],
        max_connections: int = 10,
        min_connections: int = 2,
        idle_timeout: float = 300  # 5 minutes
    ):
        self.factory_func = factory_func
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.idle_timeout = idle_timeout
        
        self.pool: deque = deque()
        self.active_connections: Set[Any] = set()
        self.lock = threading.RLock()
        self.created_count = 0
        
        self.logger = get_core_logger()
        
        # Pre-populate with minimum connections
        self._populate_pool()
    
    def _populate_pool(self) -> None:
        """Create minimum number of connections."""
        for _ in range(self.min_connections):
            try:
                conn = self.factory_func()
                self.pool.append((conn, time.time()))
                self.created_count += 1
            except Exception as e:
                self.logger.error(f"Failed to create connection: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return."""
        conn = self._acquire_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)
    
    def _acquire_connection(self) -> Any:
        """Acquire connection from pool or create new one."""
        with self.lock:
            # Try to get from pool
            while self.pool:
                conn, created_time = self.pool.popleft()
                
                # Check if connection is still valid (not timed out)
                if time.time() - created_time < self.idle_timeout:
                    self.active_connections.add(conn)
                    return conn
                else:
                    # Connection timed out, close it
                    self._close_connection(conn)
            
            # No available connections, create new one if under limit
            if len(self.active_connections) < self.max_connections:
                try:
                    conn = self.factory_func()
                    self.active_connections.add(conn)
                    self.created_count += 1
                    return conn
                except Exception as e:
                    self.logger.error(f"Failed to create new connection: {e}")
                    raise
            
            # Pool exhausted, wait for connection to be returned
            raise RuntimeError(f"Connection pool exhausted (max: {self.max_connections})")
    
    def _return_connection(self, conn: Any) -> None:
        """Return connection to pool."""
        with self.lock:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                
                # Return to pool if under max pool size
                if len(self.pool) < self.max_connections:
                    self.pool.append((conn, time.time()))
                else:
                    # Pool full, close connection
                    self._close_connection(conn)
    
    def _close_connection(self, conn: Any) -> None:
        """Close connection and clean up."""
        try:
            if hasattr(conn, 'close'):
                conn.close()
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self.lock:
            return {
                "pool_size": len(self.pool),
                "active_connections": len(self.active_connections),
                "max_connections": self.max_connections,
                "total_created": self.created_count,
                "utilization_percent": (len(self.active_connections) / self.max_connections) * 100
            }
    
    def close_all(self) -> None:
        """Close all connections and clean up pool."""
        with self.lock:
            # Close pooled connections
            while self.pool:
                conn, _ = self.pool.popleft()
                self._close_connection(conn)
            
            # Close active connections
            for conn in list(self.active_connections):
                self._close_connection(conn)
            
            self.active_connections.clear()
            self.logger.info("Connection pool closed")


class PerformanceProfiler:
    """Comprehensive performance profiling and monitoring."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.logger = get_core_logger()
    
    @contextmanager
    def profile_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                memory_usage_mb=end_memory - start_memory,
                cpu_percent=0.0,  # Could integrate psutil for CPU monitoring
                metadata=metadata or {}
            )
            
            self.metrics.append(metrics)
            self.operation_stats[operation_name].append(duration)
            
            self.logger.debug(f"Operation '{operation_name}' completed in {duration:.3f}s")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for specific operation."""
        durations = self.operation_stats.get(operation_name, [])
        
        if not durations:
            return {}
        
        durations.sort()
        count = len(durations)
        
        return {
            "count": count,
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "avg_seconds": sum(durations) / count,
            "median_seconds": durations[count // 2],
            "p95_seconds": durations[int(count * 0.95)] if count > 20 else max(durations),
            "p99_seconds": durations[int(count * 0.99)] if count > 100 else max(durations),
            "total_time_seconds": sum(durations)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return "No performance metrics collected."
        
        report_lines = ["Performance Analysis Report", "=" * 50, ""]
        
        # Overall statistics
        total_operations = len(self.metrics)
        total_time = sum(m.duration_seconds for m in self.metrics)
        avg_time = total_time / total_operations
        
        report_lines.extend([
            f"Total Operations: {total_operations}",
            f"Total Time: {total_time:.3f}s",
            f"Average Time: {avg_time:.3f}s",
            ""
        ])
        
        # Per-operation statistics
        report_lines.append("Operation Statistics:")
        report_lines.append("-" * 30)
        
        for operation_name in sorted(self.operation_stats.keys()):
            stats = self.get_operation_stats(operation_name)
            report_lines.extend([
                f"Operation: {operation_name}",
                f"  Count: {stats['count']}",
                f"  Average: {stats['avg_seconds']:.3f}s",
                f"  Min: {stats['min_seconds']:.3f}s",
                f"  Max: {stats['max_seconds']:.3f}s",
                f"  P95: {stats['p95_seconds']:.3f}s",
                ""
            ])
        
        # Slowest operations
        slowest_metrics = sorted(self.metrics, key=lambda m: m.duration_seconds, reverse=True)[:10]
        
        if slowest_metrics:
            report_lines.extend([
                "Slowest Operations:",
                "-" * 20
            ])
            
            for i, metric in enumerate(slowest_metrics, 1):
                report_lines.append(
                    f"{i}. {metric.operation_name}: {metric.duration_seconds:.3f}s"
                )
        
        return "\\n".join(report_lines)
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self.operation_stats.clear()
        self.logger.info("Performance metrics cleared")


class AutoScaler:
    """Automatic scaling based on load metrics."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.load_history: deque = deque(maxlen=60)  # 1 minute of history
        self.scale_cooldown = 30  # seconds
        self.last_scale_time = 0
        
        self.logger = get_core_logger()
    
    def record_load_metric(self, load_percent: float) -> None:
        """Record current load percentage."""
        self.load_history.append((time.time(), load_percent))
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed."""
        if not self.load_history or self.current_instances >= self.max_instances:
            return False
        
        # Check if cooldown period has passed
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Calculate average load over last 5 measurements
        recent_loads = list(self.load_history)[-5:]
        avg_load = sum(load for _, load in recent_loads) / len(recent_loads)
        
        # Scale up if average load > 80%
        return avg_load > 80.0
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is needed."""
        if not self.load_history or self.current_instances <= self.min_instances:
            return False
        
        # Check cooldown
        if time.time() - self.last_scale_time < self.scale_cooldown * 2:  # Longer cooldown for scale down
            return False
        
        # Calculate average load over last 10 measurements
        recent_loads = list(self.load_history)[-10:]
        avg_load = sum(load for _, load in recent_loads) / len(recent_loads)
        
        # Scale down if average load < 30%
        return avg_load < 30.0
    
    def execute_scaling(self) -> Optional[str]:
        """Execute scaling decision if needed."""
        if self.should_scale_up():
            old_instances = self.current_instances
            self.current_instances = min(self.current_instances + 1, self.max_instances)
            self.last_scale_time = time.time()
            
            self.logger.info(f"Scaled up from {old_instances} to {self.current_instances} instances")
            return f"scale_up_{old_instances}_to_{self.current_instances}"
        
        elif self.should_scale_down():
            old_instances = self.current_instances
            self.current_instances = max(self.current_instances - 1, self.min_instances)
            self.last_scale_time = time.time()
            
            self.logger.info(f"Scaled down from {old_instances} to {self.current_instances} instances")
            return f"scale_down_{old_instances}_to_{self.current_instances}"
        
        return None


# Global instances for easy access
performance_cache = AdvancedCache(max_size=5000, default_ttl=3600)
batch_processor = AsyncBatchProcessor()
performance_profiler = PerformanceProfiler()
auto_scaler = AutoScaler()


# Decorator utilities
def cached(ttl: float = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try cache first
            result = performance_cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            performance_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def profiled(operation_name: Optional[str] = None):
    """Decorator for automatic performance profiling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with performance_profiler.profile_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Example of cached function
    @cached(ttl=60)
    def expensive_computation(n: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return n * n
    
    # Example of profiled function
    @profiled("matrix_multiplication")
    def matrix_multiply(size: int) -> List[List[int]]:
        matrix = [[i * j for j in range(size)] for i in range(size)]
        return matrix
    
    print("Testing performance optimizations...")
    
    # Test caching
    start = time.time()
    result1 = expensive_computation(10)
    first_call_time = time.time() - start
    
    start = time.time()
    result2 = expensive_computation(10)  # Should be cached
    second_call_time = time.time() - start
    
    print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")
    print(f"Cache stats: {performance_cache.get_stats()}")
    
    # Test profiling
    for size in [10, 20, 30]:
        matrix_multiply(size)
    
    print(f"\\nPerformance Report:\\n{performance_profiler.generate_report()}")