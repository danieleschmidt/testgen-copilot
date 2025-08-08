"""High-performance optimization engine for TestGen Copilot."""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from collections import defaultdict
import time
import hashlib
import pickle
import threading
from contextlib import contextmanager

from .logging_config import get_core_logger
from .monitoring import get_health_monitor
from .resilience import circuit_breaker, retry


@dataclass
class PerformanceMetrics:
    """Performance optimization metrics."""
    operation_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    concurrent_executions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_level: str = "none"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceCache:
    """High-performance multi-level caching system."""
    
    def __init__(self, max_memory_mb: int = 100, max_disk_mb: int = 500):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        
        # L1: Memory cache
        self._memory_cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, size)
        self._memory_usage = 0
        self._memory_lock = threading.RLock()
        
        # L2: Disk cache  
        self._disk_cache_dir = Path.cwd() / ".testgen_cache"
        self._disk_cache_dir.mkdir(exist_ok=True)
        self._disk_usage = 0
        self._disk_lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_writes': 0
        }
        
        self.logger = get_core_logger()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU semantics."""
        # Try L1 cache first
        with self._memory_lock:
            if key in self._memory_cache:
                value, timestamp, size = self._memory_cache[key]
                # Update timestamp for LRU
                self._memory_cache[key] = (value, time.time(), size)
                self._stats['memory_hits'] += 1
                return value
                
        # Try L2 cache
        with self._disk_lock:
            cache_file = self._disk_cache_dir / f"{self._hash_key(key)}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Promote to L1 cache
                    self._put_memory(key, value)
                    self._stats['disk_hits'] += 1
                    return value
                    
                except Exception as e:
                    self.logger.warning("Disk cache read failed", {
                        "key": key[:50],
                        "error": str(e)
                    })
                    
        self._stats['misses'] += 1
        return None
        
    def put(self, key: str, value: Any) -> None:
        """Store value in cache."""
        # Always try to put in L1 first
        if self._put_memory(key, value):
            return
            
        # If L1 is full, put in L2
        self._put_disk(key, value)
        
    def _put_memory(self, key: str, value: Any) -> bool:
        """Try to store in memory cache."""
        try:
            # Estimate size (rough approximation)
            size = len(pickle.dumps(value))
            
            with self._memory_lock:
                # Check if we have space
                if self._memory_usage + size > self.max_memory_bytes:
                    # Try to evict old entries
                    self._evict_memory_lru(size)
                    
                    # Check again after eviction
                    if self._memory_usage + size > self.max_memory_bytes:
                        return False
                        
                # Store in memory
                if key in self._memory_cache:
                    old_size = self._memory_cache[key][2]
                    self._memory_usage -= old_size
                    
                self._memory_cache[key] = (value, time.time(), size)
                self._memory_usage += size
                return True
                
        except Exception as e:
            self.logger.warning("Memory cache write failed", {
                "key": key[:50],
                "error": str(e)
            })
            return False
            
    def _put_disk(self, key: str, value: Any) -> None:
        """Store in disk cache."""
        try:
            cache_file = self._disk_cache_dir / f"{self._hash_key(key)}.cache"
            
            with self._disk_lock:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
                    
                size = cache_file.stat().st_size
                self._disk_usage += size
                self._stats['disk_writes'] += 1
                
                # Check disk usage
                if self._disk_usage > self.max_disk_bytes:
                    self._evict_disk_lru()
                    
        except Exception as e:
            self.logger.warning("Disk cache write failed", {
                "key": key[:50],
                "error": str(e)
            })
            
    def _evict_memory_lru(self, needed_space: int) -> None:
        """Evict least recently used entries from memory."""
        # Sort by timestamp (oldest first)
        items = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1][1]  # timestamp
        )
        
        freed_space = 0
        for key, (value, timestamp, size) in items:
            del self._memory_cache[key]
            self._memory_usage -= size
            freed_space += size
            self._stats['evictions'] += 1
            
            if freed_space >= needed_space:
                break
                
    def _evict_disk_lru(self) -> None:
        """Evict old entries from disk cache."""
        cache_files = list(self._disk_cache_dir.glob("*.cache"))
        
        # Sort by modification time
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        # Remove oldest half
        to_remove = len(cache_files) // 2
        for cache_file in cache_files[:to_remove]:
            try:
                size = cache_file.stat().st_size
                cache_file.unlink()
                self._disk_usage -= size
            except Exception:
                pass
                
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache file names."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = self._stats['memory_hits'] + self._stats['disk_hits']
        total_requests = total_hits + self._stats['misses']
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_rate_percent": round(hit_rate, 2),
            "memory_hits": self._stats['memory_hits'],
            "disk_hits": self._stats['disk_hits'],
            "misses": self._stats['misses'],
            "memory_usage_mb": round(self._memory_usage / 1024 / 1024, 2),
            "disk_usage_mb": round(self._disk_usage / 1024 / 1024, 2),
            "memory_entries": len(self._memory_cache),
            "evictions": self._stats['evictions'],
            "disk_writes": self._stats['disk_writes']
        }
        
    def clear(self) -> None:
        """Clear all cached data."""
        with self._memory_lock:
            self._memory_cache.clear()
            self._memory_usage = 0
            
        with self._disk_lock:
            for cache_file in self._disk_cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            self._disk_usage = 0


class ConcurrentExecutor:
    """High-performance concurrent execution manager."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() + 4))
        self.use_processes = use_processes
        self.logger = get_core_logger()
        
        # Thread pool for I/O-bound tasks
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="testgen-worker"
        )
        
        # Process pool for CPU-bound tasks
        self._process_pool = None
        if use_processes:
            self._process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(multiprocessing.cpu_count(), 8)
            )
            
        self._active_futures: Set[concurrent.futures.Future] = set()
        self._completion_callbacks: Dict[str, Callable] = {}
        
    @circuit_breaker("concurrent_execution")
    async def execute_batch(self, tasks: List[Tuple[Callable, tuple, dict]], 
                          cpu_bound: bool = False) -> List[Any]:
        """Execute batch of tasks concurrently."""
        if not tasks:
            return []
            
        executor = self._process_pool if (cpu_bound and self._process_pool) else self._thread_pool
        
        # Submit all tasks
        futures = []
        for func, args, kwargs in tasks:
            future = executor.submit(func, *args, **kwargs)
            futures.append(future)
            self._active_futures.add(future)
            
        try:
            # Wait for completion with timeout and preserve order
            results = [None] * len(tasks)
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)
                    results[i] = result
                except Exception as e:
                    self.logger.error("Task execution failed", {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    results[i] = None
                finally:
                    self._active_futures.discard(future)
                    
            return results
            
        except concurrent.futures.TimeoutError:
            self.logger.error("Batch execution timed out")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
                self._active_futures.discard(future)
            raise
            
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance tracking."""
        start_time = time.time()
        monitor = get_health_monitor()
        
        with monitor.operation_timer(operation_name):
            yield
            
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info("Operation completed", {
            "operation": operation_name,
            "duration_ms": round(duration_ms, 2),
            "performance_tracking": True
        })
        
    def shutdown(self) -> None:
        """Shutdown executor pools."""
        self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)


class AdaptiveResourceManager:
    """Adaptive resource management for optimal performance."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self._cpu_usage_history: List[float] = []
        self._memory_usage_history: List[float] = []
        self._optimal_concurrency = multiprocessing.cpu_count()
        self._adjustment_factor = 1.0
        
        # Resource thresholds
        self.cpu_threshold_high = 85.0
        self.cpu_threshold_low = 50.0
        self.memory_threshold_high = 80.0
        self.memory_threshold_low = 40.0
        
    @retry("resource_optimization")
    def optimize_concurrency(self) -> int:
        """Dynamically optimize concurrency based on resource usage."""
        try:
            import psutil
            
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Update history
            self._cpu_usage_history.append(cpu_percent)
            self._memory_usage_history.append(memory_percent)
            
            # Keep only recent history
            max_history = 20
            if len(self._cpu_usage_history) > max_history:
                self._cpu_usage_history = self._cpu_usage_history[-max_history:]
            if len(self._memory_usage_history) > max_history:
                self._memory_usage_history = self._memory_usage_history[-max_history:]
                
            # Calculate averages
            avg_cpu = sum(self._cpu_usage_history) / len(self._cpu_usage_history)
            avg_memory = sum(self._memory_usage_history) / len(self._memory_usage_history)
            
            # Adjust concurrency based on resource usage
            previous_concurrency = self._optimal_concurrency
            
            if avg_cpu > self.cpu_threshold_high or avg_memory > self.memory_threshold_high:
                # High resource usage - reduce concurrency
                self._adjustment_factor *= 0.8
                self._optimal_concurrency = max(1, int(multiprocessing.cpu_count() * self._adjustment_factor))
                
            elif avg_cpu < self.cpu_threshold_low and avg_memory < self.memory_threshold_low:
                # Low resource usage - increase concurrency
                self._adjustment_factor *= 1.1
                self._optimal_concurrency = min(
                    multiprocessing.cpu_count() * 2,
                    int(multiprocessing.cpu_count() * self._adjustment_factor)
                )
                
            if self._optimal_concurrency != previous_concurrency:
                self.logger.info("Concurrency optimized", {
                    "previous_concurrency": previous_concurrency,
                    "new_concurrency": self._optimal_concurrency,
                    "avg_cpu_percent": round(avg_cpu, 1),
                    "avg_memory_percent": round(avg_memory, 1),
                    "adjustment_factor": round(self._adjustment_factor, 2)
                })
                
            return self._optimal_concurrency
            
        except Exception as e:
            self.logger.error("Concurrency optimization failed", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return multiprocessing.cpu_count()
            
    def get_resource_recommendation(self) -> Dict[str, Any]:
        """Get performance recommendations."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            recommendations = []
            
            if cpu_percent > 90:
                recommendations.append("High CPU usage detected. Consider reducing concurrent operations.")
                
            if memory.percent > 85:
                recommendations.append("High memory usage. Consider increasing swap or reducing cache size.")
                
            if disk.percent > 90:
                recommendations.append("Low disk space. Consider cleaning cache or temporary files.")
                
            return {
                "current_resources": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "optimal_concurrency": self._optimal_concurrency,
                "adjustment_factor": self._adjustment_factor,
                "recommendations": recommendations,
                "performance_status": self._get_performance_status(cpu_percent, memory.percent)
            }
            
        except Exception as e:
            self.logger.error("Resource analysis failed", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return {"error": "Resource analysis unavailable"}
            
    def _get_performance_status(self, cpu_percent: float, memory_percent: float) -> str:
        """Determine performance status based on resource usage."""
        if cpu_percent > 90 or memory_percent > 90:
            return "critical"
        elif cpu_percent > 75 or memory_percent > 75:
            return "high"
        elif cpu_percent > 50 or memory_percent > 50:
            return "moderate"
        else:
            return "optimal"


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self.cache = PerformanceCache()
        self.executor = ConcurrentExecutor()
        self.resource_manager = AdaptiveResourceManager()
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._optimization_enabled = True
        
    def enable_optimizations(self) -> None:
        """Enable all performance optimizations."""
        self._optimization_enabled = True
        self.logger.info("Performance optimizations enabled")
        
    def disable_optimizations(self) -> None:
        """Disable performance optimizations (for debugging)."""
        self._optimization_enabled = False
        self.logger.info("Performance optimizations disabled")
        
    @contextmanager
    def optimize_operation(self, operation_name: str, cacheable: bool = True):
        """Optimize a specific operation with caching and monitoring."""
        if not self._optimization_enabled:
            yield
            return
            
        start_time = time.time()
        
        # Initialize or update metrics
        if operation_name not in self.metrics:
            self.metrics[operation_name] = PerformanceMetrics(operation_name)
            
        metrics = self.metrics[operation_name]
        metrics.total_calls += 1
        metrics.concurrent_executions += 1
        
        try:
            with self.executor.performance_context(operation_name):
                yield
                
            # Record successful execution
            duration_ms = (time.time() - start_time) * 1000
            metrics.total_time_ms += duration_ms
            metrics.average_time_ms = metrics.total_time_ms / metrics.total_calls
            metrics.min_time_ms = min(metrics.min_time_ms, duration_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, duration_ms)
            metrics.last_updated = datetime.now(timezone.utc)
            
        except Exception:
            # Still record the call for metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics.total_time_ms += duration_ms
            metrics.average_time_ms = metrics.total_time_ms / metrics.total_calls
            raise
        finally:
            metrics.concurrent_executions -= 1
            
    async def optimize_batch_processing(self, tasks: List[Tuple[Callable, tuple, dict]],
                                      cpu_bound: bool = False) -> List[Any]:
        """Optimize batch processing with adaptive concurrency."""
        # Get optimal concurrency
        optimal_concurrency = self.resource_manager.optimize_concurrency()
        
        # Update executor concurrency if needed
        if optimal_concurrency != self.executor.max_workers:
            self.executor = ConcurrentExecutor(max_workers=optimal_concurrency)
            
        return await self.executor.execute_batch(tasks, cpu_bound=cpu_bound)
        
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = self.cache.get_stats()
        resource_info = self.resource_manager.get_resource_recommendation()
        
        # Operation performance summary
        operation_summary = {}
        for name, metrics in self.metrics.items():
            operation_summary[name] = {
                "total_calls": metrics.total_calls,
                "average_time_ms": round(metrics.average_time_ms, 2),
                "min_time_ms": round(metrics.min_time_ms, 2),
                "max_time_ms": round(metrics.max_time_ms, 2),
                "concurrent_executions": metrics.concurrent_executions,
                "optimization_level": metrics.optimization_level,
                "last_updated": metrics.last_updated.isoformat()
            }
            
        return {
            "optimization_status": "enabled" if self._optimization_enabled else "disabled",
            "cache_performance": cache_stats,
            "resource_status": resource_info,
            "operation_metrics": operation_summary,
            "recommendations": self._generate_performance_recommendations(cache_stats, resource_info),
            "system_performance": {
                "optimal_concurrency": self.resource_manager._optimal_concurrency,
                "adjustment_factor": self.resource_manager._adjustment_factor,
                "executor_max_workers": self.executor.max_workers
            }
        }
        
    def _generate_performance_recommendations(self, cache_stats: Dict, resource_info: Dict) -> List[str]:
        """Generate actionable performance recommendations."""
        recommendations = []
        
        # Cache recommendations
        if cache_stats["hit_rate_percent"] < 60:
            recommendations.append("Cache hit rate is low. Consider increasing cache size or reviewing cache keys.")
            
        if cache_stats["memory_usage_mb"] > 90:
            recommendations.append("Memory cache is near capacity. Consider increasing memory limit.")
            
        # Resource recommendations  
        if "recommendations" in resource_info:
            recommendations.extend(resource_info["recommendations"])
            
        # Operation recommendations
        slow_operations = [
            name for name, metrics in self.metrics.items()
            if metrics.average_time_ms > 1000  # > 1 second
        ]
        
        if slow_operations:
            recommendations.append(
                f"Consider optimizing slow operations: {', '.join(slow_operations[:3])}"
            )
            
        high_concurrency_ops = [
            name for name, metrics in self.metrics.items()
            if metrics.concurrent_executions > 10
        ]
        
        if high_concurrency_ops:
            recommendations.append(
                "High concurrency detected. Consider implementing queue limits or throttling."
            )
            
        if not recommendations:
            recommendations.append("System performance is optimal. Continue monitoring.")
            
        return recommendations
        
    def clear_metrics(self) -> None:
        """Clear all performance metrics."""
        self.metrics.clear()
        self.cache.clear()
        self.logger.info("Performance metrics cleared")
        
    def shutdown(self) -> None:
        """Shutdown performance optimizer."""
        self.executor.shutdown()
        self.logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer