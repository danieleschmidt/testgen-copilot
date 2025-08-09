"""Advanced performance monitoring and optimization system."""

import time
import psutil
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import json
from pathlib import Path

from .logging_config import get_generator_logger

logger = get_generator_logger()


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    operation_name: str
    duration: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Real-time performance monitoring and analysis."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics = deque(maxlen=max_history)
        self.operation_stats = defaultdict(list)
        
        # System monitoring
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
            
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started performance monitoring")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        logger.info("Stopped performance monitoring")
        
    def record_operation(self,
                        operation_name: str,
                        duration: float,
                        success: bool = True,
                        metadata: Optional[Dict[str, Any]] = None):
        """Record performance metrics for an operation.
        
        Args:
            operation_name: Name of the operation
            duration: Operation duration in seconds
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        try:
            # Get current system metrics
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            metric = PerformanceMetric(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                operation_name=operation_name,
                duration=duration,
                success=success,
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            self.operation_stats[operation_name].append(metric)
            
            # Keep operation stats within reasonable limits
            if len(self.operation_stats[operation_name]) > 100:
                self.operation_stats[operation_name] = self.operation_stats[operation_name][-100:]
                
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dict with operation statistics
        """
        metrics = self.operation_stats[operation_name]
        if not metrics:
            return {}
            
        durations = [m.duration for m in metrics]
        success_count = sum(1 for m in metrics if m.success)
        
        return {
            'operation_name': operation_name,
            'total_calls': len(metrics),
            'success_count': success_count,
            'failure_count': len(metrics) - success_count,
            'success_rate': success_count / len(metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations),
            'avg_cpu': sum(m.cpu_percent for m in metrics) / len(metrics),
            'avg_memory_mb': sum(m.memory_mb for m in metrics) / len(metrics),
        }
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system performance statistics.
        
        Returns:
            Dict with system stats
        """
        try:
            # System-wide stats
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Process-specific stats
            process_cpu = self.process.cpu_percent()
            process_memory = self.process.memory_info()
            
            return {
                'timestamp': time.time(),
                'system': {
                    'cpu_count': cpu_count,
                    'cpu_percent': cpu_percent,
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_percent': memory.percent,
                },
                'process': {
                    'cpu_percent': process_cpu,
                    'memory_mb': process_memory.rss / (1024 * 1024),
                    'memory_percent': (process_memory.rss / memory.total) * 100,
                    'threads': self.process.num_threads(),
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
            
    def export_metrics(self, filepath: Path):
        """Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
        """
        try:
            export_data = {
                'timestamp': time.time(),
                'metrics': [
                    {
                        'timestamp': m.timestamp,
                        'cpu_percent': m.cpu_percent,
                        'memory_mb': m.memory_mb,
                        'operation_name': m.operation_name,
                        'duration': m.duration,
                        'success': m.success,
                        'metadata': m.metadata,
                    }
                    for m in self.metrics
                ],
                'operation_stats': {
                    name: self.get_operation_stats(name)
                    for name in self.operation_stats.keys()
                },
                'system_stats': self.get_system_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Record system snapshot
                self.record_operation(
                    operation_name="system_monitor",
                    duration=0.0,
                    success=True,
                    metadata=self.get_system_stats()
                )
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)


def performance_timer(operation_name: str, 
                     monitor: Optional[PerformanceMonitor] = None):
    """Decorator for timing function execution.
    
    Args:
        operation_name: Name of the operation
        monitor: Performance monitor instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                if monitor:
                    monitor.record_operation(
                        operation_name=operation_name,
                        duration=duration,
                        success=success
                    )
                else:
                    logger.debug(f"{operation_name} completed in {duration:.3f}s (success: {success})")
                    
        return wrapper
    return decorator


class PerformanceOptimizer:
    """Automated performance optimization system."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize performance optimizer.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.optimization_rules = []
        self.applied_optimizations = set()
        
    def add_optimization_rule(self, 
                             condition: Callable[[Dict[str, Any]], bool],
                             action: Callable[[], None],
                             rule_id: str):
        """Add an optimization rule.
        
        Args:
            condition: Function that returns True if optimization should be applied
            action: Function to execute the optimization
            rule_id: Unique identifier for this rule
        """
        self.optimization_rules.append({
            'condition': condition,
            'action': action,
            'rule_id': rule_id
        })
        
    def check_optimizations(self):
        """Check and apply optimization rules."""
        system_stats = self.monitor.get_system_stats()
        
        for rule in self.optimization_rules:
            rule_id = rule['rule_id']
            
            # Skip if already applied
            if rule_id in self.applied_optimizations:
                continue
                
            # Check condition
            if rule['condition'](system_stats):
                try:
                    rule['action']()
                    self.applied_optimizations.add(rule_id)
                    logger.info(f"Applied optimization rule: {rule_id}")
                except Exception as e:
                    logger.error(f"Error applying optimization {rule_id}: {e}")


# Global performance monitor instance
global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return global_monitor


class PerformanceContext:
    """Context manager for performance monitoring."""
    
    def __init__(self, operation_name: str, monitor: Optional[PerformanceMonitor] = None):
        """Initialize performance context.
        
        Args:
            operation_name: Name of the operation
            monitor: Performance monitor instance
        """
        self.operation_name = operation_name
        self.monitor = monitor or global_monitor
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            self.monitor.record_operation(
                operation_name=self.operation_name,
                duration=duration,
                success=success
            )