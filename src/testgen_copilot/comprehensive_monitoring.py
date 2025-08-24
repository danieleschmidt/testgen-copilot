"""
Comprehensive Monitoring System for TestGen Copilot
Implements advanced observability, metrics collection, and alerting
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import queue

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Represents a metric with metadata."""
    name: str
    metric_type: MetricType
    value: Union[float, int]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    labels: Dict[str, str] = field(default_factory=dict)
    triggered_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals"
    threshold: float
    severity: AlertSeverity
    duration: int = 0  # seconds the condition must be true
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    function_name: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, max_history_size: int = 10000):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.max_history_size = max_history_size
        self.lock = threading.RLock()
        
        # Pre-aggregated metrics for efficiency
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        labels = labels or {}
        
        with self.lock:
            metric_key = self._get_metric_key(name, labels)
            self.counters[metric_key] += value
            
            metric = Metric(
                name=name,
                metric_type=MetricType.COUNTER,
                value=self.counters[metric_key],
                labels=labels
            )
            
            self._store_metric(name, metric)
            
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        labels = labels or {}
        
        with self.lock:
            metric_key = self._get_metric_key(name, labels)
            self.gauges[metric_key] = value
            
            metric = Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                labels=labels
            )
            
            self._store_metric(name, metric)
            
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a histogram."""
        labels = labels or {}
        
        with self.lock:
            metric_key = self._get_metric_key(name, labels)
            self.histograms[metric_key].append(value)
            
            # Keep only recent values
            if len(self.histograms[metric_key]) > 1000:
                self.histograms[metric_key] = self.histograms[metric_key][-500:]
                
            metric = Metric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                value=value,
                labels=labels
            )
            
            self._store_metric(name, metric)
            
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timer metric."""
        labels = labels or {}
        
        with self.lock:
            metric_key = self._get_metric_key(name, labels)
            self.timers[metric_key].append(duration)
            
            metric = Metric(
                name=name,
                metric_type=MetricType.TIMER,
                value=duration,
                labels=labels
            )
            
            self._store_metric(name, metric)
            
    def get_metric_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current value of a metric."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)
        
        with self.lock:
            # Try different metric types
            if metric_key in self.counters:
                return self.counters[metric_key]
            elif metric_key in self.gauges:
                return self.gauges[metric_key]
            elif metric_key in self.histograms and self.histograms[metric_key]:
                return self.histograms[metric_key][-1]
            elif metric_key in self.timers and self.timers[metric_key]:
                return self.timers[metric_key][-1]
                
        return None
        
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)
        
        with self.lock:
            values = self.histograms.get(metric_key, [])
            
        if not values:
            return {}
            
        values = np.array(values)
        return {
            "count": len(values),
            "sum": float(np.sum(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "stddev": float(np.std(values))
        }
        
    def get_timer_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)
        
        with self.lock:
            values = list(self.timers.get(metric_key, []))
            
        if not values:
            return {}
            
        values = np.array(values)
        return {
            "count": len(values),
            "sum": float(np.sum(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in Prometheus format."""
        with self.lock:
            result = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: self.get_histogram_stats(name.split("|")[0], self._parse_labels(name))
                    for name in self.histograms.keys()
                },
                "timers": {
                    name: self.get_timer_stats(name.split("|")[0], self._parse_labels(name))
                    for name in self.timers.keys()
                }
            }
            
        return result
        
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name
            
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
        
    def _parse_labels(self, metric_key: str) -> Dict[str, str]:
        """Parse labels from metric key."""
        if "|" not in metric_key:
            return {}
            
        label_str = metric_key.split("|", 1)[1]
        labels = {}
        
        for pair in label_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
                
        return labels
        
    def _store_metric(self, name: str, metric: Metric) -> None:
        """Store metric in history."""
        self.metrics[name].append(metric)
        
        # Trim history if too large
        if len(self.metrics[name]) > self.max_history_size:
            self.metrics[name] = self.metrics[name][-self.max_history_size//2:]


class AlertManager:
    """Alert management system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        self.evaluation_interval = 15  # seconds
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)
        
    def start_monitoring(self) -> None:
        """Start alert monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Alert monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
        
    def _monitoring_loop(self) -> None:
        """Main alert monitoring loop."""
        while self.monitoring_active:
            try:
                self._evaluate_alert_rules()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                time.sleep(5)
                
    def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules."""
        for rule_name, rule in self.alert_rules.items():
            try:
                self._evaluate_single_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
                
    def _evaluate_single_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        current_value = self.metrics_collector.get_metric_value(rule.metric_name, rule.labels)
        
        if current_value is None:
            return
            
        # Check condition
        condition_met = False
        
        if rule.condition == "greater_than":
            condition_met = current_value > rule.threshold
        elif rule.condition == "less_than":
            condition_met = current_value < rule.threshold
        elif rule.condition == "equals":
            condition_met = abs(current_value - rule.threshold) < 0.001
            
        alert_id = f"{rule.name}_{rule.metric_name}"
        
        if condition_met:
            # Alert should be active
            if alert_id not in self.active_alerts:
                # New alert
                alert = Alert(
                    id=alert_id,
                    name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.description} (current: {current_value}, threshold: {rule.threshold})",
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold_value=rule.threshold,
                    labels=rule.labels
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Notify handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")
                        
                logger.warning(f"Alert triggered: {rule.name}")
                
        else:
            # Alert should be resolved
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {rule.name}")
                
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = list(self.active_alerts.values())
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
            
        return {
            "total_active": len(active_alerts),
            "by_severity": dict(severity_counts),
            "total_rules": len(self.alert_rules),
            "alert_history_size": len(self.alert_history)
        }


class PerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.RLock()
        
    def start_timer(self, function_name: str) -> None:
        """Start timing a function."""
        with self.lock:
            self.active_timers[function_name] = time.time()
            
    def end_timer(self, function_name: str) -> float:
        """End timing a function and record the duration."""
        with self.lock:
            if function_name not in self.active_timers:
                return 0.0
                
            duration = time.time() - self.active_timers[function_name]
            del self.active_timers[function_name]
            
            self._record_execution(function_name, duration)
            return duration
            
    def _record_execution(self, function_name: str, duration: float) -> None:
        """Record function execution metrics."""
        if function_name not in self.profiles:
            self.profiles[function_name] = PerformanceProfile(function_name=function_name)
            
        profile = self.profiles[function_name]
        
        profile.execution_count += 1
        profile.total_time += duration
        profile.min_time = min(profile.min_time, duration)
        profile.max_time = max(profile.max_time, duration)
        profile.avg_time = profile.total_time / profile.execution_count
        
        # Store duration for percentile calculations
        if len(profile.memory_usage) > 1000:  # Reuse memory_usage list for durations
            profile.memory_usage = profile.memory_usage[-500:]
            
        profile.memory_usage.append(duration)  # Temporary storage for durations
        
        # Calculate percentiles
        if len(profile.memory_usage) >= 10:
            durations = np.array(profile.memory_usage)
            profile.percentile_95 = float(np.percentile(durations, 95))
            profile.percentile_99 = float(np.percentile(durations, 99))
            
        profile.last_updated = datetime.now()
        
    def get_profile(self, function_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for a function."""
        return self.profiles.get(function_name)
        
    def get_all_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get all performance profiles."""
        return self.profiles.copy()
        
    def get_slowest_functions(self, limit: int = 10) -> List[PerformanceProfile]:
        """Get the slowest functions by average execution time."""
        profiles = list(self.profiles.values())
        profiles.sort(key=lambda p: p.avg_time, reverse=True)
        return profiles[:limit]
        
    def get_most_called_functions(self, limit: int = 10) -> List[PerformanceProfile]:
        """Get the most frequently called functions."""
        profiles = list(self.profiles.values())
        profiles.sort(key=lambda p: p.execution_count, reverse=True)
        return profiles[:limit]


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.collection_interval = 10  # seconds
        
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main system monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(5)
                
    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.metrics_collector.set_gauge("system_cpu_count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_total", memory.total)
            self.metrics_collector.set_gauge("system_memory_available", memory.available)
            self.metrics_collector.set_gauge("system_memory_used", memory.used)
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_total", disk.total)
            self.metrics_collector.set_gauge("system_disk_used", disk.used)
            self.metrics_collector.set_gauge("system_disk_free", disk.free)
            self.metrics_collector.set_gauge("system_disk_percent", (disk.used / disk.total) * 100)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.metrics_collector.set_gauge("system_network_bytes_sent", net_io.bytes_sent)
            self.metrics_collector.set_gauge("system_network_bytes_recv", net_io.bytes_recv)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.metrics_collector.set_gauge("process_memory_rss", process_memory.rss)
            self.metrics_collector.set_gauge("process_memory_vms", process_memory.vms)
            self.metrics_collector.set_gauge("process_cpu_percent", process.cpu_percent())
            
            # File descriptor count
            try:
                fd_count = process.num_fds()
                self.metrics_collector.set_gauge("process_file_descriptors", fd_count)
            except (AttributeError, OSError):
                pass  # Not available on all platforms
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")


class ComprehensiveMonitoring:
    """
    Comprehensive monitoring system combining metrics, alerts, profiling, and system monitoring.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_profiler = PerformanceProfiler()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup default alert handlers
        self._setup_default_alert_handlers()
        
        # Start monitoring
        self.start_monitoring()
        
    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        self.alert_manager.start_monitoring()
        self.system_monitor.start_monitoring()
        logger.info("Comprehensive monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.alert_manager.stop_monitoring()
        self.system_monitor.stop_monitoring()
        logger.info("Comprehensive monitoring stopped")
        
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "metrics": self.metrics_collector.get_all_metrics(),
            "alerts": {
                "active": [
                    {
                        "id": alert.id,
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "triggered_at": alert.triggered_at.isoformat()
                    }
                    for alert in self.alert_manager.get_active_alerts()
                ],
                "summary": self.alert_manager.get_alert_summary()
            },
            "performance": {
                "slowest_functions": [
                    {
                        "name": profile.function_name,
                        "avg_time": profile.avg_time,
                        "execution_count": profile.execution_count,
                        "p95": profile.percentile_95,
                        "p99": profile.percentile_99
                    }
                    for profile in self.performance_profiler.get_slowest_functions(5)
                ],
                "most_called_functions": [
                    {
                        "name": profile.function_name,
                        "execution_count": profile.execution_count,
                        "avg_time": profile.avg_time
                    }
                    for profile in self.performance_profiler.get_most_called_functions(5)
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_percent",
                condition="greater_than",
                threshold=90,
                severity=AlertSeverity.WARNING,
                description="High CPU usage detected"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_percent",
                condition="greater_than",
                threshold=85,
                severity=AlertSeverity.WARNING,
                description="High memory usage detected"
            ),
            AlertRule(
                name="low_disk_space",
                metric_name="system_disk_percent",
                condition="greater_than",
                threshold=90,
                severity=AlertSeverity.ERROR,
                description="Low disk space detected"
            ),
            AlertRule(
                name="critical_memory_usage",
                metric_name="system_memory_percent",
                condition="greater_than",
                threshold=95,
                severity=AlertSeverity.CRITICAL,
                description="Critical memory usage detected"
            )
        ]
        
        for rule in rules:
            self.alert_manager.add_alert_rule(rule)
            
    def _setup_default_alert_handlers(self) -> None:
        """Setup default alert handlers."""
        
        def log_alert_handler(alert: Alert) -> None:
            """Log alert to logger."""
            level = logging.WARNING
            if alert.severity == AlertSeverity.ERROR:
                level = logging.ERROR
            elif alert.severity == AlertSeverity.CRITICAL:
                level = logging.CRITICAL
                
            logger.log(level, f"ALERT: {alert.name} - {alert.message}")
            
        def file_alert_handler(alert: Alert) -> None:
            """Write alert to file."""
            try:
                alert_file = Path("alerts.json")
                
                alert_data = {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "triggered_at": alert.triggered_at.isoformat()
                }
                
                # Append to file
                alerts = []
                if alert_file.exists():
                    with open(alert_file, 'r') as f:
                        alerts = json.load(f)
                        
                alerts.append(alert_data)
                
                # Keep only recent alerts
                if len(alerts) > 1000:
                    alerts = alerts[-500:]
                    
                with open(alert_file, 'w') as f:
                    json.dump(alerts, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed to write alert to file: {e}")
                
        self.alert_manager.add_alert_handler(log_alert_handler)
        self.alert_manager.add_alert_handler(file_alert_handler)


# Decorators for easy monitoring

def monitor_performance(profiler: Optional[PerformanceProfiler] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        nonlocal profiler
        if profiler is None:
            profiler = PerformanceProfiler()
            
        def wrapper(*args, **kwargs):
            function_name = f"{func.__module__}.{func.__qualname__}"
            profiler.start_timer(function_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_timer(function_name)
        return wrapper
    return decorator


def track_metrics(metrics_collector: Optional[MetricsCollector] = None):
    """Decorator to track function call metrics."""
    def decorator(func):
        nonlocal metrics_collector
        if metrics_collector is None:
            metrics_collector = MetricsCollector()
            
        def wrapper(*args, **kwargs):
            function_name = f"{func.__module__}.{func.__qualname__}"
            
            # Increment call counter
            metrics_collector.increment_counter(f"{function_name}_calls")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record success
                metrics_collector.increment_counter(f"{function_name}_success")
                
                return result
                
            except Exception as e:
                # Record error
                metrics_collector.increment_counter(f"{function_name}_errors", labels={"error_type": type(e).__name__})
                raise
                
            finally:
                # Record timing
                duration = time.time() - start_time
                metrics_collector.record_timer(f"{function_name}_duration", duration)
                
        return wrapper
    return decorator


async def main():
    """Example usage of comprehensive monitoring."""
    
    # Create monitoring system
    monitoring = ComprehensiveMonitoring()
    
    # Example function to monitor
    @monitor_performance(monitoring.performance_profiler)
    @track_metrics(monitoring.metrics_collector)
    def example_function(duration: float = 0.1):
        time.sleep(duration)
        if np.random.random() < 0.1:  # 10% failure rate
            raise Exception("Random error")
        return "Success"
    
    # Run some test calls
    for i in range(50):
        try:
            result = example_function(np.random.uniform(0.05, 0.3))
        except Exception as e:
            pass
            
        if i % 10 == 0:
            # Get status
            status = monitoring.get_comprehensive_status()
            print(f"Iteration {i}: Active alerts: {len(status['alerts']['active'])}")
            
        time.sleep(0.1)
    
    # Final status
    final_status = monitoring.get_comprehensive_status()
    print("\nFinal Monitoring Status:")
    print(f"Total metrics: {len(final_status['metrics']['counters'])}")
    print(f"Active alerts: {len(final_status['alerts']['active'])}")
    print(f"Performance profiles: {len(final_status['performance']['slowest_functions'])}")
    
    # Stop monitoring
    monitoring.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())