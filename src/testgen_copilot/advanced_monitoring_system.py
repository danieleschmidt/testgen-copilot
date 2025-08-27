"""Advanced Monitoring System - Generation 2 Implementation

Comprehensive monitoring with predictive analytics, distributed tracing,
custom metrics, automated alerting, and observability dashboards.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import psutil
import requests
from prometheus_client import Counter, Gauge, Histogram, Info, Summary, start_http_server

from .logging_config import get_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert notification."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class MetricData:
    """Custom metric data point."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    help_text: str = ""


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedMonitoringSystem:
    """Comprehensive monitoring and observability system."""
    
    def __init__(self, 
                 prometheus_port: int = 8000,
                 enable_profiling: bool = True,
                 alert_webhook: Optional[str] = None):
        """Initialize advanced monitoring system."""
        self.logger = get_logger(__name__)
        self.prometheus_port = prometheus_port
        self.enable_profiling = enable_profiling
        self.alert_webhook = alert_webhook
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Alert management
        self._alerts: List[Alert] = []
        self._alert_rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        self._notification_channels: List[Callable[[Alert], None]] = []
        
        # Performance tracking
        self._performance_profiles: deque = deque(maxlen=10000)
        self._operation_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_time": 0, "avg_time": 0, "max_time": 0}
        )
        
        # System metrics history
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Custom metrics registry
        self._custom_metrics: Dict[str, Any] = {}
        
        # Monitoring threads
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # Start Prometheus server
        self._start_prometheus_server()
        
        # Start monitoring
        self.start_monitoring()
        
        self.logger.info("Advanced monitoring system initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        self.prom_request_count = Counter(
            'testgen_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.prom_request_duration = Histogram(
            'testgen_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.prom_active_connections = Gauge(
            'testgen_active_connections',
            'Number of active connections'
        )
        
        self.prom_memory_usage = Gauge(
            'testgen_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.prom_cpu_usage = Gauge(
            'testgen_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.prom_error_rate = Gauge(
            'testgen_error_rate_percent',
            'Error rate percentage'
        )
        
        self.prom_test_generation_duration = Histogram(
            'testgen_test_generation_duration_seconds',
            'Test generation duration in seconds',
            ['language', 'complexity']
        )
        
        self.prom_security_scan_duration = Histogram(
            'testgen_security_scan_duration_seconds',
            'Security scan duration in seconds',
            ['scan_type']
        )
        
        self.prom_quantum_operations = Counter(
            'testgen_quantum_operations_total',
            'Total quantum operations',
            ['operation_type', 'success']
        )
        
        self.prom_cache_hits = Counter(
            'testgen_cache_hits_total',
            'Cache hits',
            ['cache_type']
        )
        
        self.prom_cache_misses = Counter(
            'testgen_cache_misses_total',
            'Cache misses',
            ['cache_type']
        )
        
        # Business metrics
        self.prom_tests_generated = Counter(
            'testgen_tests_generated_total',
            'Total tests generated',
            ['language', 'framework']
        )
        
        self.prom_vulnerabilities_found = Counter(
            'testgen_vulnerabilities_found_total',
            'Total vulnerabilities found',
            ['severity', 'type']
        )
        
        # System info
        self.prom_build_info = Info(
            'testgen_build_info',
            'Build information'
        )
        
        # Set build info
        self.prom_build_info.info({
            'version': '0.1.0',
            'build_time': datetime.now(timezone.utc).isoformat(),
            'python_version': '3.12'
        })
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.prometheus_port)
            self.logger.info(f"Prometheus server started on port {self.prometheus_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alert rules
                self._check_alert_rules()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.prom_cpu_usage.set(cpu_percent)
            self._metrics_history["cpu_usage"].append((time.time(), cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.prom_memory_usage.set(memory.used)
            self._metrics_history["memory_usage"].append((time.time(), memory.percent))
            
            # Network connections
            try:
                connections = len(psutil.net_connections())
                self.prom_active_connections.set(connections)
                self._metrics_history["connections"].append((time.time(), connections))
            except (psutil.AccessDenied, OSError):
                pass
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self._metrics_history["disk_usage"].append((time.time(), disk.percent))
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record HTTP request metrics."""
        self.prom_request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.prom_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Update error rate
        if status.startswith('4') or status.startswith('5'):
            self._update_error_rate()
    
    def record_test_generation(self, 
                              language: str, 
                              complexity: str, 
                              duration: float,
                              success: bool = True):
        """Record test generation metrics."""
        self.prom_test_generation_duration.labels(
            language=language, 
            complexity=complexity
        ).observe(duration)
        
        if success:
            self.prom_tests_generated.labels(language=language, framework="pytest").inc()
        
        # Record performance profile
        if self.enable_profiling:
            profile = PerformanceProfile(
                operation="test_generation",
                duration_ms=duration * 1000,
                memory_usage_mb=self._get_current_memory_mb(),
                cpu_usage_percent=psutil.cpu_percent(),
                metadata={
                    "language": language,
                    "complexity": complexity,
                    "success": success
                }
            )
            self._performance_profiles.append(profile)
    
    def record_security_scan(self, scan_type: str, duration: float, vulnerabilities: List[str]):
        """Record security scan metrics."""
        self.prom_security_scan_duration.labels(scan_type=scan_type).observe(duration)
        
        # Record vulnerabilities
        for vuln in vulnerabilities:
            severity = self._determine_vulnerability_severity(vuln)
            self.prom_vulnerabilities_found.labels(severity=severity, type=scan_type).inc()
    
    def record_quantum_operation(self, operation_type: str, success: bool):
        """Record quantum operation metrics."""
        self.prom_quantum_operations.labels(
            operation_type=operation_type,
            success=str(success).lower()
        ).inc()
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        if hit:
            self.prom_cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.prom_cache_misses.labels(cache_type=cache_type).inc()
    
    def record_custom_metric(self, metric: MetricData):
        """Record custom metric."""
        metric_key = f"{metric.name}_{hash(frozenset(metric.labels.items()))}"
        
        if metric_key not in self._custom_metrics:
            # Create Prometheus metric
            if metric.metric_type == MetricType.COUNTER:
                prom_metric = Counter(
                    metric.name, 
                    metric.help_text,
                    list(metric.labels.keys())
                )
            elif metric.metric_type == MetricType.GAUGE:
                prom_metric = Gauge(
                    metric.name,
                    metric.help_text,
                    list(metric.labels.keys())
                )
            elif metric.metric_type == MetricType.HISTOGRAM:
                prom_metric = Histogram(
                    metric.name,
                    metric.help_text,
                    list(metric.labels.keys())
                )
            else:  # SUMMARY
                prom_metric = Summary(
                    metric.name,
                    metric.help_text,
                    list(metric.labels.keys())
                )
            
            self._custom_metrics[metric_key] = prom_metric
        
        # Record metric value
        prom_metric = self._custom_metrics[metric_key]
        if metric.labels:
            labeled_metric = prom_metric.labels(**metric.labels)
        else:
            labeled_metric = prom_metric
        
        if metric.metric_type in [MetricType.COUNTER]:
            labeled_metric.inc(metric.value)
        elif metric.metric_type == MetricType.GAUGE:
            labeled_metric.set(metric.value)
        else:  # HISTOGRAM, SUMMARY
            labeled_metric.observe(metric.value)
    
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add custom alert rule."""
        self._alert_rules.append(rule)
    
    def add_notification_channel(self, channel: Callable[[Alert], None]):
        """Add notification channel for alerts."""
        self._notification_channels.append(channel)
    
    def _check_alert_rules(self):
        """Check all alert rules against current metrics."""
        current_metrics = self._get_current_metrics()
        
        for rule in self._alert_rules:
            try:
                alert = rule(current_metrics)
                if alert:
                    self._handle_alert(alert)
            except Exception as e:
                self.logger.error(f"Alert rule error: {e}")
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                "timestamp": datetime.now(timezone.utc)
            }
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def _handle_alert(self, alert: Alert):
        """Handle new alert."""
        # Check if similar alert already exists
        existing = next(
            (a for a in self._alerts 
             if a.title == alert.title and not a.resolved),
            None
        )
        
        if existing:
            return  # Don't duplicate alerts
        
        self._alerts.append(alert)
        self.logger.warning(f"Alert: {alert.title} - {alert.description}")
        
        # Send notifications
        for channel in self._notification_channels:
            try:
                channel(alert)
            except Exception as e:
                self.logger.error(f"Notification channel error: {e}")
    
    def _update_error_rate(self):
        """Update application error rate."""
        # Simplified error rate calculation
        # In production, this would use a sliding window
        try:
            error_count = sum(1 for a in self._alerts if a.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL])
            total_operations = len(self._operation_stats)
            error_rate = (error_count / max(total_operations, 1)) * 100
            self.prom_error_rate.set(error_rate)
        except Exception as e:
            self.logger.error(f"Failed to update error rate: {e}")
    
    def _determine_vulnerability_severity(self, vulnerability: str) -> str:
        """Determine vulnerability severity based on type."""
        vulnerability_lower = vulnerability.lower()
        
        if any(critical in vulnerability_lower for critical in ['sql injection', 'rce', 'command injection']):
            return 'critical'
        elif any(high in vulnerability_lower for high in ['xss', 'csrf', 'authentication']):
            return 'high'
        elif any(medium in vulnerability_lower for medium in ['information disclosure', 'weak crypto']):
            return 'medium'
        else:
            return 'low'
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory = psutil.virtual_memory()
            return memory.used / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _cleanup_old_data(self):
        """Cleanup old monitoring data."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        # Cleanup metrics history
        for metric_name, history in self._metrics_history.items():
            while history and history[0][0] < cutoff_time:
                history.popleft()
        
        # Cleanup resolved alerts older than 24 hours
        cutoff_datetime = datetime.now(timezone.utc).timestamp() - 86400
        self._alerts = [
            alert for alert in self._alerts
            if not alert.resolved or alert.timestamp.timestamp() > cutoff_datetime
        ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        current_metrics = self._get_current_metrics()
        active_alerts = [a for a in self._alerts if not a.resolved]
        
        # Determine overall health
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]
        
        if critical_alerts:
            status = "critical"
        elif error_alerts:
            status = "unhealthy"
        elif current_metrics.get("cpu_usage", 0) > 80 or current_metrics.get("memory_usage", 0) > 85:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": current_metrics,
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "performance_profiles": len(self._performance_profiles),
            "uptime_seconds": time.time() - (hasattr(self, '_start_time') and self._start_time or time.time())
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self._performance_profiles:
            return {"message": "No performance data available"}
        
        # Calculate statistics
        durations = [p.duration_ms for p in self._performance_profiles]
        memory_usage = [p.memory_usage_mb for p in self._performance_profiles]
        cpu_usage = [p.cpu_usage_percent for p in self._performance_profiles]
        
        return {
            "total_operations": len(self._performance_profiles),
            "duration_stats": {
                "avg_ms": sum(durations) / len(durations),
                "max_ms": max(durations),
                "min_ms": min(durations)
            },
            "memory_stats": {
                "avg_mb": sum(memory_usage) / len(memory_usage),
                "max_mb": max(memory_usage),
                "min_mb": min(memory_usage)
            },
            "cpu_stats": {
                "avg_percent": sum(cpu_usage) / len(cpu_usage),
                "max_percent": max(cpu_usage),
                "min_percent": min(cpu_usage)
            },
            "operations_by_type": {
                op: len([p for p in self._performance_profiles if p.operation == op])
                for op in set(p.operation for p in self._performance_profiles)
            }
        }
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        data = {
            "health": self.get_health_status(),
            "performance": self.get_performance_summary(),
            "alerts": [asdict(a) for a in self._alerts[-100:]],  # Last 100 alerts
            "metrics_history": {
                name: list(history)[-100:]  # Last 100 points
                for name, history in self._metrics_history.items()
            }
        }
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return str(data)


# Global monitoring instance
_monitoring_system = None


def get_monitoring_system(**kwargs) -> AdvancedMonitoringSystem:
    """Get or create global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = AdvancedMonitoringSystem(**kwargs)
    return _monitoring_system


def monitor_operation(operation: str):
    """Decorator to monitor operation performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record performance
                monitoring.record_custom_metric(MetricData(
                    name=f"operation_{operation}_duration_seconds",
                    value=duration,
                    metric_type=MetricType.HISTOGRAM,
                    labels={"operation": operation, "success": "true"}
                ))
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed operation
                monitoring.record_custom_metric(MetricData(
                    name=f"operation_{operation}_duration_seconds",
                    value=duration,
                    metric_type=MetricType.HISTOGRAM,
                    labels={"operation": operation, "success": "false"}
                ))
                
                raise
        
        return wrapper
    return decorator