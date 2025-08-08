"""Enhanced monitoring and observability for TestGen Copilot."""

from __future__ import annotations

import asyncio
import psutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import threading
from collections import defaultdict

from .logging_config import get_core_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"  
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(Enum):
    """Types of metrics we collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_used_gb: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    tests_generated: int = 0
    security_scans_completed: int = 0
    coverage_analyses: int = 0
    errors_encountered: int = 0
    average_processing_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    active_operations: int = 0
    total_files_processed: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass  
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    metric_value: Union[float, int, str]
    threshold: Union[float, int, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, 
                 check_interval_seconds: float = 30.0,
                 alert_cooldown_seconds: float = 300.0):
        self.check_interval = check_interval_seconds
        self.alert_cooldown = alert_cooldown_seconds
        self.logger = get_core_logger()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.metrics_history: List[SystemMetrics] = []
        self.app_metrics: ApplicationMetrics = ApplicationMetrics()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Configurable thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_high": 75.0,
            "memory_critical": 90.0,
            "memory_high": 80.0,
            "disk_critical": 95.0,
            "disk_high": 85.0,
            "error_rate_critical": 50.0,  # errors per minute
            "error_rate_high": 10.0,
            "response_time_critical": 30000.0,  # ms
            "response_time_high": 10000.0,
        }
        
        self._operation_timings: Dict[str, List[float]] = defaultdict(list)
        self._error_counts = defaultdict(int)
        
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Health monitoring started", {
            "check_interval_seconds": self.check_interval,
            "alert_cooldown_seconds": self.alert_cooldown
        })
        
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        self.logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.metrics_history.append(system_metrics)
                
                # Keep only last hour of metrics
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                
                # Check for alerts
                self._check_system_alerts(system_metrics)
                self._check_application_alerts()
                
                # Log periodic health summary
                self._log_health_summary(system_metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                time.sleep(self.check_interval)
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage for current working directory
            disk = psutil.disk_usage(Path.cwd())
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            
            # Process info
            process = psutil.Process()
            active_threads = process.num_threads()
            open_files = len(process.open_files())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024**2),
                disk_usage_percent=disk_usage_percent,
                disk_used_gb=disk_used_gb,
                active_threads=active_threads,
                open_files=open_files
            )
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return SystemMetrics()
            
    def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check system metrics against thresholds."""
        checks = [
            ("cpu_usage", metrics.cpu_percent, self.thresholds["cpu_critical"], 
             self.thresholds["cpu_high"], "CPU usage"),
            ("memory_usage", metrics.memory_percent, self.thresholds["memory_critical"],
             self.thresholds["memory_high"], "Memory usage"),
            ("disk_usage", metrics.disk_usage_percent, self.thresholds["disk_critical"],
             self.thresholds["disk_high"], "Disk usage"),
        ]
        
        for metric_name, value, critical_threshold, high_threshold, description in checks:
            if value >= critical_threshold:
                self._create_alert(
                    metric_name,
                    AlertSeverity.CRITICAL,
                    f"Critical {description}",
                    f"{description} is critically high at {value:.1f}%",
                    value,
                    critical_threshold
                )
            elif value >= high_threshold:
                self._create_alert(
                    metric_name,
                    AlertSeverity.HIGH,
                    f"High {description}",
                    f"{description} is high at {value:.1f}%",
                    value,
                    high_threshold
                )
            else:
                # Resolve alert if it exists
                self._resolve_alert(metric_name)
                
    def _check_application_alerts(self) -> None:
        """Check application metrics for issues."""
        # Error rate check (errors per minute)
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        
        recent_errors = sum(
            count for timestamp, count in 
            [(t, c) for t, c in self._error_counts.items() if t > minute_ago]
        )
        
        if recent_errors >= self.thresholds["error_rate_critical"]:
            self._create_alert(
                "error_rate",
                AlertSeverity.CRITICAL,
                "Critical Error Rate",
                f"Error rate is critically high: {recent_errors} errors/minute",
                recent_errors,
                self.thresholds["error_rate_critical"]
            )
        elif recent_errors >= self.thresholds["error_rate_high"]:
            self._create_alert(
                "error_rate",
                AlertSeverity.HIGH,
                "High Error Rate",
                f"Error rate is high: {recent_errors} errors/minute",
                recent_errors,
                self.thresholds["error_rate_high"]
            )
        else:
            self._resolve_alert("error_rate")
            
    def _create_alert(self, metric_name: str, severity: AlertSeverity, 
                     title: str, message: str, value: Union[float, int, str],
                     threshold: Union[float, int, str]) -> None:
        """Create or update an alert."""
        # Check alert cooldown
        last_alert_time = self.last_alert_times.get(metric_name)
        if last_alert_time:
            time_since_last = datetime.now(timezone.utc) - last_alert_time
            if time_since_last.total_seconds() < self.alert_cooldown:
                return
                
        alert = Alert(
            id=f"{metric_name}_{int(time.time())}",
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold
        )
        
        self.active_alerts[metric_name] = alert
        self.alert_history.append(alert)
        self.last_alert_times[metric_name] = alert.timestamp
        
        self.logger.error("Alert triggered", {
            "alert_id": alert.id,
            "severity": severity.value,
            "title": title,
            "message": message,
            "metric_name": metric_name,
            "metric_value": value,
            "threshold": threshold,
            "alert_type": "system_health"
        })
        
    def _resolve_alert(self, metric_name: str) -> None:
        """Resolve an active alert."""
        if metric_name in self.active_alerts:
            alert = self.active_alerts.pop(metric_name)
            alert.resolved = True
            
            self.logger.info("Alert resolved", {
                "alert_id": alert.id,
                "metric_name": metric_name,
                "duration_seconds": (datetime.now(timezone.utc) - alert.timestamp).total_seconds(),
                "alert_type": "system_health"
            })
            
    def _log_health_summary(self, metrics: SystemMetrics) -> None:
        """Log periodic health summary."""
        self.logger.info("Health check summary", {
            "system_metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_mb": metrics.memory_used_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "active_threads": metrics.active_threads,
                "open_files": metrics.open_files
            },
            "application_metrics": {
                "tests_generated": self.app_metrics.tests_generated,
                "security_scans": self.app_metrics.security_scans_completed,
                "coverage_analyses": self.app_metrics.coverage_analyses,
                "errors_encountered": self.app_metrics.errors_encountered,
                "active_operations": self.app_metrics.active_operations,
                "cache_hit_rate": self.app_metrics.cache_hit_rate
            },
            "active_alerts": len(self.active_alerts),
            "monitoring_health": "healthy" if len(self.active_alerts) == 0 else "degraded"
        })
        
    @contextmanager
    def operation_timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.app_metrics.active_operations += 1
        
        try:
            yield
        except Exception as e:
            self.record_error(operation_name, str(e))
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._operation_timings[operation_name].append(duration_ms)
            self.app_metrics.active_operations -= 1
            
            # Keep only recent timings (last 100)
            if len(self._operation_timings[operation_name]) > 100:
                self._operation_timings[operation_name] = \
                    self._operation_timings[operation_name][-100:]
                    
            # Update average processing time
            all_timings = []
            for timings in self._operation_timings.values():
                all_timings.extend(timings[-10:])  # Last 10 operations per type
                
            if all_timings:
                self.app_metrics.average_processing_time_ms = sum(all_timings) / len(all_timings)
                
    def record_test_generation(self) -> None:
        """Record a successful test generation."""
        self.app_metrics.tests_generated += 1
        
    def record_security_scan(self) -> None:
        """Record a completed security scan."""
        self.app_metrics.security_scans_completed += 1
        
    def record_coverage_analysis(self) -> None:
        """Record a completed coverage analysis."""
        self.app_metrics.coverage_analyses += 1
        
    def record_error(self, operation: str, error_message: str) -> None:
        """Record an error occurrence."""
        self.app_metrics.errors_encountered += 1
        now = datetime.now(timezone.utc)
        self._error_counts[now] += 1
        
        self.logger.error("Operation error recorded", {
            "operation": operation,
            "error_message": error_message,
            "total_errors": self.app_metrics.errors_encountered,
            "monitoring_event": "error_tracked"
        })
        
    def update_cache_stats(self, hit_rate: float) -> None:
        """Update cache hit rate statistics."""
        self.app_metrics.cache_hit_rate = hit_rate
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else SystemMetrics()
        
        return {
            "status": "healthy" if len(self.active_alerts) == 0 else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_used_mb": latest_metrics.memory_used_mb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "active_threads": latest_metrics.active_threads,
                "open_files": latest_metrics.open_files
            },
            "application_metrics": {
                "tests_generated": self.app_metrics.tests_generated,
                "security_scans_completed": self.app_metrics.security_scans_completed,
                "coverage_analyses": self.app_metrics.coverage_analyses,
                "errors_encountered": self.app_metrics.errors_encountered,
                "average_processing_time_ms": self.app_metrics.average_processing_time_ms,
                "cache_hit_rate": self.app_metrics.cache_hit_rate,
                "active_operations": self.app_metrics.active_operations,
                "total_files_processed": self.app_metrics.total_files_processed
            },
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts.values()
            ],
            "monitoring": {
                "is_active": self.is_monitoring,
                "check_interval_seconds": self.check_interval,
                "metrics_history_size": len(self.metrics_history),
                "alert_history_size": len(self.alert_history)
            }
        }
        
    def get_metrics_export(self) -> Dict[str, Any]:
        """Export metrics in Prometheus-compatible format."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else SystemMetrics()
        
        return {
            "testgen_cpu_usage_percent": latest_metrics.cpu_percent,
            "testgen_memory_usage_percent": latest_metrics.memory_percent,
            "testgen_memory_used_bytes": latest_metrics.memory_used_mb * 1024 * 1024,
            "testgen_disk_usage_percent": latest_metrics.disk_usage_percent,
            "testgen_active_threads": latest_metrics.active_threads,
            "testgen_open_files": latest_metrics.open_files,
            "testgen_tests_generated_total": self.app_metrics.tests_generated,
            "testgen_security_scans_total": self.app_metrics.security_scans_completed,
            "testgen_coverage_analyses_total": self.app_metrics.coverage_analyses,
            "testgen_errors_total": self.app_metrics.errors_encountered,
            "testgen_average_processing_time_ms": self.app_metrics.average_processing_time_ms,
            "testgen_cache_hit_rate": self.app_metrics.cache_hit_rate,
            "testgen_active_operations": self.app_metrics.active_operations,
            "testgen_files_processed_total": self.app_metrics.total_files_processed,
            "testgen_active_alerts": len(self.active_alerts),
            "testgen_monitoring_healthy": 1 if len(self.active_alerts) == 0 else 0
        }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
        _health_monitor.start_monitoring()
    return _health_monitor


def shutdown_monitoring():
    """Shutdown the global health monitor."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()
        _health_monitor = None