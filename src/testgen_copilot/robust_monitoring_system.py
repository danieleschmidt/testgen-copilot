"""
📊 Robust Monitoring System v2.0
================================

Comprehensive real-time monitoring, alerting, and observability platform.
Implements proactive health monitoring, performance tracking, and intelligent alerting.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union
from collections import defaultdict, deque
import threading
import psutil
import gc

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

from .logging_config import get_core_logger
from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor

logger = get_core_logger()
console = Console()


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    DOWN = "down"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class HealthCheck:
    """Represents a health check configuration"""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents an alert"""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    source_component: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    resolution_notes: str = ""


@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    thread_count: int = 0
    load_average: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class ApplicationMetrics:
    """Application-level metrics"""
    requests_total: int = 0
    requests_per_second: float = 0.0
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    concurrent_operations: int = 0


class RobustMonitoringSystem:
    """
    📊 Comprehensive monitoring and alerting system
    
    Features:
    - Real-time health monitoring
    - Proactive alerting with smart thresholds
    - Performance metrics collection
    - Resource utilization tracking
    - Custom health checks
    - Alert correlation and deduplication
    - Automated recovery triggers
    - Historical data retention
    - Dashboard and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_running = False
        self.start_time = time.time()
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        
        # Monitoring state
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.health_failures: Dict[str, int] = defaultdict(int)
        self.health_successes: Dict[str, int] = defaultdict(int)
        
        # Metrics storage
        self.system_metrics = SystemMetrics()
        self.application_metrics = ApplicationMetrics()
        self.custom_metrics: Dict[str, Any] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Alerting
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_handlers: List[Callable] = []
        self.suppressed_alerts: Set[str] = set()
        
        # Monitoring threads
        self.monitoring_thread: Optional[threading.Thread] = None
        self.health_check_threads: Dict[str, threading.Thread] = {}
        
        # Initialize built-in health checks
        self._initialize_builtin_health_checks()
    
    def _initialize_builtin_health_checks(self) -> None:
        """Initialize built-in system health checks"""
        
        # CPU utilization check
        self.register_health_check(HealthCheck(
            name="cpu_utilization",
            check_function=lambda: psutil.cpu_percent(interval=1) < 90.0,
            interval_seconds=30,
            failure_threshold=3
        ))
        
        # Memory utilization check
        self.register_health_check(HealthCheck(
            name="memory_utilization",
            check_function=lambda: psutil.virtual_memory().percent < 85.0,
            interval_seconds=30,
            failure_threshold=3
        ))
        
        # Disk space check
        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=lambda: psutil.disk_usage('/').percent < 90.0,
            interval_seconds=60,
            failure_threshold=2
        ))
        
        # Process responsiveness check
        self.register_health_check(HealthCheck(
            name="process_responsiveness",
            check_function=self._check_process_responsiveness,
            interval_seconds=15,
            failure_threshold=5
        ))
        
        # Garbage collection health check
        self.register_health_check(HealthCheck(
            name="garbage_collection",
            check_function=self._check_gc_health,
            interval_seconds=60,
            failure_threshold=3
        ))
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a new health check"""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = HealthStatus.HEALTHY
        logger.info(f"Registered health check: {health_check.name}")
    
    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self) -> None:
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("Monitoring system is already running")
            return
        
        self.is_running = True
        logger.info("Starting robust monitoring system")
        
        # Start main monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start individual health check threads
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                thread = threading.Thread(
                    target=self._health_check_loop,
                    args=(name, health_check),
                    daemon=True
                )
                thread.start()
                self.health_check_threads[name] = thread
        
        logger.info("Monitoring system started successfully")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        if not self.is_running:
            return
        
        logger.info("Stopping monitoring system")
        self.is_running = False
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        for thread in self.health_check_threads.values():
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Check metric thresholds and generate alerts
                self._check_metric_thresholds()
                
                # Update overall system health
                self._update_system_health()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep before next iteration
                time.sleep(10)  # 10-second monitoring interval
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def _health_check_loop(self, name: str, health_check: HealthCheck) -> None:
        """Execute individual health check in a loop"""
        while self.is_running:
            try:
                # Execute health check with timeout
                start_time = time.time()
                
                try:
                    # Run health check with timeout
                    result = asyncio.wait_for(
                        asyncio.to_thread(health_check.check_function),
                        timeout=health_check.timeout_seconds
                    )
                    
                    if asyncio.iscoroutine(result):
                        result = asyncio.run(result)
                    
                    check_passed = bool(result)
                    
                except asyncio.TimeoutError:
                    check_passed = False
                    logger.warning(f"Health check '{name}' timed out")
                except Exception as e:
                    check_passed = False
                    logger.error(f"Health check '{name}' failed with error: {e}")
                
                execution_time = time.time() - start_time
                
                # Update health status
                self._update_health_status(name, check_passed, execution_time)
                
                # Sleep until next check
                time.sleep(health_check.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health check loop for '{name}': {e}")
                time.sleep(60)  # Longer sleep on error
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.system_metrics.cpu_percent = cpu_percent
            self._record_metric("system.cpu.percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.system_metrics.memory_percent = memory.percent
            self._record_metric("system.memory.percent", memory.percent)
            self._record_metric("system.memory.available", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_percent = disk.percent
            self._record_metric("system.disk.percent", disk.percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.system_metrics.network_bytes_sent = network.bytes_sent
            self.system_metrics.network_bytes_recv = network.bytes_recv
            self._record_metric("system.network.bytes_sent", network.bytes_sent)
            self._record_metric("system.network.bytes_recv", network.bytes_recv)
            
            # Process metrics
            self.system_metrics.process_count = len(psutil.pids())
            self.system_metrics.uptime_seconds = time.time() - self.start_time
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()[0]
                self.system_metrics.load_average = load_avg
                self._record_metric("system.load_average", load_avg)
            except AttributeError:
                pass  # getloadavg not available on Windows
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self) -> None:
        """Collect application-level metrics"""
        try:
            # Get metrics from collector
            app_metrics = self.metrics_collector.get_current_metrics()
            
            # Update application metrics
            if app_metrics:
                self.application_metrics.requests_total = app_metrics.get("requests_total", 0)
                self.application_metrics.requests_per_second = app_metrics.get("requests_per_second", 0.0)
                self.application_metrics.response_time_avg = app_metrics.get("response_time_avg", 0.0)
                self.application_metrics.error_rate = app_metrics.get("error_rate", 0.0)
                
                # Record metrics
                for key, value in app_metrics.items():
                    self._record_metric(f"application.{key}", value)
        
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def _record_metric(self, name: str, value: Union[int, float], timestamp: Optional[datetime] = None) -> None:
        """Record a metric value with timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_data = {
            "value": value,
            "timestamp": timestamp.isoformat()
        }
        
        self.metric_history[name].append(metric_data)
        self.custom_metrics[name] = value
    
    def _check_metric_thresholds(self) -> None:
        """Check metrics against thresholds and generate alerts"""
        thresholds = self.config.get("alert_thresholds", {})
        
        for metric_name, current_value in self.custom_metrics.items():
            threshold_config = thresholds.get(metric_name, {})
            
            # Check critical threshold
            critical_threshold = threshold_config.get("critical")
            if critical_threshold and current_value >= critical_threshold:
                self._create_alert(
                    name=f"{metric_name}_critical",
                    message=f"{metric_name} is critical: {current_value} >= {critical_threshold}",
                    severity=AlertSeverity.CRITICAL,
                    source_component="monitoring",
                    metadata={"metric": metric_name, "value": current_value, "threshold": critical_threshold}
                )
            
            # Check warning threshold
            warning_threshold = threshold_config.get("warning")
            if warning_threshold and current_value >= warning_threshold and not (critical_threshold and current_value >= critical_threshold):
                self._create_alert(
                    name=f"{metric_name}_warning",
                    message=f"{metric_name} is high: {current_value} >= {warning_threshold}",
                    severity=AlertSeverity.WARNING,
                    source_component="monitoring",
                    metadata={"metric": metric_name, "value": current_value, "threshold": warning_threshold}
                )
    
    def _update_health_status(self, check_name: str, passed: bool, execution_time: float) -> None:
        """Update health status for a specific check"""
        health_check = self.health_checks.get(check_name)
        if not health_check:
            return
        
        if passed:
            self.health_successes[check_name] += 1
            self.health_failures[check_name] = 0  # Reset failure count
            
            # Check if we should recover from unhealthy state
            if (self.health_status[check_name] != HealthStatus.HEALTHY and 
                self.health_successes[check_name] >= health_check.recovery_threshold):
                
                old_status = self.health_status[check_name]
                self.health_status[check_name] = HealthStatus.HEALTHY
                
                logger.info(f"Health check '{check_name}' recovered: {old_status.value} -> healthy")
                
                # Create recovery alert
                self._create_alert(
                    name=f"{check_name}_recovered",
                    message=f"Health check '{check_name}' has recovered",
                    severity=AlertSeverity.INFO,
                    source_component="health_monitoring",
                    metadata={"check_name": check_name, "execution_time": execution_time}
                )
        
        else:
            self.health_failures[check_name] += 1
            self.health_successes[check_name] = 0  # Reset success count
            
            # Check if we should mark as unhealthy
            if (self.health_status[check_name] == HealthStatus.HEALTHY and 
                self.health_failures[check_name] >= health_check.failure_threshold):
                
                self.health_status[check_name] = HealthStatus.UNHEALTHY
                
                logger.warning(f"Health check '{check_name}' marked as unhealthy after {health_check.failure_threshold} failures")
                
                # Create failure alert
                self._create_alert(
                    name=f"{check_name}_unhealthy",
                    message=f"Health check '{check_name}' is unhealthy after {health_check.failure_threshold} consecutive failures",
                    severity=AlertSeverity.ERROR,
                    source_component="health_monitoring",
                    metadata={"check_name": check_name, "failure_count": self.health_failures[check_name]}
                )
    
    def _update_system_health(self) -> None:
        """Update overall system health based on individual health checks"""
        healthy_checks = sum(1 for status in self.health_status.values() if status == HealthStatus.HEALTHY)
        total_checks = len(self.health_status)
        
        if total_checks == 0:
            overall_status = HealthStatus.HEALTHY
        elif healthy_checks == total_checks:
            overall_status = HealthStatus.HEALTHY
        elif healthy_checks >= total_checks * 0.8:  # 80% healthy
            overall_status = HealthStatus.DEGRADED
        elif healthy_checks >= total_checks * 0.5:  # 50% healthy
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.CRITICAL
        
        # Record overall health metric
        self._record_metric("system.health.overall", overall_status.value)
        self._record_metric("system.health.healthy_checks", healthy_checks)
        self._record_metric("system.health.total_checks", total_checks)
    
    def _create_alert(self, name: str, message: str, severity: AlertSeverity, 
                     source_component: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create and process a new alert"""
        
        # Check if alert is suppressed
        if name in self.suppressed_alerts:
            return
        
        # Check for duplicate alerts (deduplication)
        existing_alert = self.alerts.get(name)
        if existing_alert and not existing_alert.resolved:
            # Update existing alert timestamp
            existing_alert.timestamp = datetime.now()
            return
        
        # Create new alert
        alert = Alert(
            id=name,
            name=name,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            source_component=source_component,
            metadata=metadata or {}
        )
        
        self.alerts[name] = alert
        self.alert_history.append(alert)
        
        # Log the alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"ALERT [{severity.value.upper()}] {name}: {message}")
        
        # Send to alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up old alerts
        resolved_alerts_to_remove = []
        for alert_id, alert in self.alerts.items():
            if alert.resolved and alert.resolved_timestamp and alert.resolved_timestamp < cutoff_time:
                resolved_alerts_to_remove.append(alert_id)
        
        for alert_id in resolved_alerts_to_remove:
            del self.alerts[alert_id]
    
    def _check_process_responsiveness(self) -> bool:
        """Check if the current process is responsive"""
        try:
            # Simple responsiveness test - try to allocate memory and perform basic operations
            test_data = list(range(1000))
            test_sum = sum(test_data)
            return test_sum == 499500  # Expected sum
        except Exception:
            return False
    
    def _check_gc_health(self) -> bool:
        """Check garbage collection health"""
        try:
            gc_stats = gc.get_stats()
            if gc_stats:
                # Check if GC is running too frequently (indicating memory pressure)
                last_gen_stats = gc_stats[-1]
                collections = last_gen_stats.get('collections', 0)
                # If there are too many collections, it might indicate memory issues
                return collections < 1000  # Arbitrary threshold
            return True
        except Exception:
            return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        return {
            "overall_status": self._get_overall_health_status().value,
            "health_checks": {
                name: status.value for name, status in self.health_status.items()
            },
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "system_metrics": {
                "cpu_percent": self.system_metrics.cpu_percent,
                "memory_percent": self.system_metrics.memory_percent,
                "disk_percent": self.system_metrics.disk_percent,
                "uptime_seconds": self.system_metrics.uptime_seconds
            },
            "application_metrics": {
                "requests_per_second": self.application_metrics.requests_per_second,
                "error_rate": self.application_metrics.error_rate,
                "response_time_avg": self.application_metrics.response_time_avg
            }
        }
    
    def _get_overall_health_status(self) -> HealthStatus:
        """Calculate overall health status"""
        if not self.health_status:
            return HealthStatus.HEALTHY
        
        status_counts = defaultdict(int)
        for status in self.health_status.values():
            status_counts[status] += 1
        
        total = len(self.health_status)
        
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > total * 0.3:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0 or status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def display_monitoring_dashboard(self) -> None:
        """Display real-time monitoring dashboard"""
        def make_dashboard():
            # System metrics table
            system_table = Table(title="System Metrics")
            system_table.add_column("Metric", style="cyan")
            system_table.add_column("Value", style="magenta")
            system_table.add_column("Status", style="green")
            
            system_table.add_row("CPU Usage", f"{self.system_metrics.cpu_percent:.1f}%", "✅" if self.system_metrics.cpu_percent < 80 else "⚠️")
            system_table.add_row("Memory Usage", f"{self.system_metrics.memory_percent:.1f}%", "✅" if self.system_metrics.memory_percent < 80 else "⚠️")
            system_table.add_row("Disk Usage", f"{self.system_metrics.disk_percent:.1f}%", "✅" if self.system_metrics.disk_percent < 80 else "⚠️")
            system_table.add_row("Uptime", f"{self.system_metrics.uptime_seconds/3600:.1f} hours", "✅")
            
            # Health checks table
            health_table = Table(title="Health Checks")
            health_table.add_column("Check", style="cyan")
            health_table.add_column("Status", style="magenta")
            health_table.add_column("Last Check", style="green")
            
            for name, status in self.health_status.items():
                status_icon = {
                    HealthStatus.HEALTHY: "✅",
                    HealthStatus.DEGRADED: "⚠️",
                    HealthStatus.UNHEALTHY: "❌",
                    HealthStatus.CRITICAL: "🚨",
                    HealthStatus.DOWN: "💀"
                }.get(status, "❓")
                
                health_table.add_row(name, f"{status_icon} {status.value}", "Just now")
            
            # Active alerts table
            alerts_table = Table(title="Active Alerts")
            alerts_table.add_column("Alert", style="cyan")
            alerts_table.add_column("Severity", style="magenta")
            alerts_table.add_column("Time", style="green")
            
            active_alerts = [a for a in self.alerts.values() if not a.resolved]
            for alert in active_alerts[-10:]:  # Show last 10 alerts
                severity_icon = {
                    AlertSeverity.INFO: "ℹ️",
                    AlertSeverity.WARNING: "⚠️",
                    AlertSeverity.ERROR: "❌",
                    AlertSeverity.CRITICAL: "🚨",
                    AlertSeverity.EMERGENCY: "💀"
                }.get(alert.severity, "❓")
                
                time_str = alert.timestamp.strftime("%H:%M:%S")
                alerts_table.add_row(alert.name, f"{severity_icon} {alert.severity.value}", time_str)
            
            if not active_alerts:
                alerts_table.add_row("No active alerts", "✅ All clear", "")
            
            return Panel.fit(
                f"{system_table}\n\n{health_table}\n\n{alerts_table}",
                title="🔍 Robust Monitoring Dashboard",
                border_style="blue"
            )
        
        console.print("Starting monitoring dashboard... Press Ctrl+C to exit")
        
        try:
            with Live(make_dashboard(), refresh_per_second=1) as live:
                while self.is_running:
                    live.update(make_dashboard())
                    time.sleep(1)
        except KeyboardInterrupt:
            console.print("\nMonitoring dashboard stopped by user")


# Built-in alert handlers
def console_alert_handler(alert: Alert) -> None:
    """Simple console alert handler"""
    severity_colors = {
        AlertSeverity.INFO: "blue",
        AlertSeverity.WARNING: "yellow",
        AlertSeverity.ERROR: "red",
        AlertSeverity.CRITICAL: "bold red",
        AlertSeverity.EMERGENCY: "bold red on white"
    }
    
    color = severity_colors.get(alert.severity, "white")
    console.print(f"[{color}]ALERT: {alert.message}[/{color}]")


def file_alert_handler(alert: Alert, log_file: str = "alerts.log") -> None:
    """File-based alert handler"""
    try:
        with open(log_file, "a") as f:
            f.write(f"{alert.timestamp.isoformat()} [{alert.severity.value.upper()}] {alert.name}: {alert.message}\n")
    except Exception as e:
        logger.error(f"Failed to write alert to file: {e}")