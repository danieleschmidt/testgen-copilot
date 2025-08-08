"""Quantum-aware monitoring and observability for task execution."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import psutil


class AlertSeverity(Enum):
    """Alert severity levels based on quantum energy states."""
    CRITICAL = "critical"      # System failure imminent
    HIGH = "high"             # Performance degradation
    MEDIUM = "medium"         # Potential issues
    LOW = "low"              # Informational
    INFO = "info"            # Normal operations


@dataclass
class QuantumMetric:
    """Metric with quantum uncertainty and coherence tracking."""

    name: str
    value: float
    timestamp: datetime
    uncertainty: float = 0.0      # Quantum uncertainty in measurement
    coherence_time: float = 30.0  # How long metric remains valid (seconds)
    quantum_state: str = "stable" # stable, fluctuating, decoherent

    def is_coherent(self) -> bool:
        """Check if metric is still within coherence time."""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age <= self.coherence_time

    def apply_uncertainty(self) -> float:
        """Apply quantum uncertainty to metric value."""
        import random
        uncertainty_factor = random.uniform(1 - self.uncertainty, 1 + self.uncertainty)
        return self.value * uncertainty_factor


@dataclass
class QuantumAlert:
    """Alert with quantum properties and entanglement tracking."""

    id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source_metric: str
    entangled_alerts: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    quantum_signature: Dict[str, Any] = field(default_factory=dict)

    def entangle_with(self, other_alert_id: str):
        """Create quantum entanglement with another alert."""
        if other_alert_id not in self.entangled_alerts:
            self.entangled_alerts.append(other_alert_id)

    def resolve(self):
        """Resolve alert and collapse quantum state."""
        self.resolved = True
        self.resolution_time = datetime.now(timezone.utc)


class QuantumHealthChecker:
    """Quantum-aware health monitoring with coherence tracking."""

    def __init__(self, coherence_threshold: float = 0.8):
        """Initialize quantum health checker."""
        self.coherence_threshold = coherence_threshold
        self.health_metrics: Dict[str, QuantumMetric] = {}
        self.logger = logging.getLogger(__name__)

    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive quantum-aware health check."""

        health_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "quantum_coherence": 1.0,
            "system_metrics": {},
            "quantum_metrics": {},
            "alerts": []
        }

        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Create quantum metrics with uncertainty
        self.health_metrics["cpu_usage"] = QuantumMetric(
            "cpu_usage",
            cpu_percent,
            datetime.now(timezone.utc),
            uncertainty=0.05,  # 5% measurement uncertainty
            coherence_time=10.0
        )

        self.health_metrics["memory_usage"] = QuantumMetric(
            "memory_usage",
            memory.percent,
            datetime.now(timezone.utc),
            uncertainty=0.02,
            coherence_time=15.0
        )

        self.health_metrics["disk_usage"] = QuantumMetric(
            "disk_usage",
            disk.percent,
            datetime.now(timezone.utc),
            uncertainty=0.01,
            coherence_time=60.0
        )

        # Calculate quantum coherence
        coherent_metrics = sum(1 for m in self.health_metrics.values() if m.is_coherent())
        total_metrics = len(self.health_metrics)
        quantum_coherence = coherent_metrics / total_metrics if total_metrics > 0 else 1.0

        health_report["quantum_coherence"] = quantum_coherence

        # Populate system metrics
        health_report["system_metrics"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }

        # Populate quantum metrics with uncertainty
        health_report["quantum_metrics"] = {
            name: {
                "value": metric.value,
                "uncertainty": metric.uncertainty,
                "coherent": metric.is_coherent(),
                "quantum_state": metric.quantum_state,
                "coherence_time": metric.coherence_time
            }
            for name, metric in self.health_metrics.items()
        }

        # Generate alerts based on thresholds
        alerts = self._generate_health_alerts()
        health_report["alerts"] = [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source_metric": alert.source_metric
            }
            for alert in alerts
        ]

        # Overall status based on alerts and coherence
        if any(alert.severity == AlertSeverity.CRITICAL for alert in alerts):
            health_report["overall_status"] = "critical"
        elif any(alert.severity == AlertSeverity.HIGH for alert in alerts):
            health_report["overall_status"] = "degraded"
        elif quantum_coherence < self.coherence_threshold:
            health_report["overall_status"] = "incoherent"
        else:
            health_report["overall_status"] = "healthy"

        return health_report

    def _generate_health_alerts(self) -> List[QuantumAlert]:
        """Generate alerts based on quantum metrics."""
        alerts = []

        for name, metric in self.health_metrics.items():
            # Apply quantum uncertainty to thresholds
            uncertain_value = metric.apply_uncertainty()

            if name == "cpu_usage" and uncertain_value > 90:
                alert = QuantumAlert(
                    id=f"cpu_critical_{int(time.time())}",
                    severity=AlertSeverity.CRITICAL,
                    message=f"CPU usage critical: {uncertain_value:.1f}%",
                    timestamp=datetime.now(timezone.utc),
                    source_metric=name
                )
                alerts.append(alert)

            elif name == "memory_usage" and uncertain_value > 85:
                alert = QuantumAlert(
                    id=f"memory_high_{int(time.time())}",
                    severity=AlertSeverity.HIGH,
                    message=f"Memory usage high: {uncertain_value:.1f}%",
                    timestamp=datetime.now(timezone.utc),
                    source_metric=name
                )
                alerts.append(alert)

            elif name == "disk_usage" and uncertain_value > 95:
                alert = QuantumAlert(
                    id=f"disk_critical_{int(time.time())}",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Disk usage critical: {uncertain_value:.1f}%",
                    timestamp=datetime.now(timezone.utc),
                    source_metric=name
                )
                alerts.append(alert)

            # Coherence-based alerts
            if not metric.is_coherent():
                alert = QuantumAlert(
                    id=f"coherence_{name}_{int(time.time())}",
                    severity=AlertSeverity.MEDIUM,
                    message=f"Metric {name} lost quantum coherence",
                    timestamp=datetime.now(timezone.utc),
                    source_metric=name
                )
                alerts.append(alert)

        return alerts


class QuantumCircuitBreaker:
    """Quantum-aware circuit breaker with superposition states."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=5),
        half_open_max_calls: int = 3
    ):
        """Initialize quantum circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0

        self.logger = logging.getLogger(__name__)

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through quantum circuit breaker."""

        # Check circuit state
        if self.state == "open":
            if self._should_attempt_recovery():
                self.state = "half_open"
                self.half_open_calls = 0
                self.logger.info("Circuit breaker transitioning to half-open state")
            else:
                raise CircuitBreakerException("Circuit breaker is open")

        elif self.state == "half_open":
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerException("Circuit breaker half-open call limit exceeded")

        try:
            # Execute with quantum uncertainty monitoring
            start_time = time.time()
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Success - reset failure count
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("Circuit breaker recovered to closed state")
            elif self.state == "closed":
                self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery

            self.logger.debug(f"Circuit breaker call succeeded in {execution_time:.3f}s")
            return result

        except Exception as e:
            # Failure - increment count and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)

            if self.state == "half_open":
                self.state = "open"
                self.logger.warning("Circuit breaker failed during half-open, returning to open state")
            elif self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            self.logger.error(f"Circuit breaker recorded failure: {e}")
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure >= self.recovery_timeout

    def force_open(self):
        """Manually open circuit breaker."""
        self.state = "open"
        self.last_failure_time = datetime.now(timezone.utc)
        self.logger.warning("Circuit breaker manually opened")

    def force_close(self):
        """Manually close circuit breaker."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        self.logger.info("Circuit breaker manually closed")


class QuantumMetricsCollector:
    """Advanced metrics collection with quantum sampling and uncertainty."""

    def __init__(self, sampling_rate: float = 1.0, max_history: int = 1000):
        """Initialize quantum metrics collector."""
        self.sampling_rate = sampling_rate
        self.max_history = max_history

        self.metrics_history: Dict[str, Deque[QuantumMetric]] = defaultdict(lambda: deque(maxlen=max_history))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(__name__)

    async def start_collection(self):
        """Start continuous metrics collection."""
        if self.collection_active:
            self.logger.warning("Metrics collection already active")
            return

        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collect_metrics_loop())
        self.logger.info("Started quantum metrics collection")

    async def stop_collection(self):
        """Stop metrics collection gracefully."""
        self.collection_active = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped quantum metrics collection")

    async def _collect_metrics_loop(self):
        """Main metrics collection loop with quantum sampling."""
        try:
            while self.collection_active:
                # Quantum sampling - only collect if probability threshold met
                if random.random() < self.sampling_rate:
                    await self._collect_system_metrics()
                    self._update_aggregated_metrics()

                # Wait with quantum jitter to avoid measurement bias
                base_interval = 5.0  # 5 seconds
                jitter = random.uniform(0.8, 1.2)
                await asyncio.sleep(base_interval * jitter)

        except asyncio.CancelledError:
            self.logger.info("Metrics collection loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Metrics collection loop failed: {e}")
            self.collection_active = False

    async def _collect_system_metrics(self):
        """Collect system metrics with quantum uncertainty."""
        timestamp = datetime.now(timezone.utc)

        # CPU metrics with quantum uncertainty
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_metric = QuantumMetric(
            "cpu_usage_percent",
            cpu_percent,
            timestamp,
            uncertainty=0.05,  # 5% uncertainty
            coherence_time=10.0
        )
        self.metrics_history["cpu_usage_percent"].append(cpu_metric)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metric = QuantumMetric(
            "memory_usage_percent",
            memory.percent,
            timestamp,
            uncertainty=0.02,
            coherence_time=15.0
        )
        self.metrics_history["memory_usage_percent"].append(memory_metric)

        # Network I/O with higher uncertainty
        net_io = psutil.net_io_counters()
        if hasattr(net_io, 'bytes_sent'):
            network_metric = QuantumMetric(
                "network_bytes_sent",
                float(net_io.bytes_sent),
                timestamp,
                uncertainty=0.1,  # Higher uncertainty for network
                coherence_time=5.0
            )
            self.metrics_history["network_bytes_sent"].append(network_metric)

        # Task-specific quantum metrics
        await self._collect_quantum_task_metrics(timestamp)

    async def _collect_quantum_task_metrics(self, timestamp: datetime):
        """Collect quantum-specific task execution metrics."""

        # Quantum coherence metric
        coherence_metric = QuantumMetric(
            "quantum_coherence",
            self._calculate_system_coherence(),
            timestamp,
            uncertainty=0.15,  # Coherence is inherently uncertain
            coherence_time=20.0
        )
        self.metrics_history["quantum_coherence"].append(coherence_metric)

        # Entanglement density
        entanglement_metric = QuantumMetric(
            "entanglement_density",
            self._calculate_entanglement_density(),
            timestamp,
            uncertainty=0.1,
            coherence_time=30.0
        )
        self.metrics_history["entanglement_density"].append(entanglement_metric)

    def _calculate_system_coherence(self) -> float:
        """Calculate overall quantum coherence of the system."""
        import random

        # Simulate quantum coherence based on system stability
        coherent_metrics = sum(
            1 for metrics in self.metrics_history.values()
            for metric in metrics
            if metric.is_coherent()
        )

        total_metrics = sum(len(metrics) for metrics in self.metrics_history.values())

        if total_metrics == 0:
            return 1.0

        base_coherence = coherent_metrics / total_metrics

        # Add quantum fluctuations
        fluctuation = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_coherence + fluctuation))

    def _calculate_entanglement_density(self) -> float:
        """Calculate density of quantum entanglements in the system."""
        # This would be calculated from actual task entanglements
        # For now, simulate based on system complexity
        import random

        metric_count = sum(len(metrics) for metrics in self.metrics_history.values())
        base_density = min(metric_count / 100.0, 1.0)  # Normalize to 0-1

        # Add quantum uncertainty
        uncertainty = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, base_density + uncertainty))

    def _update_aggregated_metrics(self):
        """Update aggregated quantum metrics for reporting."""
        for metric_name, history in self.metrics_history.items():
            if not history:
                continue

            # Calculate quantum statistics
            recent_values = [m.value for m in list(history)[-10:]]  # Last 10 measurements

            self.aggregated_metrics[metric_name] = {
                "mean": sum(recent_values) / len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "current": recent_values[-1] if recent_values else 0.0,
                "trend": self._calculate_trend(recent_values),
                "quantum_uncertainty": sum(m.uncertainty for m in list(history)[-10:]) / min(len(history), 10)
            }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction with quantum smoothing."""
        if len(values) < 3:
            return "stable"

        # Simple linear trend with quantum smoothing
        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:3]) / 3

        diff_threshold = 0.05  # 5% change threshold

        if recent_avg > older_avg * (1 + diff_threshold):
            return "increasing"
        elif recent_avg < older_avg * (1 - diff_threshold):
            return "decreasing"
        else:
            return "stable"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected quantum metrics."""
        return {
            "collection_active": self.collection_active,
            "total_metrics": len(self.metrics_history),
            "total_samples": sum(len(history) for history in self.metrics_history.values()),
            "aggregated_metrics": self.aggregated_metrics.copy(),
            "quantum_coherence": self._calculate_system_coherence(),
            "entanglement_density": self._calculate_entanglement_density()
        }


class QuantumAlertManager:
    """Manages quantum alerts with entanglement and correlation detection."""

    def __init__(self, alert_history_limit: int = 500):
        """Initialize quantum alert manager."""
        self.alerts: Dict[str, QuantumAlert] = {}
        self.alert_history: Deque[QuantumAlert] = deque(maxlen=alert_history_limit)
        self.alert_callbacks: List[Callable[[QuantumAlert], None]] = []

        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def add_alert_callback(self, callback: Callable[[QuantumAlert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        message: str,
        source_metric: str,
        quantum_signature: Optional[Dict[str, Any]] = None
    ) -> QuantumAlert:
        """Create new quantum alert with entanglement detection."""

        with self._lock:
            alert = QuantumAlert(
                id=alert_id,
                severity=severity,
                message=message,
                timestamp=datetime.now(timezone.utc),
                source_metric=source_metric,
                quantum_signature=quantum_signature or {}
            )

            # Detect entanglements with existing alerts
            self._detect_alert_entanglements(alert)

            # Store alert
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")

            self.logger.info(f"Created quantum alert: {alert_id} [{severity.value}]")
            return alert

    def _detect_alert_entanglements(self, new_alert: QuantumAlert):
        """Detect quantum entanglements between alerts."""
        for existing_alert in self.alerts.values():
            if existing_alert.resolved:
                continue

            # Entangle alerts from same metric source
            if existing_alert.source_metric == new_alert.source_metric:
                new_alert.entangle_with(existing_alert.id)
                existing_alert.entangle_with(new_alert.id)

            # Entangle alerts with temporal correlation
            time_diff = abs((new_alert.timestamp - existing_alert.timestamp).total_seconds())
            if time_diff < 60:  # Within 1 minute
                correlation_score = self._calculate_alert_correlation(new_alert, existing_alert)
                if correlation_score > 0.7:
                    new_alert.entangle_with(existing_alert.id)
                    existing_alert.entangle_with(new_alert.id)

    def _calculate_alert_correlation(self, alert_a: QuantumAlert, alert_b: QuantumAlert) -> float:
        """Calculate correlation between two alerts."""

        # Same severity increases correlation
        severity_correlation = 1.0 if alert_a.severity == alert_b.severity else 0.5

        # Similar quantum signatures increase correlation
        sig_a = alert_a.quantum_signature
        sig_b = alert_b.quantum_signature

        if not sig_a or not sig_b:
            signature_correlation = 0.5
        else:
            common_keys = set(sig_a.keys()) & set(sig_b.keys())
            if common_keys:
                signature_correlation = len(common_keys) / max(len(sig_a), len(sig_b))
            else:
                signature_correlation = 0.0

        return (severity_correlation + signature_correlation) / 2.0

    def resolve_alert(self, alert_id: str):
        """Resolve alert and handle entangled alerts."""
        with self._lock:
            if alert_id not in self.alerts:
                self.logger.warning(f"Alert not found: {alert_id}")
                return

            alert = self.alerts[alert_id]
            alert.resolve()

            # Check if entangled alerts should also be resolved
            for entangled_id in alert.entangled_alerts:
                if entangled_id in self.alerts:
                    entangled_alert = self.alerts[entangled_id]
                    if not entangled_alert.resolved:
                        # Quantum correlation - high chance of resolving entangled alerts
                        if random.random() < 0.8:  # 80% chance
                            entangled_alert.resolve()
                            self.logger.info(f"Auto-resolved entangled alert: {entangled_id}")

            self.logger.info(f"Resolved quantum alert: {alert_id}")

    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[QuantumAlert]:
        """Get currently active alerts."""
        active = [alert for alert in self.alerts.values() if not alert.resolved]

        if severity_filter:
            active = [alert for alert in active if alert.severity == severity_filter]

        return sorted(active, key=lambda a: (a.severity.value, a.timestamp), reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len([a for a in self.alerts.values() if not a.resolved])

        # Calculate resolution time statistics
        resolved_alerts = [a for a in self.alert_history if a.resolved and a.resolution_time]
        avg_resolution_time = 0.0

        if resolved_alerts:
            resolution_times = [
                (alert.resolution_time - alert.timestamp).total_seconds()
                for alert in resolved_alerts
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)

        # Severity distribution
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "average_resolution_time_seconds": avg_resolution_time,
            "severity_distribution": dict(severity_counts),
            "entanglement_count": sum(len(a.entangled_alerts) for a in self.alerts.values()) // 2
        }


class QuantumMonitoringDashboard:
    """Real-time monitoring dashboard with quantum visualizations."""

    def __init__(self):
        """Initialize quantum monitoring dashboard."""
        self.health_checker = QuantumHealthChecker()
        self.metrics_collector = QuantumMetricsCollector()
        self.alert_manager = QuantumAlertManager()

        self.logger = logging.getLogger(__name__)

        # Setup alert callback
        self.alert_manager.add_alert_callback(self._handle_alert_notification)

    async def start_monitoring(self):
        """Start comprehensive quantum monitoring."""
        self.logger.info("Starting quantum monitoring dashboard")

        await self.metrics_collector.start_collection()

        # Start periodic health checks
        asyncio.create_task(self._periodic_health_checks())

    async def stop_monitoring(self):
        """Stop monitoring gracefully."""
        self.logger.info("Stopping quantum monitoring dashboard")
        await self.metrics_collector.stop_collection()

    async def _periodic_health_checks(self):
        """Run periodic health checks with quantum timing."""
        try:
            while self.metrics_collector.collection_active:
                health_report = self.health_checker.check_system_health()

                # Generate alerts based on health report
                for alert_data in health_report.get("alerts", []):
                    self.alert_manager.create_alert(
                        alert_id=alert_data["id"],
                        severity=AlertSeverity(alert_data["severity"]),
                        message=alert_data["message"],
                        source_metric=alert_data["source_metric"],
                        quantum_signature={"health_check": True}
                    )

                # Quantum-influenced check interval
                base_interval = 30.0  # 30 seconds
                quantum_jitter = random.uniform(0.9, 1.1)
                await asyncio.sleep(base_interval * quantum_jitter)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Health check loop failed: {e}")

    def _handle_alert_notification(self, alert: QuantumAlert):
        """Handle alert notifications with quantum routing."""

        # Log alert with quantum context
        self.logger.warning(
            f"Quantum Alert [{alert.severity.value.upper()}]: {alert.message}",
            extra={
                "alert_id": alert.id,
                "source_metric": alert.source_metric,
                "entangled_alerts": alert.entangled_alerts,
                "quantum_signature": alert.quantum_signature
            }
        )

        # Quantum-based alert routing
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            # High-priority alerts trigger immediate escalation
            self._escalate_alert(alert)

    def _escalate_alert(self, alert: QuantumAlert):
        """Escalate critical alerts using quantum channels."""

        escalation_data = {
            "alert": {
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            },
            "quantum_context": {
                "entangled_alerts": alert.entangled_alerts,
                "quantum_signature": alert.quantum_signature
            },
            "system_context": self.health_checker.check_system_health()
        }

        # Log escalation
        self.logger.critical(
            f"ESCALATED: {alert.message}",
            extra=escalation_data
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for visualization."""

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_status": self.health_checker.check_system_health(),
            "metrics_summary": self.metrics_collector.get_metrics_summary(),
            "alert_statistics": self.alert_manager.get_alert_statistics(),
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "age_seconds": (datetime.now(timezone.utc) - alert.timestamp).total_seconds(),
                    "entangled_count": len(alert.entangled_alerts)
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "quantum_state": {
                "coherence": self.health_checker.health_metrics.get("quantum_coherence",
                    QuantumMetric("coherence", 1.0, datetime.now(timezone.utc))).value,
                "entanglement_density": self._calculate_current_entanglement_density()
            }
        }

    def _calculate_current_entanglement_density(self) -> float:
        """Calculate current entanglement density from active alerts."""
        if not self.alert_manager.alerts:
            return 0.0

        total_entanglements = sum(len(alert.entangled_alerts) for alert in self.alert_manager.alerts.values())
        total_alerts = len(self.alert_manager.alerts)

        return total_entanglements / (total_alerts * 2) if total_alerts > 0 else 0.0

    async def export_monitoring_data(self, output_path: str, time_range: Optional[timedelta] = None):
        """Export monitoring data for analysis."""

        time_range = time_range or timedelta(hours=24)
        cutoff_time = datetime.now(timezone.utc) - time_range

        export_data = {
            "export_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range_hours": time_range.total_seconds() / 3600,
                "export_type": "quantum_monitoring"
            },
            "metrics_data": {},
            "alerts_data": [],
            "quantum_analysis": {}
        }

        # Export metrics within time range
        for metric_name, history in self.metrics_collector.metrics_history.items():
            recent_metrics = [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value,
                    "uncertainty": metric.uncertainty,
                    "coherent": metric.is_coherent(),
                    "quantum_state": metric.quantum_state
                }
                for metric in history
                if metric.timestamp >= cutoff_time
            ]
            export_data["metrics_data"][metric_name] = recent_metrics

        # Export alerts within time range
        export_data["alerts_data"] = [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source_metric": alert.source_metric,
                "entangled_alerts": alert.entangled_alerts,
                "resolved": alert.resolved,
                "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
            }
            for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

        # Quantum analysis
        export_data["quantum_analysis"] = {
            "average_coherence": self._calculate_average_coherence(cutoff_time),
            "entanglement_patterns": self._analyze_entanglement_patterns(),
            "quantum_efficiency": self._calculate_quantum_efficiency()
        }

        # Write to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Monitoring data exported to {output_path}")
        return output_path_obj

    def _calculate_average_coherence(self, cutoff_time: datetime) -> float:
        """Calculate average quantum coherence over time period."""
        coherence_metrics = self.metrics_collector.metrics_history.get("quantum_coherence", deque())
        recent_coherence = [
            metric.value for metric in coherence_metrics
            if metric.timestamp >= cutoff_time
        ]

        return sum(recent_coherence) / len(recent_coherence) if recent_coherence else 1.0

    def _analyze_entanglement_patterns(self) -> Dict[str, Any]:
        """Analyze quantum entanglement patterns in alerts."""

        entanglement_graph = defaultdict(set)

        for alert in self.alerts.values():
            for entangled_id in alert.entangled_alerts:
                entanglement_graph[alert.id].add(entangled_id)

        # Find strongly connected components (entanglement clusters)
        clusters = []
        visited = set()

        for alert_id in entanglement_graph:
            if alert_id not in visited:
                cluster = self._dfs_cluster(alert_id, entanglement_graph, visited)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return {
            "total_entanglements": sum(len(entangled) for entangled in entanglement_graph.values()) // 2,
            "entanglement_clusters": clusters,
            "largest_cluster_size": max(len(cluster) for cluster in clusters) if clusters else 0,
            "cluster_count": len(clusters)
        }

    def _dfs_cluster(self, start_id: str, graph: Dict[str, Set[str]], visited: Set[str]) -> List[str]:
        """Find entanglement cluster using depth-first search."""
        cluster = []
        stack = [start_id]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                cluster.append(node)

                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return cluster

    def _calculate_quantum_efficiency(self) -> float:
        """Calculate overall quantum efficiency of the monitoring system."""

        # Base efficiency from coherence
        coherence_history = self.metrics_collector.metrics_history.get("quantum_coherence", deque())
        if coherence_history:
            avg_coherence = sum(m.value for m in coherence_history) / len(coherence_history)
        else:
            avg_coherence = 1.0

        # Efficiency boost from successful entanglements
        resolved_entangled_alerts = sum(
            1 for alert in self.alert_history
            if alert.resolved and alert.entangled_alerts
        )
        total_entangled_alerts = sum(
            1 for alert in self.alert_history
            if alert.entangled_alerts
        )

        entanglement_efficiency = (
            resolved_entangled_alerts / total_entangled_alerts
            if total_entangled_alerts > 0 else 1.0
        )

        # Combined quantum efficiency
        return (avg_coherence + entanglement_efficiency) / 2.0


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Factory function for easy dashboard creation
def create_quantum_monitoring_dashboard() -> QuantumMonitoringDashboard:
    """Create quantum monitoring dashboard with default configuration."""
    return QuantumMonitoringDashboard()


# Demo function for monitoring capabilities
async def demo_quantum_monitoring():
    """Demonstrate quantum monitoring capabilities."""

    dashboard = create_quantum_monitoring_dashboard()

    try:
        # Start monitoring
        await dashboard.start_monitoring()

        # Simulate some activity and alerts
        for i in range(5):
            # Create test alerts
            dashboard.alert_manager.create_alert(
                f"demo_alert_{i}",
                AlertSeverity.MEDIUM,
                f"Demo alert {i} for monitoring test",
                "demo_metric",
                {"demo": True, "iteration": i}
            )

            await asyncio.sleep(1)

        # Let monitoring run for a bit
        await asyncio.sleep(5)

        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()

        # Export monitoring data
        export_path = await dashboard.export_monitoring_data("quantum_monitoring_demo.json")

        print("Quantum monitoring demo completed")
        print(f"Active alerts: {len(dashboard_data['active_alerts'])}")
        print(f"Quantum coherence: {dashboard_data['quantum_state']['coherence']:.2f}")
        print(f"Monitoring data exported to: {export_path}")

        return dashboard_data

    finally:
        await dashboard.stop_monitoring()


if __name__ == "__main__":
    import random
    asyncio.run(demo_quantum_monitoring())
