"""Comprehensive health monitoring and system status tracking."""

from __future__ import annotations

import asyncio
import json
import psutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from .logging_config import get_core_logger


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of system components to monitor."""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Union[str, int, float]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemMetrics:
    """System performance and resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]  # 1, 5, 15 minute load averages


class HealthMonitor:
    """Comprehensive system health monitoring and alerting."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = get_core_logger()
        self.config = self._load_config(config_path)
        self.health_checks: Dict[str, callable] = {}
        self.last_metrics: Optional[SystemMetrics] = None
        self.alerts_sent: Dict[str, datetime] = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load health monitoring configuration."""
        default_config = {
            "thresholds": {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_usage_percent": 90.0,
                "response_time_ms": 5000.0
            },
            "alert_cooldown_minutes": 15,
            "metrics_retention_days": 7,
            "health_check_interval_seconds": 60
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded health monitoring config from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def register_health_check(self, name: str, check_func: callable, component_type: ComponentType = ComponentType.APPLICATION):
        """Register a custom health check function."""
        self.health_checks[name] = {
            "func": check_func,
            "type": component_type
        }
        self.logger.info(f"Registered health check: {name}")
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_health_check("system_resources", self._check_system_resources, ComponentType.SYSTEM)
        self.register_health_check("disk_space", self._check_disk_space, ComponentType.FILE_SYSTEM)
        self.register_health_check("memory_usage", self._check_memory_usage, ComponentType.SYSTEM)
        self.register_health_check("cpu_usage", self._check_cpu_usage, ComponentType.SYSTEM)
    
    async def run_all_health_checks(self) -> List[HealthCheckResult]:
        """Execute all registered health checks."""
        self.logger.info("Running comprehensive health checks")
        
        results = []
        for name, check_info in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_info["func"]):
                    result = await check_info["func"]()
                else:
                    result = check_info["func"]()
                
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                if isinstance(result, HealthCheckResult):
                    result.response_time_ms = response_time
                    results.append(result)
                else:
                    # Handle simple return values
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    results.append(HealthCheckResult(
                        component_name=name,
                        component_type=check_info["type"],
                        status=status,
                        message="Health check completed",
                        timestamp=datetime.now(timezone.utc),
                        response_time_ms=response_time
                    ))
                
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                results.append(HealthCheckResult(
                    component_name=name,
                    component_type=check_info["type"],
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(timezone.utc),
                    response_time_ms=0.0
                ))
        
        # Check for overall system health
        overall_status = self._determine_overall_health(results)
        self.logger.info(f"Overall system health: {overall_status.value}")
        
        return results
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_average = [0.0, 0.0, 0.0]  # Fallback for Windows
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            # Return minimal metrics in case of error
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource health."""
        metrics = self.collect_system_metrics()
        
        issues = []
        status = HealthStatus.HEALTHY
        
        # Check CPU usage
        if metrics.cpu_percent > self.config["thresholds"]["cpu_percent"]:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            status = HealthStatus.DEGRADED
        
        # Check memory usage
        if metrics.memory_percent > self.config["thresholds"]["memory_percent"]:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            status = HealthStatus.DEGRADED
        
        # Check disk usage
        if metrics.disk_usage_percent > self.config["thresholds"]["disk_usage_percent"]:
            issues.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
            status = HealthStatus.UNHEALTHY
        
        # Determine final status
        if len(issues) >= 2:
            status = HealthStatus.UNHEALTHY
        elif len(issues) >= 3:
            status = HealthStatus.CRITICAL
        
        message = "System resources healthy" if not issues else "; ".join(issues)
        
        return HealthCheckResult(
            component_name="system_resources",
            component_type=ComponentType.SYSTEM,
            status=status,
            message=message,
            timestamp=datetime.now(timezone.utc),
            response_time_ms=0.0,
            metadata={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_usage_percent": metrics.disk_usage_percent,
                "process_count": metrics.process_count
            }
        )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            free_gb = disk.free / (1024**3)
            
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk space: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
            elif usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Low disk space: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
            elif usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk space warning: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
            
            return HealthCheckResult(
                component_name="disk_space",
                component_type=ComponentType.FILE_SYSTEM,
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                response_time_ms=0.0,
                metadata={
                    "usage_percent": usage_percent,
                    "free_gb": free_gb,
                    "total_gb": disk.total / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component_name="disk_space",
                component_type=ComponentType.FILE_SYSTEM,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                response_time_ms=0.0
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent > 85:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory.percent:.1f}%"
            elif memory.percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return HealthCheckResult(
                component_name="memory_usage",
                component_type=ComponentType.SYSTEM,
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                response_time_ms=0.0,
                metadata={
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component_name="memory_usage",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                response_time_ms=0.0
            )
    
    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                component_name="cpu_usage",
                component_type=ComponentType.SYSTEM,
                status=status,
                message=message,
                timestamp=datetime.now(timezone.utc),
                response_time_ms=0.0,
                metadata={
                    "percent": cpu_percent,
                    "core_count": psutil.cpu_count()
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component_name="cpu_usage",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check CPU usage: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                response_time_ms=0.0
            )
    
    def _determine_overall_health(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health from individual check results."""
        if not results:
            return HealthStatus.UNHEALTHY
        
        status_counts = {status: 0 for status in HealthStatus}
        for result in results:
            status_counts[result.status] += 1
        
        # Priority order: CRITICAL > UNHEALTHY > DEGRADED > HEALTHY
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def save_health_report(self, results: List[HealthCheckResult], output_path: Path):
        """Save health check results to a JSON report."""
        try:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_status": self._determine_overall_health(results).value,
                "system_metrics": asdict(self.last_metrics) if self.last_metrics else None,
                "health_checks": [asdict(result) for result in results],
                "summary": {
                    "total_checks": len(results),
                    "healthy": sum(1 for r in results if r.status == HealthStatus.HEALTHY),
                    "degraded": sum(1 for r in results if r.status == HealthStatus.DEGRADED),
                    "unhealthy": sum(1 for r in results if r.status == HealthStatus.UNHEALTHY),
                    "critical": sum(1 for r in results if r.status == HealthStatus.CRITICAL)
                }
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Health report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save health report: {e}")
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring."""
        self.logger.info(f"Starting health monitoring with {interval_seconds}s interval")
        
        try:
            while True:
                results = await self.run_all_health_checks()
                
                # Save health report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = Path(f"health_report_{timestamp}.json")
                self.save_health_report(results, report_path)
                
                # Check for alerts
                self._check_alerts(results)
                
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Health monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Health monitoring failed: {e}")
            raise
    
    def _check_alerts(self, results: List[HealthCheckResult]):
        """Check if any alerts should be sent based on health check results."""
        current_time = datetime.now(timezone.utc)
        cooldown_minutes = self.config["alert_cooldown_minutes"]
        
        for result in results:
            if result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                alert_key = f"{result.component_name}_{result.status.value}"
                
                # Check if alert was recently sent
                if alert_key in self.alerts_sent:
                    time_since_alert = (current_time - self.alerts_sent[alert_key]).total_seconds() / 60
                    if time_since_alert < cooldown_minutes:
                        continue  # Skip this alert due to cooldown
                
                # Send alert (implement your preferred alerting mechanism)
                self._send_alert(result)
                self.alerts_sent[alert_key] = current_time
    
    def _send_alert(self, result: HealthCheckResult):
        """Send alert for critical health check results."""
        self.logger.warning(f"HEALTH ALERT: {result.component_name} is {result.status.value}", {
            "component": result.component_name,
            "status": result.status.value,
            "message": result.message,
            "metadata": result.metadata
        })
        
        # Here you could integrate with external alerting systems:
        # - Send email notifications
        # - Post to Slack/Discord
        # - Send to PagerDuty/OpsGenie
        # - Update monitoring dashboards


# Global health monitor instance
health_monitor = HealthMonitor()


# Example usage
if __name__ == "__main__":
    async def main():
        # Run health checks once
        results = await health_monitor.run_all_health_checks()
        
        print("Health Check Results:")
        print("=" * 50)
        for result in results:
            print(f"{result.component_name}: {result.status.value} - {result.message}")
        
        # Collect system metrics
        metrics = health_monitor.collect_system_metrics()
        print(f"\\nSystem Metrics:")
        print(f"CPU: {metrics.cpu_percent:.1f}%")
        print(f"Memory: {metrics.memory_percent:.1f}%")
        print(f"Disk: {metrics.disk_usage_percent:.1f}%")
        
        # Save report
        report_path = Path("health_report_example.json")
        health_monitor.save_health_report(results, report_path)
        print(f"\\nHealth report saved to {report_path}")
    
    asyncio.run(main())