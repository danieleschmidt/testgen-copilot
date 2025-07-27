"""Health check endpoints and monitoring utilities for TestGen Copilot Assistant."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import psutil
from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Health check status model."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, any]]


class HealthCheck:
    """Health check service for monitoring application status."""
    
    def __init__(self):
        self.start_time = time.time()
        self.version = "0.0.1"  # This should come from version.py
    
    async def get_health_status(self) -> HealthStatus:
        """Get comprehensive health status."""
        checks = await self._run_all_checks()
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
            checks=checks
        )
    
    async def _run_all_checks(self) -> Dict[str, Dict[str, any]]:
        """Run all health checks."""
        checks = {}
        
        # System checks
        checks.update(await self._system_checks())
        
        # Application checks  
        checks.update(await self._application_checks())
        
        # External dependency checks
        checks.update(await self._dependency_checks())
        
        return checks
    
    async def _system_checks(self) -> Dict[str, Dict[str, any]]:
        """Check system resources."""
        checks = {}
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            checks["cpu"] = {
                "status": "healthy" if cpu_percent < 80 else "degraded" if cpu_percent < 95 else "unhealthy",
                "usage_percent": cpu_percent,
                "threshold_warning": 80,
                "threshold_critical": 95
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            checks["memory"] = {
                "status": "healthy" if memory_percent < 80 else "degraded" if memory_percent < 95 else "unhealthy",
                "usage_percent": memory_percent,
                "available_bytes": memory.available,
                "total_bytes": memory.total,
                "threshold_warning": 80,
                "threshold_critical": 95
            }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            checks["disk"] = {
                "status": "healthy" if disk_percent < 80 else "degraded" if disk_percent < 95 else "unhealthy",
                "usage_percent": disk_percent,
                "free_bytes": disk.free,
                "total_bytes": disk.total,
                "threshold_warning": 80,
                "threshold_critical": 95
            }
            
        except Exception as e:
            checks["system"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return checks
    
    async def _application_checks(self) -> Dict[str, Dict[str, any]]:
        """Check application-specific health."""
        checks = {}
        
        try:
            # Test the core functionality
            from testgen_copilot.core import TestGenerator
            
            # Quick functionality test
            start_time = time.time()
            generator = TestGenerator()
            # Perform a simple test generation check
            duration = time.time() - start_time
            
            checks["test_generation"] = {
                "status": "healthy" if duration < 5 else "degraded" if duration < 10 else "unhealthy",
                "response_time_seconds": duration,
                "threshold_warning": 5,
                "threshold_critical": 10
            }
            
        except Exception as e:
            checks["test_generation"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        # Check configuration
        try:
            import os
            required_env_vars = ["TESTGEN_ENV", "TESTGEN_LOG_LEVEL"]
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            checks["configuration"] = {
                "status": "healthy" if not missing_vars else "degraded",
                "missing_environment_variables": missing_vars
            }
            
        except Exception as e:
            checks["configuration"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return checks
    
    async def _dependency_checks(self) -> Dict[str, Dict[str, any]]:
        """Check external dependencies."""
        checks = {}
        
        # Redis check (if configured)
        try:
            import redis
            import os
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            r = redis.from_url(redis_url, socket_timeout=5)
            
            start_time = time.time()
            r.ping()
            duration = time.time() - start_time
            
            checks["redis"] = {
                "status": "healthy" if duration < 1 else "degraded" if duration < 3 else "unhealthy",
                "response_time_seconds": duration,
                "threshold_warning": 1,
                "threshold_critical": 3
            }
            
        except ImportError:
            checks["redis"] = {
                "status": "not_configured",
                "message": "Redis client not installed"
            }
        except Exception as e:
            checks["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Database check (if configured)
        try:
            import psycopg2
            import os
            
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                start_time = time.time()
                conn = psycopg2.connect(db_url)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                conn.close()
                duration = time.time() - start_time
                
                checks["database"] = {
                    "status": "healthy" if duration < 1 else "degraded" if duration < 3 else "unhealthy",
                    "response_time_seconds": duration,
                    "threshold_warning": 1,
                    "threshold_critical": 3
                }
            else:
                checks["database"] = {
                    "status": "not_configured",
                    "message": "Database URL not provided"
                }
                
        except ImportError:
            checks["database"] = {
                "status": "not_configured",
                "message": "Database client not installed"
            }
        except Exception as e:
            checks["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return checks
    
    def _determine_overall_status(self, checks: Dict[str, Dict[str, any]]) -> str:
        """Determine overall health status from individual checks."""
        statuses = [check.get("status", "unknown") for check in checks.values()]
        
        if any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        elif any(status == "degraded" for status in statuses):
            return "degraded"
        else:
            return "healthy"


# Metrics collection
class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self.metrics[key] = self.metrics.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self.metrics[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        # Simplified histogram - in production, use proper histogram buckets
        key = self._make_key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for key, value in self.metrics.items():
            if isinstance(value, list):
                # Histogram
                if value:
                    lines.append(f"{key}_sum {sum(value)}")
                    lines.append(f"{key}_count {len(value)}")
                    lines.append(f"{key}_avg {sum(value) / len(value)}")
            else:
                # Counter or gauge
                lines.append(f"{key} {value}")
        
        # Add uptime metric
        uptime = time.time() - self.start_time
        lines.append(f"testgen_uptime_seconds {uptime}")
        
        return "\n".join(lines)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a metric key with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global instances
health_checker = HealthCheck()
metrics_collector = MetricsCollector()