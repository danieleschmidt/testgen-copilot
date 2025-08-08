"""
Multi-region deployment and data governance for TestGen-Copilot.

Provides regional data residency, cross-region replication, failover capabilities,
and region-specific compliance management for global deployment.
"""

import asyncio
import json
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from uuid import uuid4
import threading
import time
import hashlib

from .logging_config import get_core_logger
from .compliance import ComplianceFramework, DataClassification


class Region(str, Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"           # US East (N. Virginia)
    US_EAST_2 = "us-east-2"           # US East (Ohio)
    US_WEST_1 = "us-west-1"           # US West (N. California)
    US_WEST_2 = "us-west-2"           # US West (Oregon)
    EU_WEST_1 = "eu-west-1"           # Europe (Ireland)
    EU_WEST_2 = "eu-west-2"           # Europe (London)
    EU_CENTRAL_1 = "eu-central-1"     # Europe (Frankfurt)
    EU_NORTH_1 = "eu-north-1"         # Europe (Stockholm)
    AP_SOUTHEAST_1 = "ap-southeast-1" # Asia Pacific (Singapore)
    AP_SOUTHEAST_2 = "ap-southeast-2" # Asia Pacific (Sydney)
    AP_NORTHEAST_1 = "ap-northeast-1" # Asia Pacific (Tokyo)
    AP_NORTHEAST_2 = "ap-northeast-2" # Asia Pacific (Seoul)
    AP_SOUTH_1 = "ap-south-1"         # Asia Pacific (Mumbai)
    CA_CENTRAL_1 = "ca-central-1"     # Canada (Central)
    SA_EAST_1 = "sa-east-1"           # South America (São Paulo)
    AF_SOUTH_1 = "af-south-1"         # Africa (Cape Town)


class DataResidencyRequirement(str, Enum):
    """Data residency requirements."""
    STRICT = "STRICT"                 # Data must remain in specific region
    FLEXIBLE = "FLEXIBLE"             # Data can be replicated across regions
    EU_ONLY = "EU_ONLY"              # Data must remain within EU
    GDPR_COMPLIANT = "GDPR_COMPLIANT" # Data handling must comply with GDPR
    SOVEREIGN = "SOVEREIGN"           # Data must remain within national borders


class ReplicationStrategy(str, Enum):
    """Data replication strategies."""
    NONE = "NONE"                     # No replication
    ASYNC = "ASYNC"                   # Asynchronous replication
    SYNC = "SYNC"                     # Synchronous replication
    MULTI_MASTER = "MULTI_MASTER"     # Multi-master replication
    BACKUP_ONLY = "BACKUP_ONLY"       # Backup replication only


class HealthStatus(str, Enum):
    """Region health status."""
    HEALTHY = "HEALTHY"               # Region operating normally
    DEGRADED = "DEGRADED"             # Region experiencing issues
    UNAVAILABLE = "UNAVAILABLE"       # Region not accessible
    MAINTENANCE = "MAINTENANCE"       # Region under maintenance


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    name: str
    country_code: str
    jurisdiction: str
    primary: bool = False
    enabled: bool = True
    data_residency: DataResidencyRequirement = DataResidencyRequirement.FLEXIBLE
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    replication_targets: Set[Region] = field(default_factory=set)
    replication_strategy: ReplicationStrategy = ReplicationStrategy.ASYNC
    endpoint_url: str = ""
    api_key: str = ""
    encryption_key: str = ""
    max_latency_ms: int = 1000
    backup_region: Optional[Region] = None
    local_storage_path: Optional[str] = None
    timezone: str = "UTC"


@dataclass
class RegionHealth:
    """Health information for a region."""
    region: Region
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_rate: float
    throughput_ops_per_sec: float
    storage_utilization: float
    cpu_utilization: float
    memory_utilization: float
    issues: List[str] = field(default_factory=list)
    last_incident: Optional[datetime] = None


@dataclass
class DataLocation:
    """Information about where data is stored."""
    data_id: str
    primary_region: Region
    replicated_regions: Set[Region]
    data_classification: DataClassification
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    checksum: str = ""
    encryption_enabled: bool = True


class RegionHealthMonitor:
    """Monitors health of all regions."""
    
    def __init__(self, regions: Dict[Region, RegionConfig]):
        self.logger = get_core_logger()
        self.regions = regions
        self._health_status: Dict[Region, RegionHealth] = {}
        self._monitoring_enabled = True
        self._check_interval = 30  # seconds
        self._lock = threading.RLock()
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Initialize health status
        for region in regions.keys():
            self._health_status[region] = RegionHealth(
                region=region,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(timezone.utc),
                response_time_ms=0.0,
                error_rate=0.0,
                throughput_ops_per_sec=0.0,
                storage_utilization=0.0,
                cpu_utilization=0.0,
                memory_utilization=0.0
            )
    
    def start_monitoring(self) -> None:
        """Start region health monitoring."""
        if not self._monitoring_enabled:
            return
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            self._monitor_task = loop.create_task(self._monitor_regions())
            loop.run_until_complete(self._monitor_task)
        except Exception as e:
            self.logger.error("Region monitoring failed", {"error": str(e)})
        finally:
            loop.close()
    
    async def _monitor_regions(self) -> None:
        """Monitor all regions continuously."""
        while self._monitoring_enabled:
            try:
                # Check health of all regions
                for region, config in self.regions.items():
                    if config.enabled:
                        await self._check_region_health(region, config)
                
                # Sleep until next check
                await asyncio.sleep(self._check_interval)
                
            except Exception as e:
                self.logger.error("Error in region monitoring loop", {"error": str(e)})
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _check_region_health(self, region: Region, config: RegionConfig) -> None:
        """Check health of a specific region."""
        start_time = time.time()
        
        try:
            # Simulate health check (in real implementation, this would make API calls)
            if config.endpoint_url:
                # In real implementation: make HTTP health check to region endpoint
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Simulate health metrics
                import random
                status = HealthStatus.HEALTHY
                error_rate = random.uniform(0.0, 0.05)  # 0-5% error rate
                throughput = random.uniform(100.0, 1000.0)
                storage_util = random.uniform(0.1, 0.8)
                cpu_util = random.uniform(0.1, 0.7)
                memory_util = random.uniform(0.1, 0.6)
                
                # Determine status based on metrics
                if response_time > config.max_latency_ms * 2:
                    status = HealthStatus.DEGRADED
                if response_time > config.max_latency_ms * 5:
                    status = HealthStatus.UNAVAILABLE
                
                issues = []
                if error_rate > 0.1:
                    issues.append("High error rate")
                if cpu_util > 0.9:
                    issues.append("High CPU utilization")
                if memory_util > 0.9:
                    issues.append("High memory utilization")
                if storage_util > 0.9:
                    issues.append("High storage utilization")
            else:
                # Local region (no endpoint)
                response_time = 1.0
                status = HealthStatus.HEALTHY
                error_rate = 0.0
                throughput = 500.0
                storage_util = 0.3
                cpu_util = 0.2
                memory_util = 0.3
                issues = []
            
            # Update health status
            with self._lock:
                health = self._health_status[region]
                health.status = status
                health.last_check = datetime.now(timezone.utc)
                health.response_time_ms = response_time
                health.error_rate = error_rate
                health.throughput_ops_per_sec = throughput
                health.storage_utilization = storage_util
                health.cpu_utilization = cpu_util
                health.memory_utilization = memory_util
                health.issues = issues
                
                if status != HealthStatus.HEALTHY:
                    health.last_incident = datetime.now(timezone.utc)
            
            self.logger.debug("Region health checked", {
                "region": region.value,
                "status": status.value,
                "response_time_ms": response_time
            })
            
        except Exception as e:
            with self._lock:
                health = self._health_status[region]
                health.status = HealthStatus.UNAVAILABLE
                health.last_check = datetime.now(timezone.utc)
                health.issues = [f"Health check failed: {str(e)}"]
                health.last_incident = datetime.now(timezone.utc)
            
            self.logger.warning("Region health check failed", {
                "region": region.value,
                "error": str(e)
            })
    
    def get_region_health(self, region: Region) -> Optional[RegionHealth]:
        """Get health status for a specific region."""
        with self._lock:
            return self._health_status.get(region)
    
    def get_healthy_regions(self) -> List[Region]:
        """Get list of healthy regions."""
        with self._lock:
            return [
                region for region, health in self._health_status.items()
                if health.status == HealthStatus.HEALTHY
            ]
    
    def get_best_region_for_location(self, user_location: str) -> Optional[Region]:
        """Get the best region for a user location based on latency."""
        # Simple mapping - in real implementation would use geolocation
        location_mappings = {
            "US": [Region.US_EAST_1, Region.US_WEST_2],
            "CA": [Region.CA_CENTRAL_1, Region.US_EAST_1],
            "EU": [Region.EU_WEST_1, Region.EU_CENTRAL_1],
            "UK": [Region.EU_WEST_2, Region.EU_WEST_1],
            "DE": [Region.EU_CENTRAL_1, Region.EU_WEST_1],
            "JP": [Region.AP_NORTHEAST_1, Region.AP_SOUTHEAST_1],
            "AU": [Region.AP_SOUTHEAST_2, Region.AP_SOUTHEAST_1],
            "SG": [Region.AP_SOUTHEAST_1, Region.AP_SOUTHEAST_2],
        }
        
        preferred_regions = location_mappings.get(user_location, [Region.US_EAST_1])
        healthy_regions = set(self.get_healthy_regions())
        
        # Return first healthy preferred region
        for region in preferred_regions:
            if region in healthy_regions and self.regions.get(region, {}).enabled:
                return region
        
        # Fallback to any healthy region
        if healthy_regions:
            return list(healthy_regions)[0]
        
        return None
    
    def stop_monitoring(self) -> None:
        """Stop region health monitoring."""
        self._monitoring_enabled = False
        if self._monitor_task:
            self._monitor_task.cancel()


class DataReplicationManager:
    """Manages data replication across regions."""
    
    def __init__(self, regions: Dict[Region, RegionConfig]):
        self.logger = get_core_logger()
        self.regions = regions
        self._data_locations: Dict[str, DataLocation] = {}
        self._replication_queue: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def store_data(self, data_id: str, content: Any, data_classification: DataClassification,
                   preferred_region: Optional[Region] = None) -> DataLocation:
        """Store data with appropriate regional placement."""
        # Determine primary region based on data classification and compliance
        primary_region = self._select_primary_region(data_classification, preferred_region)
        
        # Determine replication regions
        replicated_regions = self._select_replication_regions(primary_region, data_classification)
        
        # Calculate data checksum
        content_str = json.dumps(content, default=str) if not isinstance(content, str) else content
        checksum = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        
        # Create data location record
        data_location = DataLocation(
            data_id=data_id,
            primary_region=primary_region,
            replicated_regions=replicated_regions,
            data_classification=data_classification,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            size_bytes=len(content_str.encode('utf-8')),
            checksum=checksum
        )
        
        # Store locally (in real implementation, would store to region-specific storage)
        with self._lock:
            self._data_locations[data_id] = data_location
            
            # Queue for replication if needed
            if replicated_regions:
                self._replication_queue.append({
                    "data_id": data_id,
                    "source_region": primary_region,
                    "target_regions": list(replicated_regions),
                    "content": content,
                    "queued_at": datetime.now(timezone.utc)
                })
        
        self.logger.info("Data stored with regional placement", {
            "data_id": data_id,
            "primary_region": primary_region.value,
            "replicated_regions": [r.value for r in replicated_regions],
            "data_classification": data_classification.value
        })
        
        return data_location
    
    def _select_primary_region(self, data_classification: DataClassification,
                              preferred_region: Optional[Region] = None) -> Region:
        """Select primary region based on data classification and requirements."""
        if preferred_region and preferred_region in self.regions:
            region_config = self.regions[preferred_region]
            
            # Check if region can handle this data classification
            if self._can_region_handle_data(region_config, data_classification):
                return preferred_region
        
        # Find suitable regions based on data classification
        suitable_regions = []
        for region, config in self.regions.items():
            if config.enabled and self._can_region_handle_data(config, data_classification):
                suitable_regions.append((region, config))
        
        if not suitable_regions:
            # Fallback to primary region
            primary_regions = [r for r, c in self.regions.items() if c.primary]
            return primary_regions[0] if primary_regions else Region.US_EAST_1
        
        # Prefer primary regions
        primary_suitable = [r for r, c in suitable_regions if c.primary]
        if primary_suitable:
            return primary_suitable[0]
        
        return suitable_regions[0][0]
    
    def _can_region_handle_data(self, config: RegionConfig, 
                               data_classification: DataClassification) -> bool:
        """Check if a region can handle specific data classification."""
        # Check data residency requirements
        if data_classification in {DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL}:
            # EU data must stay in EU for GDPR compliance
            if (config.data_residency == DataResidencyRequirement.EU_ONLY and
                not config.region.value.startswith("eu-")):
                return False
            
            # Strict residency means data cannot leave the region
            if config.data_residency == DataResidencyRequirement.STRICT:
                return True  # Can handle but won't replicate
        
        return True
    
    def _select_replication_regions(self, primary_region: Region,
                                   data_classification: DataClassification) -> Set[Region]:
        """Select regions for data replication."""
        primary_config = self.regions.get(primary_region)
        if not primary_config:
            return set()
        
        # Check if replication is allowed for this data classification
        if (data_classification in {DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL} and
            primary_config.data_residency == DataResidencyRequirement.STRICT):
            return set()  # No replication for strict residency
        
        # Get configured replication targets
        replication_targets = set()
        for target_region in primary_config.replication_targets:
            target_config = self.regions.get(target_region)
            if (target_config and target_config.enabled and
                self._can_region_handle_data(target_config, data_classification)):
                replication_targets.add(target_region)
        
        # For EU data, only replicate within EU
        if (data_classification in {DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL} and
            primary_config.data_residency in {DataResidencyRequirement.EU_ONLY, DataResidencyRequirement.GDPR_COMPLIANT}):
            replication_targets = {
                r for r in replication_targets 
                if r.value.startswith("eu-")
            }
        
        return replication_targets
    
    def get_data_location(self, data_id: str) -> Optional[DataLocation]:
        """Get data location information."""
        with self._lock:
            location = self._data_locations.get(data_id)
            if location:
                location.last_accessed = datetime.now(timezone.utc)
                location.access_count += 1
            return location
    
    def delete_data(self, data_id: str) -> bool:
        """Delete data from all regions."""
        with self._lock:
            location = self._data_locations.get(data_id)
            if not location:
                return False
            
            # In real implementation, would delete from all regions
            regions_to_delete = {location.primary_region} | location.replicated_regions
            
            self.logger.info("Data deleted from all regions", {
                "data_id": data_id,
                "regions": [r.value for r in regions_to_delete]
            })
            
            del self._data_locations[data_id]
            return True
    
    def process_replication_queue(self) -> int:
        """Process pending replication requests."""
        processed = 0
        
        with self._lock:
            pending_replications = list(self._replication_queue)
            self._replication_queue.clear()
        
        for replication in pending_replications:
            try:
                # In real implementation, would replicate data to target regions
                self.logger.debug("Data replicated", {
                    "data_id": replication["data_id"],
                    "source_region": replication["source_region"].value,
                    "target_regions": [r.value for r in replication["target_regions"]]
                })
                processed += 1
                
            except Exception as e:
                self.logger.error("Data replication failed", {
                    "data_id": replication["data_id"],
                    "error": str(e)
                })
                
                # Re-queue failed replication
                with self._lock:
                    self._replication_queue.append(replication)
        
        return processed
    
    def get_replication_status(self) -> Dict[str, Any]:
        """Get replication status summary."""
        with self._lock:
            return {
                "total_data_objects": len(self._data_locations),
                "pending_replications": len(self._replication_queue),
                "regions_in_use": len(set(
                    [loc.primary_region for loc in self._data_locations.values()] +
                    [r for loc in self._data_locations.values() for r in loc.replicated_regions]
                )),
                "data_by_classification": {
                    dc.value: sum(1 for loc in self._data_locations.values() 
                                 if loc.data_classification == dc)
                    for dc in DataClassification
                },
                "data_by_region": {
                    region.value: sum(
                        1 for loc in self._data_locations.values()
                        if loc.primary_region == region or region in loc.replicated_regions
                    )
                    for region in Region
                }
            }


class MultiRegionManager:
    """Main manager for multi-region deployment."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self._regions: Dict[Region, RegionConfig] = {}
        self._current_region: Optional[Region] = None
        self._health_monitor: Optional[RegionHealthMonitor] = None
        self._replication_manager: Optional[DataReplicationManager] = None
        self._failover_enabled = True
        self._lock = threading.RLock()
        
        # Initialize with default regions
        self._init_default_regions()
    
    def _init_default_regions(self) -> None:
        """Initialize default region configurations."""
        default_regions = {
            Region.US_EAST_1: RegionConfig(
                region=Region.US_EAST_1,
                name="US East (Virginia)",
                country_code="US",
                jurisdiction="United States",
                primary=True,
                data_residency=DataResidencyRequirement.FLEXIBLE,
                replication_targets={Region.US_WEST_2, Region.EU_WEST_1},
                backup_region=Region.US_WEST_2
            ),
            Region.EU_WEST_1: RegionConfig(
                region=Region.EU_WEST_1,
                name="EU West (Ireland)",
                country_code="IE",
                jurisdiction="European Union",
                data_residency=DataResidencyRequirement.GDPR_COMPLIANT,
                compliance_frameworks={ComplianceFramework.GDPR},
                replication_targets={Region.EU_CENTRAL_1},
                backup_region=Region.EU_CENTRAL_1
            ),
            Region.AP_SOUTHEAST_1: RegionConfig(
                region=Region.AP_SOUTHEAST_1,
                name="Asia Pacific (Singapore)",
                country_code="SG",
                jurisdiction="Singapore",
                data_residency=DataResidencyRequirement.FLEXIBLE,
                replication_targets={Region.AP_NORTHEAST_1},
                backup_region=Region.AP_NORTHEAST_1
            )
        }
        
        self._regions = default_regions
        self._current_region = Region.US_EAST_1  # Default region
    
    def configure_region(self, region_config: RegionConfig) -> None:
        """Configure a specific region."""
        with self._lock:
            self._regions[region_config.region] = region_config
            
            # Reinitialize health monitor and replication manager
            self._init_managers()
        
        self.logger.info("Region configured", {
            "region": region_config.region.value,
            "name": region_config.name,
            "enabled": region_config.enabled
        })
    
    def _init_managers(self) -> None:
        """Initialize health monitor and replication manager."""
        if self._health_monitor:
            self._health_monitor.stop_monitoring()
        
        self._health_monitor = RegionHealthMonitor(self._regions)
        self._replication_manager = DataReplicationManager(self._regions)
    
    def start_monitoring(self) -> None:
        """Start region monitoring."""
        if not self._health_monitor:
            self._init_managers()
        
        # Start health monitoring in background
        monitor_thread = threading.Thread(
            target=self._health_monitor.start_monitoring,
            daemon=True
        )
        monitor_thread.start()
        
        self.logger.info("Multi-region monitoring started")
    
    def set_current_region(self, region: Region) -> bool:
        """Set the current active region."""
        if region not in self._regions:
            self.logger.warning("Unknown region", {"region": region.value})
            return False
        
        region_config = self._regions[region]
        if not region_config.enabled:
            self.logger.warning("Region not enabled", {"region": region.value})
            return False
        
        with self._lock:
            self._current_region = region
        
        self.logger.info("Current region changed", {"region": region.value})
        return True
    
    def get_current_region(self) -> Optional[Region]:
        """Get the current active region."""
        return self._current_region
    
    def store_data_with_compliance(self, data_id: str, content: Any,
                                  data_classification: DataClassification,
                                  user_location: str = "US") -> Optional[DataLocation]:
        """Store data with regional compliance considerations."""
        if not self._replication_manager:
            self._init_managers()
        
        # Determine best region for user location
        if self._health_monitor:
            preferred_region = self._health_monitor.get_best_region_for_location(user_location)
        else:
            preferred_region = self._current_region
        
        # Store data with replication
        return self._replication_manager.store_data(
            data_id, content, data_classification, preferred_region
        )
    
    def retrieve_data(self, data_id: str, user_location: str = "US") -> Tuple[Optional[Any], Optional[Region]]:
        """Retrieve data from the optimal region."""
        if not self._replication_manager:
            return None, None
        
        # Get data location
        data_location = self._replication_manager.get_data_location(data_id)
        if not data_location:
            return None, None
        
        # Determine best region to serve from
        available_regions = {data_location.primary_region} | data_location.replicated_regions
        
        if self._health_monitor:
            healthy_regions = set(self._health_monitor.get_healthy_regions())
            available_healthy = available_regions & healthy_regions
            
            if available_healthy:
                # Get best region for user location among available healthy regions
                best_region = None
                for region in available_healthy:
                    if self._health_monitor.get_region_health(region).response_time_ms < 500:
                        best_region = region
                        break
                
                if not best_region:
                    best_region = list(available_healthy)[0]
                
                # In real implementation, would retrieve from best_region
                return f"data_content_from_{best_region.value}", best_region
        
        # Fallback to primary region
        return f"data_content_from_{data_location.primary_region.value}", data_location.primary_region
    
    def handle_region_failover(self, failed_region: Region) -> Optional[Region]:
        """Handle failover when a region becomes unavailable."""
        if not self._failover_enabled:
            return None
        
        region_config = self._regions.get(failed_region)
        if not region_config or not region_config.backup_region:
            return None
        
        backup_region = region_config.backup_region
        backup_config = self._regions.get(backup_region)
        
        if backup_config and backup_config.enabled:
            self.logger.warning("Failing over to backup region", {
                "failed_region": failed_region.value,
                "backup_region": backup_region.value
            })
            
            # In real implementation, would redirect traffic to backup region
            return backup_region
        
        return None
    
    def get_region_status(self) -> Dict[str, Any]:
        """Get status of all regions."""
        status = {
            "current_region": self._current_region.value if self._current_region else None,
            "total_regions": len(self._regions),
            "enabled_regions": sum(1 for c in self._regions.values() if c.enabled),
            "regions": []
        }
        
        for region, config in self._regions.items():
            region_info = {
                "region": region.value,
                "name": config.name,
                "country": config.country_code,
                "enabled": config.enabled,
                "primary": config.primary,
                "data_residency": config.data_residency.value,
                "compliance_frameworks": [f.value for f in config.compliance_frameworks]
            }
            
            # Add health information if available
            if self._health_monitor:
                health = self._health_monitor.get_region_health(region)
                if health:
                    region_info.update({
                        "status": health.status.value,
                        "response_time_ms": health.response_time_ms,
                        "last_check": health.last_check.isoformat(),
                        "issues": health.issues
                    })
            
            status["regions"].append(region_info)
        
        # Add replication status if available
        if self._replication_manager:
            status["replication"] = self._replication_manager.get_replication_status()
        
        return status
    
    def get_compliance_regions(self, framework: ComplianceFramework) -> List[Region]:
        """Get regions that support a specific compliance framework."""
        compliant_regions = []
        
        for region, config in self._regions.items():
            if framework in config.compliance_frameworks or not config.compliance_frameworks:
                # Include regions with no specific frameworks (assume general compliance)
                compliant_regions.append(region)
        
        return compliant_regions
    
    def validate_data_placement(self, data_classification: DataClassification,
                               target_region: Region) -> Dict[str, Any]:
        """Validate if data can be placed in a specific region."""
        config = self._regions.get(target_region)
        if not config:
            return {
                "valid": False,
                "reason": "Region not configured",
                "recommendations": ["Configure the target region"]
            }
        
        if not config.enabled:
            return {
                "valid": False,
                "reason": "Region is disabled",
                "recommendations": ["Enable the target region"]
            }
        
        # Check data residency requirements
        issues = []
        recommendations = []
        
        if data_classification in {DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL}:
            if config.data_residency == DataResidencyRequirement.EU_ONLY and not target_region.value.startswith("eu-"):
                issues.append("Personal data requires EU region for EU_ONLY residency requirement")
                recommendations.append("Use an EU region for this data")
        
        if data_classification == DataClassification.HEALTH and ComplianceFramework.HIPAA not in config.compliance_frameworks:
            issues.append("Health data requires HIPAA compliance")
            recommendations.append("Configure HIPAA compliance for this region")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "data_classification": data_classification.value,
            "target_region": target_region.value,
            "region_config": {
                "data_residency": config.data_residency.value,
                "compliance_frameworks": [f.value for f in config.compliance_frameworks]
            }
        }


# Global multi-region manager instance
_multi_region_manager: Optional[MultiRegionManager] = None
_manager_lock = threading.Lock()


def get_multi_region_manager() -> MultiRegionManager:
    """Get the global multi-region manager instance."""
    global _multi_region_manager
    
    if _multi_region_manager is None:
        with _manager_lock:
            if _multi_region_manager is None:
                _multi_region_manager = MultiRegionManager()
    
    return _multi_region_manager


def store_data_globally(data_id: str, content: Any, data_classification: DataClassification,
                       user_location: str = "US") -> Optional[DataLocation]:
    """Store data with global multi-region compliance."""
    return get_multi_region_manager().store_data_with_compliance(
        data_id, content, data_classification, user_location
    )


def get_optimal_region(user_location: str) -> Optional[Region]:
    """Get the optimal region for a user location."""
    manager = get_multi_region_manager()
    if manager._health_monitor:
        return manager._health_monitor.get_best_region_for_location(user_location)
    return manager.get_current_region()