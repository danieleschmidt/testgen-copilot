"""
Progressive Enhancement Module

Implements the 3-generation enhancement strategy:
- Generation 1: Make It Work (Simple)
- Generation 2: Make It Robust (Reliable) 
- Generation 3: Make It Scale (Optimized)

With adaptive intelligence and self-improving patterns.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from .execution_engine import SDLCPhase, ExecutionStatus


class EnhancementGeneration(Enum):
    """Progressive enhancement generations"""
    GENERATION_1_SIMPLE = "generation_1_simple"
    GENERATION_2_ROBUST = "generation_2_robust" 
    GENERATION_3_SCALE = "generation_3_scale"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    response_time: float = 0.0
    throughput: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    concurrent_users: int = 0
    requests_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScalingTrigger:
    """Auto-scaling trigger configuration"""
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    action: str      # "scale_up", "scale_down", "optimize"
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None


class AdaptiveCache:
    """Adaptive caching system with smart eviction"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access tracking"""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        
        # Check TTL
        if datetime.utcnow() - entry["timestamp"] > timedelta(seconds=self.ttl_seconds):
            await self._evict(key)
            return None
        
        # Update access tracking
        self.access_count[key] = self.access_count.get(key, 0) + 1
        self.last_access[key] = datetime.utcnow()
        
        return entry["value"]
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with intelligent eviction"""
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            await self._evict_lru()
        
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.utcnow()
        }
        self.access_count[key] = 1
        self.last_access[key] = datetime.utcnow()
    
    async def _evict(self, key: str) -> None:
        """Evict specific key"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_count:
            del self.access_count[key]
        if key in self.last_access:
            del self.last_access[key]
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.last_access:
            return
            
        # Find LRU key
        lru_key = min(self.last_access.keys(), key=lambda k: self.last_access[k])
        await self._evict(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "total_accesses": sum(self.access_count.values()),
            "unique_keys": len(self.access_count)
        }


class ResourcePool:
    """Dynamic resource pool with auto-scaling"""
    
    def __init__(self, resource_factory: Callable, min_size: int = 2, max_size: int = 20):
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        
        self.available_resources = []
        self.in_use_resources = set()
        self.total_created = 0
        self.creation_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize resource pool"""
        async with self.creation_lock:
            for _ in range(self.min_size):
                resource = await self.resource_factory()
                self.available_resources.append(resource)
                self.total_created += 1
    
    async def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire resource from pool"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Try to get available resource
            if self.available_resources:
                resource = self.available_resources.pop()
                self.in_use_resources.add(resource)
                return resource
            
            # Create new resource if under max limit
            if self.total_created < self.max_size:
                async with self.creation_lock:
                    if self.total_created < self.max_size:
                        resource = await self.resource_factory()
                        self.in_use_resources.add(resource)
                        self.total_created += 1
                        return resource
            
            # Wait and retry
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError("Failed to acquire resource within timeout")
    
    async def release(self, resource: Any) -> None:
        """Release resource back to pool"""
        if resource in self.in_use_resources:
            self.in_use_resources.remove(resource)
            self.available_resources.append(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "available": len(self.available_resources),
            "in_use": len(self.in_use_resources),
            "total_created": self.total_created,
            "utilization": len(self.in_use_resources) / max(self.total_created, 1)
        }


class PerformanceOptimizer:
    """Intelligent performance optimization system"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_strategies: Dict[str, Callable] = {
            "cache_optimization": self._optimize_caching,
            "concurrency_tuning": self._tune_concurrency,
            "resource_pooling": self._optimize_resource_pools,
            "load_balancing": self._balance_load
        }
        
        self.adaptive_cache = AdaptiveCache()
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Calculate cache hit rate
        cache_stats = self.adaptive_cache.get_stats()
        cache_hit_rate = cache_stats.get("hit_rate", 0.0)
        
        metrics = PerformanceMetrics(
            response_time=await self._measure_response_time(),
            throughput=await self._measure_throughput(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            cache_hit_rate=cache_hit_rate,
            error_rate=await self._measure_error_rate(),
            concurrent_users=await self._count_concurrent_users(),
            requests_per_second=await self._measure_rps()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run comprehensive performance optimization"""
        
        self.logger.info("🚀 Starting performance optimization...")
        
        current_metrics = await self.collect_metrics()
        optimization_results = {}
        
        # Run optimization strategies
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                result = await strategy_func(current_metrics)
                optimization_results[strategy_name] = result
                self.logger.info(f"✅ {strategy_name} completed: {result.get('improvement', 'no change')}")
                
            except Exception as e:
                self.logger.error(f"❌ {strategy_name} failed: {e}")
                optimization_results[strategy_name] = {"error": str(e)}
        
        return optimization_results
    
    async def _optimize_caching(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize caching strategy"""
        
        cache_stats = self.adaptive_cache.get_stats()
        
        # Adjust cache size based on hit rate and memory usage
        if metrics.cache_hit_rate < 0.5 and metrics.memory_usage < 70:
            # Increase cache size
            new_size = min(self.adaptive_cache.max_size * 2, 5000)
            self.adaptive_cache.max_size = new_size
            improvement = f"Increased cache size to {new_size}"
            
        elif metrics.memory_usage > 85 and cache_stats["size"] > 100:
            # Decrease cache size to free memory
            new_size = max(self.adaptive_cache.max_size // 2, 100)
            self.adaptive_cache.max_size = new_size
            improvement = f"Decreased cache size to {new_size}"
            
        else:
            improvement = "Cache size optimal"
        
        # Adjust TTL based on access patterns
        if metrics.cache_hit_rate > 0.8:
            # High hit rate, can increase TTL
            self.adaptive_cache.ttl_seconds = min(self.adaptive_cache.ttl_seconds * 1.2, 7200)
        elif metrics.cache_hit_rate < 0.3:
            # Low hit rate, decrease TTL for fresher data
            self.adaptive_cache.ttl_seconds = max(self.adaptive_cache.ttl_seconds * 0.8, 300)
        
        return {
            "improvement": improvement,
            "cache_stats": cache_stats,
            "new_ttl": self.adaptive_cache.ttl_seconds
        }
    
    async def _tune_concurrency(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Tune concurrency levels"""
        
        optimal_concurrency = await self._calculate_optimal_concurrency(metrics)
        
        # Adjust based on CPU and response time
        if metrics.cpu_usage > 80:
            # High CPU, reduce concurrency
            adjustment = "reduce_concurrency"
            recommended_workers = max(2, optimal_concurrency - 2)
        elif metrics.response_time > 5.0 and metrics.cpu_usage < 60:
            # Slow responses but CPU available, increase concurrency
            adjustment = "increase_concurrency" 
            recommended_workers = min(20, optimal_concurrency + 2)
        else:
            adjustment = "maintain_current"
            recommended_workers = optimal_concurrency
        
        return {
            "improvement": adjustment,
            "recommended_workers": recommended_workers,
            "current_cpu": metrics.cpu_usage,
            "current_response_time": metrics.response_time
        }
    
    async def _optimize_resource_pools(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize resource pool configurations"""
        
        optimizations = []
        
        for pool_name, pool in self.resource_pools.items():
            pool_stats = pool.get_stats()
            
            # Adjust pool sizes based on utilization
            if pool_stats["utilization"] > 0.8:
                # High utilization, increase max size
                new_max = min(pool.max_size * 2, 50)
                pool.max_size = new_max
                optimizations.append(f"Increased {pool_name} max size to {new_max}")
                
            elif pool_stats["utilization"] < 0.2 and pool.max_size > pool.min_size * 2:
                # Low utilization, decrease max size
                new_max = max(pool.max_size // 2, pool.min_size * 2)
                pool.max_size = new_max
                optimizations.append(f"Decreased {pool_name} max size to {new_max}")
        
        return {
            "improvement": optimizations or ["Resource pools optimal"],
            "pool_stats": {name: pool.get_stats() for name, pool in self.resource_pools.items()}
        }
    
    async def _balance_load(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Implement intelligent load balancing"""
        
        # Simple load balancing strategy based on metrics
        if metrics.response_time > 3.0:
            strategy = "round_robin_with_health_check"
        elif metrics.error_rate > 0.05:
            strategy = "least_connections"
        else:
            strategy = "weighted_round_robin"
        
        return {
            "improvement": f"Applied {strategy} load balancing",
            "strategy": strategy,
            "metrics_considered": {
                "response_time": metrics.response_time,
                "error_rate": metrics.error_rate
            }
        }
    
    async def _calculate_optimal_concurrency(self, metrics: PerformanceMetrics) -> int:
        """Calculate optimal concurrency level"""
        
        # Simple heuristic based on CPU cores and current metrics
        import multiprocessing
        
        cpu_cores = multiprocessing.cpu_count()
        base_concurrency = cpu_cores * 2  # Start with 2x CPU cores
        
        # Adjust based on metrics
        if metrics.cpu_usage > 70:
            # High CPU, reduce concurrency
            return max(2, int(base_concurrency * 0.7))
        elif metrics.cpu_usage < 40:
            # Low CPU, can increase concurrency
            return min(20, int(base_concurrency * 1.5))
        else:
            return base_concurrency
    
    async def _measure_response_time(self) -> float:
        """Measure average response time"""
        # Simulate response time measurement
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]
            return sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        return 0.5  # Default
    
    async def _measure_throughput(self) -> float:
        """Measure system throughput"""
        # Simulate throughput measurement
        return 100.0  # requests per second
    
    async def _measure_error_rate(self) -> float:
        """Measure error rate"""
        # Simulate error rate calculation
        return 0.02  # 2% error rate
    
    async def _count_concurrent_users(self) -> int:
        """Count concurrent users/connections"""
        # Simulate concurrent user counting
        return 25
    
    async def _measure_rps(self) -> float:
        """Measure requests per second"""
        # Simulate RPS measurement
        return 150.0


class AutoScalingManager:
    """Intelligent auto-scaling system"""
    
    def __init__(self):
        self.scaling_triggers: List[ScalingTrigger] = []
        self.current_scale = 1
        self.min_scale = 1
        self.max_scale = 10
        
        # Default scaling triggers
        self._setup_default_triggers()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_triggers(self) -> None:
        """Setup default scaling triggers"""
        
        self.scaling_triggers = [
            ScalingTrigger(
                metric_name="cpu_usage",
                threshold=75.0,
                comparison="gt",
                action="scale_up",
                cooldown_seconds=300
            ),
            ScalingTrigger(
                metric_name="cpu_usage", 
                threshold=25.0,
                comparison="lt",
                action="scale_down",
                cooldown_seconds=600
            ),
            ScalingTrigger(
                metric_name="response_time",
                threshold=3.0,
                comparison="gt", 
                action="scale_up",
                cooldown_seconds=180
            ),
            ScalingTrigger(
                metric_name="error_rate",
                threshold=0.1,
                comparison="gt",
                action="scale_up",
                cooldown_seconds=120
            )
        ]
    
    async def evaluate_scaling(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate if scaling action is needed"""
        
        scaling_actions = []
        now = datetime.utcnow()
        
        for trigger in self.scaling_triggers:
            # Check cooldown
            if (trigger.last_triggered and 
                (now - trigger.last_triggered).total_seconds() < trigger.cooldown_seconds):
                continue
            
            # Get metric value
            metric_value = getattr(metrics, trigger.metric_name, 0)
            
            # Evaluate trigger condition
            should_trigger = False
            
            if trigger.comparison == "gt" and metric_value > trigger.threshold:
                should_trigger = True
            elif trigger.comparison == "lt" and metric_value < trigger.threshold:
                should_trigger = True
            elif trigger.comparison == "eq" and abs(metric_value - trigger.threshold) < 0.01:
                should_trigger = True
            
            if should_trigger:
                action_result = await self._execute_scaling_action(trigger, metric_value)
                scaling_actions.append(action_result)
                trigger.last_triggered = now
        
        return {
            "actions_taken": scaling_actions,
            "current_scale": self.current_scale,
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "response_time": metrics.response_time,
                "error_rate": metrics.error_rate
            }
        }
    
    async def _execute_scaling_action(self, trigger: ScalingTrigger, metric_value: float) -> Dict[str, Any]:
        """Execute scaling action"""
        
        if trigger.action == "scale_up" and self.current_scale < self.max_scale:
            new_scale = min(self.current_scale + 1, self.max_scale)
            await self._scale_to(new_scale)
            
            return {
                "action": "scale_up",
                "trigger_metric": trigger.metric_name,
                "metric_value": metric_value,
                "threshold": trigger.threshold,
                "old_scale": self.current_scale,
                "new_scale": new_scale
            }
            
        elif trigger.action == "scale_down" and self.current_scale > self.min_scale:
            new_scale = max(self.current_scale - 1, self.min_scale)
            await self._scale_to(new_scale)
            
            return {
                "action": "scale_down", 
                "trigger_metric": trigger.metric_name,
                "metric_value": metric_value,
                "threshold": trigger.threshold,
                "old_scale": self.current_scale,
                "new_scale": new_scale
            }
            
        else:
            return {
                "action": "no_action",
                "reason": f"Scale limits reached or action not applicable",
                "current_scale": self.current_scale
            }
    
    async def _scale_to(self, target_scale: int) -> None:
        """Scale system to target level"""
        
        self.logger.info(f"🔄 Scaling from {self.current_scale} to {target_scale}")
        
        # Simulate scaling process
        await asyncio.sleep(1)  # Simulate scaling delay
        
        self.current_scale = target_scale
        
        self.logger.info(f"✅ Scaled to {self.current_scale} instances")


class ProgressiveEnhancer:
    """
    Main progressive enhancement orchestrator implementing the 3-generation strategy.
    
    Coordinates performance optimization, auto-scaling, and adaptive intelligence.
    """
    
    def __init__(self, project_path: Path, config: Optional[Dict[str, Any]] = None):
        self.project_path = project_path
        self.config = config or {}
        
        # Core components
        self.performance_optimizer = PerformanceOptimizer()
        self.auto_scaler = AutoScalingManager()
        
        # Enhancement tracking
        self.current_generation = EnhancementGeneration.GENERATION_1_SIMPLE
        self.enhancement_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize progressive enhancement system"""
        try:
            self.logger.info("🚀 Initializing Progressive Enhancement System...")
            
            # Initialize performance optimizer
            await self.performance_optimizer.adaptive_cache.set("init", True)
            
            self.logger.info("✅ Progressive Enhancement System initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize progressive enhancement: {e}")
            return False
    
    async def execute_generation_3_scaling(self) -> Dict[str, Any]:
        """Execute Generation 3: Make It Scale (Optimized)"""
        
        self.logger.info("🎯 Executing Generation 3: Make It Scale")
        
        self.current_generation = EnhancementGeneration.GENERATION_3_SCALE
        
        results = {
            "generation": "3_scale",
            "optimizations": [],
            "scaling_actions": [],
            "performance_improvements": {},
            "start_time": datetime.utcnow().isoformat()
        }
        
        try:
            # 1. Performance Optimization
            self.logger.info("🔧 Optimizing performance...")
            optimization_results = await self.performance_optimizer.optimize_performance()
            results["performance_improvements"] = optimization_results
            results["optimizations"].append("performance_optimization")
            
            # 2. Implement Adaptive Caching
            self.logger.info("💾 Implementing adaptive caching...")
            await self._implement_adaptive_caching()
            results["optimizations"].append("adaptive_caching")
            
            # 3. Setup Resource Pooling
            self.logger.info("🏊 Setting up resource pooling...")
            await self._setup_resource_pooling()
            results["optimizations"].append("resource_pooling")
            
            # 4. Enable Auto-scaling
            self.logger.info("📈 Enabling auto-scaling...")
            current_metrics = await self.performance_optimizer.collect_metrics()
            scaling_result = await self.auto_scaler.evaluate_scaling(current_metrics)
            results["scaling_actions"] = scaling_result["actions_taken"]
            results["optimizations"].append("auto_scaling")
            
            # 5. Load Balancing Implementation
            self.logger.info("⚖️ Implementing load balancing...")
            await self._implement_load_balancing()
            results["optimizations"].append("load_balancing")
            
            # 6. Concurrent Processing Enhancement
            self.logger.info("🔄 Enhancing concurrent processing...")
            await self._enhance_concurrent_processing()
            results["optimizations"].append("concurrent_processing")
            
            results["status"] = "completed"
            results["end_time"] = datetime.utcnow().isoformat()
            
            self.enhancement_history.append(results)
            
            self.logger.info("🎉 Generation 3 scaling completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Generation 3 scaling failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results
    
    async def _implement_adaptive_caching(self) -> None:
        """Implement adaptive caching system"""
        
        # Configure cache with intelligent defaults
        cache_config = self.config.get("cache", {})
        
        max_size = cache_config.get("max_size", 2000)
        ttl_seconds = cache_config.get("ttl_seconds", 1800)
        
        self.performance_optimizer.adaptive_cache = AdaptiveCache(max_size, ttl_seconds)
        
        # Pre-warm cache with common data
        await self._prewarm_cache()
        
        self.logger.info(f"✅ Adaptive caching configured (size: {max_size}, ttl: {ttl_seconds}s)")
    
    async def _setup_resource_pooling(self) -> None:
        """Setup dynamic resource pools"""
        
        # Database connection pool
        async def db_connection_factory():
            # Simulate database connection creation
            return {"connection_id": f"db_conn_{time.time()}", "created_at": datetime.utcnow()}
        
        db_pool = ResourcePool(db_connection_factory, min_size=3, max_size=15)
        await db_pool.initialize()
        self.performance_optimizer.resource_pools["database"] = db_pool
        
        # HTTP client pool  
        async def http_client_factory():
            # Simulate HTTP client creation
            return {"client_id": f"http_client_{time.time()}", "created_at": datetime.utcnow()}
        
        http_pool = ResourcePool(http_client_factory, min_size=2, max_size=10)
        await http_pool.initialize()
        self.performance_optimizer.resource_pools["http_client"] = http_pool
        
        self.logger.info("✅ Resource pools configured and initialized")
    
    async def _implement_load_balancing(self) -> None:
        """Implement intelligent load balancing"""
        
        load_balancing_config = {
            "strategy": "adaptive_round_robin",
            "health_check_interval": 30,
            "failover_enabled": True,
            "circuit_breaker_threshold": 5
        }
        
        # Store configuration for later use
        await self.performance_optimizer.adaptive_cache.set(
            "load_balancing_config", 
            load_balancing_config
        )
        
        self.logger.info("✅ Load balancing configured with adaptive round-robin")
    
    async def _enhance_concurrent_processing(self) -> None:
        """Enhance concurrent processing capabilities"""
        
        # Configure optimal concurrency levels
        current_metrics = await self.performance_optimizer.collect_metrics()
        optimal_concurrency = await self.performance_optimizer._calculate_optimal_concurrency(current_metrics)
        
        concurrency_config = {
            "max_workers": optimal_concurrency,
            "queue_size": optimal_concurrency * 10,
            "timeout": 30.0,
            "retry_attempts": 3
        }
        
        # Store configuration
        await self.performance_optimizer.adaptive_cache.set(
            "concurrency_config",
            concurrency_config
        )
        
        self.logger.info(f"✅ Concurrent processing enhanced (max_workers: {optimal_concurrency})")
    
    async def _prewarm_cache(self) -> None:
        """Pre-warm cache with frequently accessed data"""
        
        # Simulate pre-warming with common data
        common_data = {
            "config": {"version": "4.0.0", "environment": "production"},
            "metadata": {"project_type": "cli_tool", "language": "python"},
            "health_check": {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        }
        
        for key, value in common_data.items():
            await self.performance_optimizer.adaptive_cache.set(key, value)
        
        self.logger.debug("Cache pre-warmed with common data")
    
    async def monitor_and_adapt(self) -> Dict[str, Any]:
        """Continuous monitoring and adaptive optimization"""
        
        self.logger.info("👁️ Starting continuous monitoring...")
        
        monitoring_results = {
            "monitoring_start": datetime.utcnow().isoformat(),
            "metrics_collected": 0,
            "optimizations_applied": 0,
            "scaling_actions": 0
        }
        
        # Collect metrics and optimize periodically
        for cycle in range(5):  # Monitor for 5 cycles
            try:
                # Collect current metrics
                metrics = await self.performance_optimizer.collect_metrics()
                monitoring_results["metrics_collected"] += 1
                
                # Apply optimizations if needed
                if metrics.response_time > 2.0 or metrics.cpu_usage > 70:
                    await self.performance_optimizer.optimize_performance()
                    monitoring_results["optimizations_applied"] += 1
                
                # Evaluate scaling
                scaling_result = await self.auto_scaler.evaluate_scaling(metrics)
                if scaling_result["actions_taken"]:
                    monitoring_results["scaling_actions"] += len(scaling_result["actions_taken"])
                
                # Wait before next cycle
                await asyncio.sleep(2)  # 2-second intervals for demo
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle {cycle} failed: {e}")
        
        monitoring_results["monitoring_end"] = datetime.utcnow().isoformat()
        self.logger.info("✅ Monitoring cycle completed")
        
        return monitoring_results
    
    def get_enhancement_status(self) -> Dict[str, Any]:
        """Get current enhancement status"""
        
        cache_stats = self.performance_optimizer.adaptive_cache.get_stats()
        pool_stats = {
            name: pool.get_stats() 
            for name, pool in self.performance_optimizer.resource_pools.items()
        }
        
        return {
            "current_generation": self.current_generation.value,
            "enhancements_applied": len(self.enhancement_history),
            "cache_stats": cache_stats,
            "resource_pool_stats": pool_stats,
            "auto_scaler_scale": self.auto_scaler.current_scale,
            "metrics_history_size": len(self.performance_optimizer.metrics_history)
        }