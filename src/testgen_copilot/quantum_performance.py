"""High-performance quantum computing optimizations and scalability enhancements."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing

# import psutil  # Mock for compatibility
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

# from .quantum_planner import QuantumTask, ResourceQuantum, TaskPriority, TaskState
# from .quantum_monitoring import QuantumMetric

# Mock classes for standalone testing
class QuantumTask:
    def __init__(self, id, name, description, priority, resources_required=None, estimated_duration=None):
        self.id = id
        self.name = name
        self.description = description
        self.priority = priority
        self.resources_required = resources_required or {}
        self.estimated_duration = estimated_duration
        self.entangled_tasks = set()

    def calculate_urgency_score(self):
        return 0.5

class TaskPriority:
    GROUND_STATE = type('Priority', (), {'value': 0})()
    EXCITED_1 = type('Priority', (), {'value': 1})()
    EXCITED_2 = type('Priority', (), {'value': 2})()


@dataclass
class PerformanceConfig:
    """Configuration for quantum performance optimizations."""

    # Parallel processing
    max_workers: int = field(default_factory=lambda: min(multiprocessing.cpu_count() * 2, 16))
    worker_pool_type: str = "thread"  # "thread" or "process"
    task_batch_size: int = 100

    # Memory management
    memory_limit_mb: int = 4096
    cache_size_mb: int = 512
    enable_memory_monitoring: bool = True

    # Performance tuning
    enable_jit_compilation: bool = True
    enable_vectorization: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Quantum-specific optimizations
    quantum_coherence_optimization: bool = True
    entanglement_batch_processing: bool = True
    superposition_parallelization: bool = True

    # Scaling parameters
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 0.8  # CPU utilization
    scale_down_threshold: float = 0.3
    min_instances: int = 1
    max_instances: int = 10


class QuantumMemoryPool:
    """High-performance memory pool for quantum objects with lifecycle management."""

    def __init__(self, max_size_mb: int = 512):
        """Initialize quantum memory pool."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.pools: Dict[str, deque] = defaultdict(deque)
        self.usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"allocated": 0, "reused": 0})
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Weak references for automatic cleanup
        self.active_objects = weakref.WeakSet()

    def get_object(self, object_type: str, factory: Callable[[], Any]) -> Any:
        """Get object from pool or create new one."""

        with self.lock:
            if self.pools[object_type]:
                obj = self.pools[object_type].popleft()
                self.usage_stats[object_type]["reused"] += 1
                self.logger.debug(f"Reused {object_type} from pool")
                return obj

            # Create new object
            obj = factory()
            self.usage_stats[object_type]["allocated"] += 1
            # Only add to weak set if object supports weak references
            try:
                self.active_objects.add(obj)
            except TypeError:
                pass  # Object doesn't support weak references
            self.logger.debug(f"Created new {object_type}")
            return obj

    def return_object(self, object_type: str, obj: Any):
        """Return object to pool for reuse."""

        with self.lock:
            if len(self.pools[object_type]) < 100:  # Pool size limit
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()

                self.pools[object_type].append(obj)
                self.logger.debug(f"Returned {object_type} to pool")
            else:
                # Pool is full, object will be garbage collected
                self.logger.debug(f"Pool full, discarding {object_type}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""

        with self.lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())

            return {
                "current_size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": (self.current_size / self.max_size_bytes) * 100,
                "total_pooled_objects": total_pooled,
                "pools": {k: len(v) for k, v in self.pools.items()},
                "usage_stats": dict(self.usage_stats),
                "active_objects": len(self.active_objects)
            }

    def clear_pool(self, object_type: Optional[str] = None):
        """Clear specific pool or all pools."""

        with self.lock:
            if object_type:
                self.pools[object_type].clear()
                self.logger.info(f"Cleared {object_type} pool")
            else:
                self.pools.clear()
                self.usage_stats.clear()
                self.current_size = 0
                self.logger.info("Cleared all pools")


class QuantumTaskBatcher:
    """Batch processor for quantum tasks with intelligent grouping."""

    def __init__(self, batch_size: int = 100, batch_timeout: float = 5.0):
        """Initialize quantum task batcher."""
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_tasks: List[QuantumTask] = []
        self.batch_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # Batch statistics
        self.batches_processed = 0
        self.tasks_processed = 0
        self.avg_batch_size = 0.0

    async def add_task(self, task: QuantumTask) -> bool:
        """Add task to batch. Returns True if batch is ready for processing."""

        with self.batch_lock:
            self.pending_tasks.append(task)

            if len(self.pending_tasks) >= self.batch_size:
                return True

            return False

    async def get_batch(self, min_size: int = 1) -> List[QuantumTask]:
        """Get batch of tasks for processing."""

        with self.batch_lock:
            if len(self.pending_tasks) < min_size:
                return []

            # Intelligent batching - group by priority and entanglement
            batch = self._create_optimal_batch()

            # Update statistics
            self.batches_processed += 1
            self.tasks_processed += len(batch)
            self.avg_batch_size = self.tasks_processed / self.batches_processed

            self.logger.debug(f"Created batch of {len(batch)} tasks")
            return batch

    def _create_optimal_batch(self) -> List[QuantumTask]:
        """Create optimally grouped batch based on quantum properties."""

        if not self.pending_tasks:
            return []

        # Sort by priority and entanglement potential
        self.pending_tasks.sort(key=lambda t: (
            t.priority.value,
            -len(t.entangled_tasks),  # More entangled tasks first
            t.calculate_urgency_score()
        ))

        # Take up to batch_size tasks
        batch = self.pending_tasks[:self.batch_size]
        self.pending_tasks = self.pending_tasks[self.batch_size:]

        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""

        with self.batch_lock:
            return {
                "batches_processed": self.batches_processed,
                "tasks_processed": self.tasks_processed,
                "avg_batch_size": self.avg_batch_size,
                "pending_tasks": len(self.pending_tasks),
                "batch_size_limit": self.batch_size
            }


class QuantumPerformanceProfiler:
    """Advanced profiler for quantum operations with real-time metrics."""

    def __init__(self):
        """Initialize quantum performance profiler."""
        self.operation_metrics: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.cpu_usage_history: deque = deque(maxlen=100)
        self.profiling_active = False
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def profile_operation(self, operation_name: str):
        """Context manager for profiling quantum operations."""

        start_time = time.perf_counter()
        start_memory = 0  # Mock memory info
        start_cpu = 0  # Mock CPU info

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = 0  # Mock memory info
            end_cpu = 0  # Mock CPU info

            # Record metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_delta = end_cpu - start_cpu

            with self.lock:
                self.operation_metrics[operation_name].append(execution_time)

                # Keep only recent measurements
                if len(self.operation_metrics[operation_name]) > 1000:
                    self.operation_metrics[operation_name] = self.operation_metrics[operation_name][-1000:]

                # Record system metrics
                self.memory_snapshots.append({
                    "timestamp": time.time(),
                    "operation": operation_name,
                    "memory_mb": end_memory / (1024 * 1024),
                    "memory_delta_mb": memory_delta / (1024 * 1024),
                    "execution_time": execution_time
                })

                self.cpu_usage_history.append(end_cpu)

            self.logger.debug(f"{operation_name}: {execution_time:.4f}s, {memory_delta/1024/1024:.2f}MB delta")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        with self.lock:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation_statistics": {},
                "system_performance": {},
                "recommendations": []
            }

            # Operation statistics
            for op_name, timings in self.operation_metrics.items():
                if timings:
                    report["operation_statistics"][op_name] = {
                        "count": len(timings),
                        "avg_time": sum(timings) / len(timings),
                        "min_time": min(timings),
                        "max_time": max(timings),
                        "total_time": sum(timings),
                        "p95_time": self._calculate_percentile(timings, 0.95),
                        "p99_time": self._calculate_percentile(timings, 0.99)
                    }

            # System performance
            if self.memory_snapshots:
                recent_memory = [s["memory_mb"] for s in self.memory_snapshots[-50:]]
                report["system_performance"]["memory"] = {
                    "current_mb": recent_memory[-1] if recent_memory else 0,
                    "avg_mb": sum(recent_memory) / len(recent_memory),
                    "peak_mb": max(recent_memory),
                    "snapshots_count": len(self.memory_snapshots)
                }

            if self.cpu_usage_history:
                cpu_list = list(self.cpu_usage_history)
                report["system_performance"]["cpu"] = {
                    "current_percent": cpu_list[-1] if cpu_list else 0,
                    "avg_percent": sum(cpu_list) / len(cpu_list),
                    "peak_percent": max(cpu_list)
                }

            # Generate recommendations
            report["recommendations"] = self._generate_performance_recommendations(report)

            return report

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""

        recommendations = []

        # CPU recommendations
        cpu_data = report.get("system_performance", {}).get("cpu", {})
        avg_cpu = cpu_data.get("avg_percent", 0)

        if avg_cpu > 80:
            recommendations.append("🔥 HIGH CPU USAGE: Consider increasing worker pool size or optimizing hot code paths")
        elif avg_cpu < 20:
            recommendations.append("📉 LOW CPU USAGE: Consider reducing worker pool size to save resources")

        # Memory recommendations
        memory_data = report.get("system_performance", {}).get("memory", {})
        current_memory = memory_data.get("current_mb", 0)

        if current_memory > 2048:  # 2GB
            recommendations.append("💾 HIGH MEMORY USAGE: Consider implementing memory pooling and object reuse")

        # Operation-specific recommendations
        op_stats = report.get("operation_statistics", {})

        for op_name, stats in op_stats.items():
            avg_time = stats.get("avg_time", 0)
            p99_time = stats.get("p99_time", 0)

            if avg_time > 1.0:  # 1 second
                recommendations.append(f"⏱️ SLOW OPERATION: {op_name} averaging {avg_time:.2f}s - consider optimization")

            if p99_time > avg_time * 3:  # High variance
                recommendations.append(f"📊 HIGH VARIANCE: {op_name} has inconsistent performance - investigate outliers")

        if not recommendations:
            recommendations.append("✅ Performance looks optimal - no immediate recommendations")

        return recommendations


class QuantumLoadBalancer:
    """Intelligent load balancer for quantum task distribution."""

    def __init__(self, worker_configs: List[Dict[str, Any]]):
        """Initialize quantum load balancer."""
        self.workers = []
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.load_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # Initialize worker configurations
        for i, config in enumerate(worker_configs):
            worker_id = f"quantum_worker_{i}"
            self.workers.append({
                "id": worker_id,
                "capacity": config.get("capacity", 1.0),
                "current_load": 0.0,
                "quantum_efficiency": config.get("quantum_efficiency", 1.0),
                "specialization": config.get("specialization", "general"),
                "health_score": 1.0
            })

            self.worker_stats[worker_id] = {
                "tasks_completed": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "error_count": 0,
                "last_activity": time.time()
            }

    def select_worker(self, task: QuantumTask) -> Optional[str]:
        """Select optimal worker for quantum task using intelligent load balancing."""

        with self.lock:
            if not self.workers:
                return None

            # Calculate scores for each worker
            worker_scores = []

            for worker in self.workers:
                if worker["current_load"] >= worker["capacity"]:
                    continue  # Worker at capacity

                score = self._calculate_worker_score(worker, task)
                worker_scores.append((worker["id"], score))

            if not worker_scores:
                # All workers at capacity - find least loaded
                least_loaded = min(self.workers, key=lambda w: w["current_load"])
                return least_loaded["id"]

            # Select worker with highest score
            best_worker_id, best_score = max(worker_scores, key=lambda x: x[1])

            # Update worker load
            for worker in self.workers:
                if worker["id"] == best_worker_id:
                    task_load = self._estimate_task_load(task)
                    worker["current_load"] += task_load
                    break

            self.logger.debug(f"Selected worker {best_worker_id} with score {best_score:.3f}")
            return best_worker_id

    def _calculate_worker_score(self, worker: Dict[str, Any], task: QuantumTask) -> float:
        """Calculate suitability score for worker-task combination."""

        # Base score from available capacity
        available_capacity = worker["capacity"] - worker["current_load"]
        capacity_score = available_capacity / worker["capacity"]

        # Quantum efficiency bonus
        efficiency_score = worker["quantum_efficiency"] / 2.0  # Max 0.5 bonus

        # Specialization bonus
        specialization_score = 0.0
        if worker["specialization"] == "general":
            specialization_score = 0.1
        elif worker["specialization"] == "high_priority" and task.priority in [TaskPriority.GROUND_STATE, TaskPriority.EXCITED_1]:
            specialization_score = 0.3
        elif worker["specialization"] == "entanglement" and len(task.entangled_tasks) > 0:
            specialization_score = 0.3

        # Health score
        health_score = worker["health_score"] * 0.2

        # Performance history
        worker_id = worker["id"]
        stats = self.worker_stats[worker_id]
        performance_score = 0.0

        if stats["tasks_completed"] > 0:
            # Lower avg execution time = better performance
            normalized_time = min(stats["avg_execution_time"] / 10.0, 1.0)  # Normalize to 10 seconds
            performance_score = (1.0 - normalized_time) * 0.2

        total_score = capacity_score + efficiency_score + specialization_score + health_score + performance_score
        return total_score

    def _estimate_task_load(self, task: QuantumTask) -> float:
        """Estimate load impact of task on worker."""

        base_load = 0.1  # Minimum load per task

        # Duration-based load
        duration_hours = 1.0  # Default 1 hour if None
        if task.estimated_duration:
            duration_hours = task.estimated_duration.total_seconds() / 3600
        duration_load = min(duration_hours / 8.0, 1.0)  # Normalize to 8 hours

        # Resource-based load
        resource_load = sum(task.resources_required.values()) / 10.0  # Normalize

        # Priority-based load (higher priority = more resources)
        priority_load = (4 - task.priority.value) * 0.1

        # Entanglement overhead
        entanglement_load = len(task.entangled_tasks) * 0.05

        total_load = base_load + duration_load + resource_load + priority_load + entanglement_load
        return min(total_load, 1.0)  # Cap at 1.0

    def complete_task(self, worker_id: str, execution_time: float, success: bool):
        """Update worker statistics after task completion."""

        with self.lock:
            # Update worker load
            for worker in self.workers:
                if worker["id"] == worker_id:
                    # Reduce load (task completed)
                    worker["current_load"] = max(0.0, worker["current_load"] - 0.1)

                    # Update health score
                    if success:
                        worker["health_score"] = min(1.0, worker["health_score"] + 0.01)
                    else:
                        worker["health_score"] = max(0.1, worker["health_score"] - 0.05)
                    break

            # Update statistics
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats["last_activity"] = time.time()

                if success:
                    stats["tasks_completed"] += 1
                    stats["total_execution_time"] += execution_time
                    stats["avg_execution_time"] = stats["total_execution_time"] / stats["tasks_completed"]
                else:
                    stats["error_count"] += 1

    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across workers."""

        with self.lock:
            distribution = {
                "workers": [],
                "total_capacity": 0.0,
                "total_load": 0.0,
                "utilization_percent": 0.0,
                "load_balance_score": 0.0
            }

            loads = []
            for worker in self.workers:
                worker_data = {
                    "id": worker["id"],
                    "capacity": worker["capacity"],
                    "current_load": worker["current_load"],
                    "utilization": (worker["current_load"] / worker["capacity"]) * 100,
                    "health_score": worker["health_score"],
                    "quantum_efficiency": worker["quantum_efficiency"],
                    "specialization": worker["specialization"]
                }
                distribution["workers"].append(worker_data)

                distribution["total_capacity"] += worker["capacity"]
                distribution["total_load"] += worker["current_load"]
                loads.append(worker["current_load"] / worker["capacity"])

            if distribution["total_capacity"] > 0:
                distribution["utilization_percent"] = (distribution["total_load"] / distribution["total_capacity"]) * 100

            # Calculate load balance score (lower variance = better balance)
            if loads:
                mean_load = sum(loads) / len(loads)
                variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
                distribution["load_balance_score"] = max(0.0, 1.0 - variance)  # Higher is better

            return distribution


class QuantumAutoScaler:
    """Automatic scaling system for quantum task processing."""

    def __init__(self, config: PerformanceConfig):
        """Initialize quantum auto-scaler."""
        self.config = config
        self.current_instances = config.min_instances
        self.scaling_history: List[Dict[str, Any]] = []
        self.metrics_window: deque = deque(maxlen=50)
        self.last_scale_time = 0.0
        self.cooldown_period = 300.0  # 5 minutes
        self.logger = logging.getLogger(__name__)

    def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Determine if system should scale up."""

        if not self.config.auto_scaling_enabled:
            return False

        if self.current_instances >= self.config.max_instances:
            return False

        if time.time() - self.last_scale_time < self.cooldown_period:
            return False

        # Check CPU utilization
        cpu_usage = metrics.get("cpu_percent", 0)
        if cpu_usage > self.config.scale_up_threshold * 100:
            return True

        # Check queue length
        queue_length = metrics.get("queue_length", 0)
        if queue_length > self.current_instances * 10:  # 10 tasks per instance
            return True

        # Check response time
        avg_response_time = metrics.get("avg_response_time", 0)
        if avg_response_time > 2.0:  # 2 seconds
            return True

        return False

    def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Determine if system should scale down."""

        if not self.config.auto_scaling_enabled:
            return False

        if self.current_instances <= self.config.min_instances:
            return False

        if time.time() - self.last_scale_time < self.cooldown_period:
            return False

        # Check CPU utilization
        cpu_usage = metrics.get("cpu_percent", 0)
        if cpu_usage < self.config.scale_down_threshold * 100:
            # Also check that queue is not growing
            queue_length = metrics.get("queue_length", 0)
            if queue_length < self.current_instances * 2:
                return True

        return False

    def scale_up(self, target_instances: Optional[int] = None) -> Dict[str, Any]:
        """Scale up the system."""

        if target_instances is None:
            target_instances = min(self.current_instances + 1, self.config.max_instances)

        old_instances = self.current_instances
        self.current_instances = target_instances
        self.last_scale_time = time.time()

        scaling_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "scale_up",
            "from_instances": old_instances,
            "to_instances": target_instances,
            "reason": "High load detected"
        }

        self.scaling_history.append(scaling_event)
        self.logger.info(f"Scaled up from {old_instances} to {target_instances} instances")

        return scaling_event

    def scale_down(self, target_instances: Optional[int] = None) -> Dict[str, Any]:
        """Scale down the system."""

        if target_instances is None:
            target_instances = max(self.current_instances - 1, self.config.min_instances)

        old_instances = self.current_instances
        self.current_instances = target_instances
        self.last_scale_time = time.time()

        scaling_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "scale_down",
            "from_instances": old_instances,
            "to_instances": target_instances,
            "reason": "Low load detected"
        }

        self.scaling_history.append(scaling_event)
        self.logger.info(f"Scaled down from {old_instances} to {target_instances} instances")

        return scaling_event

    def get_scaling_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get scaling recommendations based on current metrics."""

        recommendations = []

        if self.should_scale_up(metrics):
            recommendations.append("📈 SCALE UP: High load detected - consider adding more instances")
        elif self.should_scale_down(metrics):
            recommendations.append("📉 SCALE DOWN: Low utilization - consider reducing instances to save costs")

        # Resource-specific recommendations
        cpu_usage = metrics.get("cpu_percent", 0)
        if cpu_usage > 90:
            recommendations.append("🔥 CRITICAL CPU: Immediate scaling recommended")

        memory_usage = metrics.get("memory_percent", 0)
        if memory_usage > 85:
            recommendations.append("💾 HIGH MEMORY: Consider memory optimization or scaling")

        queue_length = metrics.get("queue_length", 0)
        if queue_length > self.current_instances * 20:
            recommendations.append("📋 QUEUE BACKLOG: Scale up to handle task backlog")

        return recommendations

    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        return self.scaling_history[-limit:] if self.scaling_history else []


# Factory functions and integrations
def create_performance_optimized_planner(config: Optional[PerformanceConfig] = None):
    """Create quantum planner with performance optimizations enabled."""

    # Mock planner for testing
    class MockPlanner:
        def __init__(self):
            self.tasks = {}

        def add_task(self, task_id=None, **kwargs):
            # Map task_id to id for compatibility
            if task_id:
                kwargs['id'] = task_id
            task = QuantumTask(**kwargs)
            self.tasks[task.id] = task
            return task

    config = config or PerformanceConfig()
    planner = MockPlanner()

    # Add performance enhancements
    planner.memory_pool = QuantumMemoryPool(config.cache_size_mb)
    planner.task_batcher = QuantumTaskBatcher(config.task_batch_size)
    planner.profiler = QuantumPerformanceProfiler()
    planner.load_balancer = QuantumLoadBalancer([
        {"capacity": 1.0, "quantum_efficiency": 2.0, "specialization": "general"},
        {"capacity": 1.0, "quantum_efficiency": 1.8, "specialization": "high_priority"},
        {"capacity": 1.0, "quantum_efficiency": 1.5, "specialization": "entanglement"}
    ])
    planner.auto_scaler = QuantumAutoScaler(config)

    return planner


async def benchmark_quantum_performance():
    """Comprehensive performance benchmark for quantum systems."""

    print("🚀 Starting Quantum Performance Benchmark...")

    config = PerformanceConfig(
        max_workers=8,
        task_batch_size=50,
        memory_limit_mb=2048
    )

    planner = create_performance_optimized_planner(config)

    # Generate test workload
    tasks = []
    for i in range(500):  # Large task set
        task = planner.add_task(
            task_id=f"benchmark_task_{i}",
            name=f"Benchmark Task {i}",
            description=f"Performance test task {i}",
            priority=TaskPriority.EXCITED_2,
            estimated_duration=timedelta(minutes=random.randint(5, 60)),
            resources_required={
                "cpu": random.uniform(0.5, 3.0),
                "memory": random.uniform(1.0, 4.0),
                "io": random.uniform(0.1, 1.0)
            }
        )
        tasks.append(task)

    # Run performance tests
    results = {}

    # Test 1: Task batching performance
    print("📦 Testing task batching performance...")
    start_time = time.perf_counter()

    batch_count = 0
    while len(tasks) > 0:
        batch_ready = await planner.task_batcher.add_task(tasks.pop(0))
        if batch_ready:
            batch = await planner.task_batcher.get_batch()
            batch_count += 1

            # Simulate processing
            await asyncio.sleep(0.01)

    batching_time = time.perf_counter() - start_time
    results["batching"] = {
        "time_seconds": batching_time,
        "batches_created": batch_count,
        "batches_per_second": batch_count / batching_time if batching_time > 0 else 0
    }

    # Test 2: Memory pool performance
    print("💾 Testing memory pool performance...")
    start_time = time.perf_counter()

    objects = []
    for _ in range(1000):
        obj = planner.memory_pool.get_object("test_object", lambda: {"data": "test"})
        objects.append(obj)

    for obj in objects:
        planner.memory_pool.return_object("test_object", obj)

    memory_time = time.perf_counter() - start_time
    memory_stats = planner.memory_pool.get_memory_usage()

    results["memory_pool"] = {
        "time_seconds": memory_time,
        "operations_per_second": 2000 / memory_time if memory_time > 0 else 0,  # 1000 get + 1000 return
        "memory_stats": memory_stats
    }

    # Test 3: Load balancer performance
    print("⚖️ Testing load balancer performance...")
    start_time = time.perf_counter()

    selections = []
    for i in range(1000):
        task = QuantumTask(
            id=f"lb_test_{i}",
            name=f"LB Test {i}",
            description="Load balancer test",
            priority=TaskPriority.EXCITED_1,
            resources_required={"cpu": 1.0}
        )
        worker_id = planner.load_balancer.select_worker(task)
        selections.append(worker_id)

        # Simulate task completion
        if worker_id and i % 100 == 0:
            planner.load_balancer.complete_task(worker_id, 1.0, True)

    lb_time = time.perf_counter() - start_time
    load_distribution = planner.load_balancer.get_load_distribution()

    results["load_balancer"] = {
        "time_seconds": lb_time,
        "selections_per_second": 1000 / lb_time if lb_time > 0 else 0,
        "load_distribution": load_distribution
    }

    # Generate performance report
    performance_report = planner.profiler.get_performance_report()

    print("✅ Quantum Performance Benchmark Complete!")
    print(f"📊 Batching: {results['batching']['batches_per_second']:.1f} batches/sec")
    print(f"💾 Memory Pool: {results['memory_pool']['operations_per_second']:.1f} ops/sec")
    print(f"⚖️ Load Balancer: {results['load_balancer']['selections_per_second']:.1f} selections/sec")
    print(f"🎯 Load Balance Score: {results['load_balancer']['load_distribution']['load_balance_score']:.3f}")

    return {
        "benchmark_results": results,
        "performance_report": performance_report,
        "recommendations": planner.auto_scaler.get_scaling_recommendations({
            "cpu_percent": 75.0,
            "memory_percent": 60.0,
            "queue_length": 100
        })
    }


if __name__ == "__main__":
    import random
    asyncio.run(benchmark_quantum_performance())
