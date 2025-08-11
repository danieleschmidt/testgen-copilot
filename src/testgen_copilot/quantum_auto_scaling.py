"""
⚡ Quantum Auto-Scaling System
=============================

Advanced auto-scaling system using quantum-inspired algorithms for optimal resource allocation.
Implements intelligent scaling decisions, predictive resource management, and adaptive optimization.
"""

import asyncio
import logging
import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import math

from rich.console import Console
from rich.table import Table

from .logging_config import setup_logger
from .quantum_planner import QuantumTaskPlanner

logger = setup_logger(__name__)
console = Console()


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    QUANTUM_PROCESSORS = "quantum_processors"
    WORKERS = "workers"
    CONNECTIONS = "connections"


class ScalingStrategy(Enum):
    """Available scaling strategies"""
    REACTIVE = "reactive"          # React to current load
    PREDICTIVE = "predictive"      # Predict future load
    QUANTUM_OPTIMAL = "quantum_optimal"  # Quantum-inspired optimization
    ADAPTIVE = "adaptive"          # Learn and adapt over time
    CONSERVATIVE = "conservative"  # Minimize resource changes
    AGGRESSIVE = "aggressive"      # Maximize performance


@dataclass
class ResourceMetrics:
    """Current resource usage metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    network_throughput_mbps: float = 0.0
    storage_usage_percent: float = 0.0
    active_connections: int = 0
    queue_length: int = 0
    response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    throughput_rps: float = 0.0


@dataclass
class ScalingTrigger:
    """Defines when scaling should be triggered"""
    name: str
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    window_seconds: int = 300  # 5 minutes
    min_data_points: int = 5
    enabled: bool = True


@dataclass
class ScalingAction:
    """Represents a scaling action"""
    action_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    resource_type: ResourceType
    direction: ScalingDirection
    from_value: float = 0.0
    to_value: float = 0.0
    strategy_used: ScalingStrategy = ScalingStrategy.REACTIVE
    trigger_reason: str = ""
    execution_time: float = 0.0
    success: bool = False
    cost_impact: float = 0.0
    performance_impact: Dict[str, float] = field(default_factory=dict)


@dataclass
class QuantumScalingState:
    """Quantum superposition state for scaling decisions"""
    resource_allocations: Dict[ResourceType, List[float]]  # Possible allocation values
    probabilities: Dict[ResourceType, List[float]]         # Quantum probabilities
    entangled_resources: List[Tuple[ResourceType, ResourceType]]  # Resource correlations
    coherence_time: float = 60.0  # How long quantum states remain stable
    last_measurement: datetime = field(default_factory=datetime.now)


class QuantumAutoScaler:
    """
    ⚡ Quantum Auto-Scaling System
    
    Features:
    - Quantum-inspired resource optimization
    - Predictive scaling using ML algorithms
    - Multi-dimensional resource correlation analysis
    - Adaptive learning from scaling decisions
    - Cost-aware scaling optimization
    - Real-time performance monitoring
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        
        # Scaling state
        self.current_resources: Dict[ResourceType, float] = {}
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_history: List[ScalingAction] = []
        self.quantum_state = QuantumScalingState(
            resource_allocations={},
            probabilities={}
        )
        
        # Configuration
        self.scaling_triggers: List[ScalingTrigger] = []
        self.current_strategy = ScalingStrategy.QUANTUM_OPTIMAL
        self.min_scaling_interval = 60.0  # Minimum seconds between scaling actions
        self.max_scale_factor = 10.0      # Maximum scaling multiplier
        self.cost_weight = 0.3            # Weight of cost in optimization (0.0 to 1.0)
        
        # Performance tracking
        self.scaling_success_rates: Dict[ScalingStrategy, float] = {}
        self.resource_efficiency_history: Dict[ResourceType, List[float]] = {}
        
        # Initialize system
        self._initialize_scaling_triggers()
        self._initialize_quantum_state()
        self._load_scaling_knowledge()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._quantum_evolution_task: Optional[asyncio.Task] = None
    
    def _initialize_scaling_triggers(self) -> None:
        """Initialize default scaling triggers"""
        triggers = [
            ScalingTrigger(
                name="cpu_high_utilization",
                resource_type=ResourceType.CPU,
                metric_name="cpu_usage_percent",
                threshold_up=80.0,
                threshold_down=30.0,
                window_seconds=300
            ),
            ScalingTrigger(
                name="memory_high_utilization",
                resource_type=ResourceType.MEMORY,
                metric_name="memory_usage_percent",
                threshold_up=85.0,
                threshold_down=40.0,
                window_seconds=300
            ),
            ScalingTrigger(
                name="response_time_degradation",
                resource_type=ResourceType.WORKERS,
                metric_name="response_time_ms",
                threshold_up=500.0,
                threshold_down=100.0,
                window_seconds=180
            ),
            ScalingTrigger(
                name="queue_length_buildup",
                resource_type=ResourceType.WORKERS,
                metric_name="queue_length",
                threshold_up=50.0,
                threshold_down=5.0,
                window_seconds=120
            ),
            ScalingTrigger(
                name="error_rate_spike",
                resource_type=ResourceType.QUANTUM_PROCESSORS,
                metric_name="error_rate_percent",
                threshold_up=5.0,
                threshold_down=1.0,
                window_seconds=600
            )
        ]
        
        self.scaling_triggers = triggers
    
    def _initialize_quantum_state(self) -> None:
        """Initialize quantum superposition state for scaling"""
        resource_types = list(ResourceType)
        
        for resource_type in resource_types:
            # Create superposition of possible resource allocations
            base_allocation = self._get_current_allocation(resource_type)
            allocation_range = np.linspace(
                base_allocation * 0.5,  # Minimum 50% of current
                base_allocation * 3.0,  # Maximum 300% of current
                10  # 10 possible states
            )
            
            # Initialize with uniform probability distribution
            probabilities = np.ones(len(allocation_range)) / len(allocation_range)
            
            self.quantum_state.resource_allocations[resource_type] = allocation_range.tolist()
            self.quantum_state.probabilities[resource_type] = probabilities.tolist()
        
        # Initialize resource entanglements (correlations)
        self.quantum_state.entangled_resources = [
            (ResourceType.CPU, ResourceType.MEMORY),
            (ResourceType.WORKERS, ResourceType.CPU),
            (ResourceType.CONNECTIONS, ResourceType.NETWORK),
            (ResourceType.QUANTUM_PROCESSORS, ResourceType.MEMORY)
        ]
    
    def _load_scaling_knowledge(self) -> None:
        """Load previously learned scaling knowledge"""
        knowledge_file = self.project_path / ".scaling_knowledge.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file) as f:
                    data = json.load(f)
                    
                    # Load scaling success rates
                    for strategy_str, rate in data.get("success_rates", {}).items():
                        try:
                            strategy = ScalingStrategy(strategy_str)
                            self.scaling_success_rates[strategy] = rate
                        except ValueError:
                            continue
                    
                    # Load resource efficiency history
                    for resource_str, history in data.get("efficiency_history", {}).items():
                        try:
                            resource = ResourceType(resource_str)
                            self.resource_efficiency_history[resource] = history[-100:]  # Keep last 100
                        except ValueError:
                            continue
                
                logger.info("Loaded scaling knowledge from previous runs")
            except Exception as e:
                logger.warning(f"Failed to load scaling knowledge: {e}")
    
    def _save_scaling_knowledge(self) -> None:
        """Save learned scaling knowledge"""
        knowledge_file = self.project_path / ".scaling_knowledge.json"
        try:
            data = {
                "success_rates": {
                    strategy.value: rate 
                    for strategy, rate in self.scaling_success_rates.items()
                },
                "efficiency_history": {
                    resource.value: history 
                    for resource, history in self.resource_efficiency_history.items()
                },
                "last_updated": datetime.now().isoformat()
            }
            
            with open(knowledge_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scaling knowledge: {e}")
    
    async def start_auto_scaling(self) -> None:
        """Start automatic scaling monitoring and quantum evolution"""
        if self._monitoring_task and not self._monitoring_task.done():
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._quantum_evolution_task = asyncio.create_task(self._quantum_evolution_loop())
        
        logger.info("Started quantum auto-scaling system")
    
    async def stop_auto_scaling(self) -> None:
        """Stop automatic scaling"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._quantum_evolution_task:
            self._quantum_evolution_task.cancel()
        
        for task in [self._monitoring_task, self._quantum_evolution_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Stopped quantum auto-scaling system")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for scaling decisions"""
        while True:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
                
                # Check scaling triggers
                scaling_needed = await self._evaluate_scaling_triggers(metrics)
                
                if scaling_needed:
                    await self._execute_scaling_decision()
                
                # Update quantum state probabilities based on performance
                await self._update_quantum_probabilities(metrics)
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _quantum_evolution_loop(self) -> None:
        """Quantum state evolution loop"""
        while True:
            try:
                await self._evolve_quantum_state()
                await asyncio.sleep(10.0)  # Evolve every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quantum evolution error: {e}")
                await asyncio.sleep(5.0)
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource usage metrics"""
        # In practice, this would integrate with monitoring systems
        # For now, simulate realistic metrics
        
        return ResourceMetrics(
            cpu_usage_percent=np.random.normal(60, 15),
            memory_usage_percent=np.random.normal(65, 20),
            network_throughput_mbps=np.random.normal(100, 25),
            storage_usage_percent=np.random.normal(40, 10),
            active_connections=max(0, int(np.random.normal(500, 100))),
            queue_length=max(0, int(np.random.normal(10, 5))),
            response_time_ms=max(10, np.random.normal(150, 50)),
            error_rate_percent=max(0, np.random.normal(2, 1)),
            throughput_rps=max(0, np.random.normal(50, 15))
        )
    
    async def _evaluate_scaling_triggers(self, current_metrics: ResourceMetrics) -> bool:
        """Evaluate if any scaling triggers are activated"""
        
        for trigger in self.scaling_triggers:
            if not trigger.enabled:
                continue
            
            # Get recent metrics for the trigger's window
            window_start = datetime.now() - timedelta(seconds=trigger.window_seconds)
            recent_metrics = [m for m in self.metrics_history if m.timestamp > window_start]
            
            if len(recent_metrics) < trigger.min_data_points:
                continue
            
            # Extract the metric values
            metric_values = []
            for metrics in recent_metrics:
                value = getattr(metrics, trigger.metric_name, 0.0)
                metric_values.append(value)
            
            # Calculate average over the window
            avg_value = np.mean(metric_values)
            
            # Check thresholds
            if avg_value >= trigger.threshold_up:
                logger.info(f"Scaling trigger activated: {trigger.name} (avg: {avg_value:.2f} >= {trigger.threshold_up})")
                return True
            elif avg_value <= trigger.threshold_down:
                logger.info(f"Scale-down trigger activated: {trigger.name} (avg: {avg_value:.2f} <= {trigger.threshold_down})")
                return True
        
        return False
    
    async def _execute_scaling_decision(self) -> None:
        """Execute scaling decision using current strategy"""
        
        if self.current_strategy == ScalingStrategy.QUANTUM_OPTIMAL:
            await self._quantum_optimal_scaling()
        elif self.current_strategy == ScalingStrategy.PREDICTIVE:
            await self._predictive_scaling()
        elif self.current_strategy == ScalingStrategy.ADAPTIVE:
            await self._adaptive_scaling()
        else:
            await self._reactive_scaling()
    
    async def _quantum_optimal_scaling(self) -> None:
        """Execute quantum-optimal scaling decision"""
        logger.info("Executing quantum-optimal scaling")
        
        # Measure quantum states to get optimal resource allocations
        optimal_allocations = await self._measure_quantum_states()
        
        # Execute scaling actions for each resource
        for resource_type, target_allocation in optimal_allocations.items():
            current_allocation = self._get_current_allocation(resource_type)
            
            if abs(target_allocation - current_allocation) / current_allocation > 0.1:  # 10% threshold
                direction = ScalingDirection.UP if target_allocation > current_allocation else ScalingDirection.DOWN
                
                action = ScalingAction(
                    action_id=f"quantum_{resource_type.value}_{int(time.time())}",
                    resource_type=resource_type,
                    direction=direction,
                    from_value=current_allocation,
                    to_value=target_allocation,
                    strategy_used=ScalingStrategy.QUANTUM_OPTIMAL,
                    trigger_reason="Quantum optimization"
                )
                
                await self._execute_scaling_action(action)
    
    async def _measure_quantum_states(self) -> Dict[ResourceType, float]:
        """Measure quantum states to collapse superposition into optimal allocations"""
        optimal_allocations = {}
        
        for resource_type in self.quantum_state.resource_allocations.keys():
            allocations = self.quantum_state.resource_allocations[resource_type]
            probabilities = self.quantum_state.probabilities[resource_type]
            
            # Choose allocation based on probability distribution
            optimal_index = np.argmax(probabilities)
            optimal_allocations[resource_type] = allocations[optimal_index]
        
        # Handle entangled resources
        for resource1, resource2 in self.quantum_state.entangled_resources:
            if resource1 in optimal_allocations and resource2 in optimal_allocations:
                # Adjust for entanglement correlation
                correlation_factor = 0.7  # Strong correlation
                avg_ratio = (optimal_allocations[resource1] + optimal_allocations[resource2]) / 2
                optimal_allocations[resource1] = optimal_allocations[resource1] * correlation_factor + avg_ratio * (1 - correlation_factor)
                optimal_allocations[resource2] = optimal_allocations[resource2] * correlation_factor + avg_ratio * (1 - correlation_factor)
        
        return optimal_allocations
    
    async def _execute_scaling_action(self, action: ScalingAction) -> None:
        """Execute a specific scaling action"""
        logger.info(f"Executing scaling action: {action.resource_type.value} from {action.from_value:.2f} to {action.to_value:.2f}")
        
        start_time = time.time()
        
        try:
            # Execute the actual scaling (placeholder implementation)
            success = await self._scale_resource(action.resource_type, action.to_value)
            
            action.execution_time = time.time() - start_time
            action.success = success
            
            if success:
                # Update current resource allocation
                self.current_resources[action.resource_type] = action.to_value
                
                # Calculate cost and performance impact
                action.cost_impact = self._calculate_cost_impact(action)
                action.performance_impact = await self._measure_performance_impact(action)
                
                # Update success rate
                self._update_strategy_success_rate(action.strategy_used, True)
                
                logger.info(f"Scaling action completed successfully")
            else:
                action.trigger_reason += " (FAILED)"
                self._update_strategy_success_rate(action.strategy_used, False)
                logger.error(f"Scaling action failed")
        
        except Exception as e:
            action.execution_time = time.time() - start_time
            action.success = False
            action.trigger_reason += f" (ERROR: {str(e)})"
            self._update_strategy_success_rate(action.strategy_used, False)
            logger.error(f"Scaling action error: {e}")
        
        self.scaling_history.append(action)
        self._save_scaling_knowledge()
    
    async def _scale_resource(self, resource_type: ResourceType, target_value: float) -> bool:
        """Execute actual resource scaling (placeholder)"""
        # Simulate scaling delay
        await asyncio.sleep(0.5)
        
        # Simulate success/failure
        return np.random.random() > 0.1  # 90% success rate
    
    def _calculate_cost_impact(self, action: ScalingAction) -> float:
        """Calculate cost impact of scaling action"""
        # Simple cost model - more resources = higher cost
        base_cost = 0.10  # $0.10 per unit per hour
        
        resource_multipliers = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 0.5,
            ResourceType.STORAGE: 0.2,
            ResourceType.NETWORK: 0.3,
            ResourceType.QUANTUM_PROCESSORS: 5.0,  # Quantum is expensive
            ResourceType.WORKERS: 2.0,
            ResourceType.CONNECTIONS: 0.1
        }
        
        multiplier = resource_multipliers.get(action.resource_type, 1.0)
        cost_change = (action.to_value - action.from_value) * base_cost * multiplier
        
        return cost_change
    
    async def _measure_performance_impact(self, action: ScalingAction) -> Dict[str, float]:
        """Measure performance impact of scaling action"""
        # Wait a bit for scaling to take effect
        await asyncio.sleep(2.0)
        
        # Collect metrics before and after
        after_metrics = await self._collect_metrics()
        
        # Calculate improvements (placeholder)
        return {
            "response_time_improvement_percent": np.random.normal(10, 5) if action.direction == ScalingDirection.UP else -5,
            "throughput_improvement_percent": np.random.normal(15, 8) if action.direction == ScalingDirection.UP else -8,
            "error_rate_improvement_percent": np.random.normal(5, 2)
        }
    
    async def _update_quantum_probabilities(self, current_metrics: ResourceMetrics) -> None:
        """Update quantum state probabilities based on current performance"""
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(current_metrics)
        
        # Update probabilities for each resource type
        for resource_type in self.quantum_state.resource_allocations.keys():
            current_allocation = self._get_current_allocation(resource_type)
            allocations = self.quantum_state.resource_allocations[resource_type]
            probabilities = self.quantum_state.probabilities[resource_type]
            
            # Find closest allocation in quantum states
            closest_index = np.argmin([abs(a - current_allocation) for a in allocations])
            
            # Increase probability for current state if performance is good
            if performance_score > 0.7:
                probabilities[closest_index] *= 1.1
            elif performance_score < 0.3:
                probabilities[closest_index] *= 0.9
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                self.quantum_state.probabilities[resource_type] = probabilities
    
    def _calculate_performance_score(self, metrics: ResourceMetrics) -> float:
        """Calculate overall performance score from metrics"""
        # Normalize metrics to 0-1 scale (higher is better)
        cpu_score = max(0, (100 - metrics.cpu_usage_percent) / 100)  # Lower CPU usage is better
        memory_score = max(0, (100 - metrics.memory_usage_percent) / 100)  # Lower memory usage is better
        response_time_score = max(0, (1000 - metrics.response_time_ms) / 1000)  # Lower response time is better
        error_rate_score = max(0, (10 - metrics.error_rate_percent) / 10)  # Lower error rate is better
        throughput_score = min(1, metrics.throughput_rps / 100)  # Higher throughput is better (cap at 100 RPS)
        
        # Weighted average
        weights = [0.25, 0.25, 0.25, 0.15, 0.1]  # CPU, Memory, Response Time, Error Rate, Throughput
        scores = [cpu_score, memory_score, response_time_score, error_rate_score, throughput_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    async def _evolve_quantum_state(self) -> None:
        """Evolve quantum state over time (quantum decoherence)"""
        current_time = datetime.now()
        time_since_last_measurement = (current_time - self.quantum_state.last_measurement).total_seconds()
        
        if time_since_last_measurement > self.quantum_state.coherence_time:
            # Quantum decoherence - probabilities become more uniform
            decoherence_factor = min(1.0, time_since_last_measurement / self.quantum_state.coherence_time)
            
            for resource_type in self.quantum_state.probabilities.keys():
                probabilities = self.quantum_state.probabilities[resource_type]
                uniform_prob = 1.0 / len(probabilities)
                
                # Move towards uniform distribution
                new_probabilities = []
                for prob in probabilities:
                    new_prob = prob * (1 - decoherence_factor) + uniform_prob * decoherence_factor
                    new_probabilities.append(new_prob)
                
                self.quantum_state.probabilities[resource_type] = new_probabilities
    
    def _update_strategy_success_rate(self, strategy: ScalingStrategy, success: bool) -> None:
        """Update success rate for scaling strategy"""
        if strategy not in self.scaling_success_rates:
            self.scaling_success_rates[strategy] = 0.5
        
        # Exponential moving average
        alpha = 0.1
        current_rate = self.scaling_success_rates[strategy]
        new_rate = current_rate + alpha * (1.0 if success else 0.0 - current_rate)
        self.scaling_success_rates[strategy] = new_rate
    
    def _get_current_allocation(self, resource_type: ResourceType) -> float:
        """Get current allocation for a resource type"""
        # Default allocations if not set
        defaults = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 4.0,
            ResourceType.STORAGE: 100.0,
            ResourceType.NETWORK: 1000.0,
            ResourceType.QUANTUM_PROCESSORS: 1.0,
            ResourceType.WORKERS: 4.0,
            ResourceType.CONNECTIONS: 100.0
        }
        
        return self.current_resources.get(resource_type, defaults.get(resource_type, 1.0))
    
    async def _predictive_scaling(self) -> None:
        """Execute predictive scaling based on historical patterns"""
        logger.info("Executing predictive scaling")
        
        # Analyze historical patterns to predict future load
        if len(self.metrics_history) < 10:
            # Not enough data, fall back to reactive scaling
            await self._reactive_scaling()
            return
        
        # Simple trend analysis
        recent_metrics = self.metrics_history[-10:]
        cpu_trend = np.polyfit(range(len(recent_metrics)), 
                              [m.cpu_usage_percent for m in recent_metrics], 1)[0]
        memory_trend = np.polyfit(range(len(recent_metrics)), 
                                 [m.memory_usage_percent for m in recent_metrics], 1)[0]
        
        # Predict future resource needs
        prediction_horizon = 300  # 5 minutes ahead
        predicted_cpu = recent_metrics[-1].cpu_usage_percent + cpu_trend * prediction_horizon / 30
        predicted_memory = recent_metrics[-1].memory_usage_percent + memory_trend * prediction_horizon / 30
        
        # Scale proactively if trends indicate future issues
        if predicted_cpu > 80:
            await self._scale_resource_proactively(ResourceType.CPU, 1.5)
        if predicted_memory > 85:
            await self._scale_resource_proactively(ResourceType.MEMORY, 1.3)
    
    async def _scale_resource_proactively(self, resource_type: ResourceType, scale_factor: float) -> None:
        """Scale a resource proactively"""
        current_allocation = self._get_current_allocation(resource_type)
        target_allocation = current_allocation * scale_factor
        
        action = ScalingAction(
            action_id=f"predictive_{resource_type.value}_{int(time.time())}",
            resource_type=resource_type,
            direction=ScalingDirection.UP,
            from_value=current_allocation,
            to_value=target_allocation,
            strategy_used=ScalingStrategy.PREDICTIVE,
            trigger_reason="Predictive scaling based on trend analysis"
        )
        
        await self._execute_scaling_action(action)
    
    async def _reactive_scaling(self) -> None:
        """Execute reactive scaling based on current conditions"""
        logger.info("Executing reactive scaling")
        
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # Simple reactive scaling rules
        if current_metrics.cpu_usage_percent > 80:
            await self._scale_resource_reactively(ResourceType.CPU, 1.25)
        elif current_metrics.cpu_usage_percent < 30:
            await self._scale_resource_reactively(ResourceType.CPU, 0.8)
        
        if current_metrics.memory_usage_percent > 85:
            await self._scale_resource_reactively(ResourceType.MEMORY, 1.2)
        elif current_metrics.memory_usage_percent < 40:
            await self._scale_resource_reactively(ResourceType.MEMORY, 0.9)
        
        if current_metrics.response_time_ms > 500:
            await self._scale_resource_reactively(ResourceType.WORKERS, 1.5)
        elif current_metrics.response_time_ms < 100 and current_metrics.queue_length < 5:
            await self._scale_resource_reactively(ResourceType.WORKERS, 0.8)
    
    async def _scale_resource_reactively(self, resource_type: ResourceType, scale_factor: float) -> None:
        """Scale a resource reactively"""
        current_allocation = self._get_current_allocation(resource_type)
        target_allocation = current_allocation * scale_factor
        
        # Apply limits
        target_allocation = max(target_allocation, current_allocation * 0.5)  # Don't scale down more than 50%
        target_allocation = min(target_allocation, current_allocation * self.max_scale_factor)  # Don't scale up more than max_scale_factor
        
        direction = ScalingDirection.UP if target_allocation > current_allocation else ScalingDirection.DOWN
        
        action = ScalingAction(
            action_id=f"reactive_{resource_type.value}_{int(time.time())}",
            resource_type=resource_type,
            direction=direction,
            from_value=current_allocation,
            to_value=target_allocation,
            strategy_used=ScalingStrategy.REACTIVE,
            trigger_reason="Reactive scaling based on current metrics"
        )
        
        await self._execute_scaling_action(action)
    
    async def _adaptive_scaling(self) -> None:
        """Execute adaptive scaling based on learned patterns"""
        logger.info("Executing adaptive scaling")
        
        # Choose strategy with highest success rate
        if self.scaling_success_rates:
            best_strategy = max(self.scaling_success_rates.items(), key=lambda x: x[1])[0]
            
            if best_strategy == ScalingStrategy.PREDICTIVE:
                await self._predictive_scaling()
            elif best_strategy == ScalingStrategy.QUANTUM_OPTIMAL:
                await self._quantum_optimal_scaling()
            else:
                await self._reactive_scaling()
        else:
            # No learning data, use quantum optimal
            await self._quantum_optimal_scaling()
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status"""
        recent_actions = [a for a in self.scaling_history 
                         if (datetime.now() - a.timestamp).total_seconds() < 3600]
        
        return {
            "current_strategy": self.current_strategy.value,
            "active_resources": dict(self.current_resources),
            "recent_actions_count": len(recent_actions),
            "success_rates": {s.value: rate for s, rate in self.scaling_success_rates.items()},
            "quantum_state_coherence": max(0, 1.0 - (datetime.now() - self.quantum_state.last_measurement).total_seconds() / self.quantum_state.coherence_time),
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
        }
    
    def display_scaling_dashboard(self) -> None:
        """Display scaling system dashboard"""
        table = Table(title="⚡ Quantum Auto-Scaling Dashboard")
        table.add_column("Resource", style="cyan")
        table.add_column("Current", style="magenta")
        table.add_column("Quantum State", style="green")
        table.add_column("Status", style="yellow")
        
        for resource_type in ResourceType:
            current_allocation = self._get_current_allocation(resource_type)
            quantum_probs = self.quantum_state.probabilities.get(resource_type, [])
            max_prob_index = np.argmax(quantum_probs) if quantum_probs else 0
            
            quantum_state = "Superposition" if len(quantum_probs) > 1 and max(quantum_probs) < 0.9 else "Collapsed"
            status = "✅ Optimal" if quantum_state == "Collapsed" else "🌀 Evolving"
            
            table.add_row(
                resource_type.value.title(),
                f"{current_allocation:.2f}",
                quantum_state,
                status
            )
        
        console.print(table)
        
        # Display recent scaling actions
        if self.scaling_history:
            recent_actions = self.scaling_history[-5:]
            
            actions_table = Table(title="Recent Scaling Actions")
            actions_table.add_column("Time", style="dim")
            actions_table.add_column("Resource", style="cyan")
            actions_table.add_column("Direction", style="magenta")
            actions_table.add_column("Strategy", style="green")
            actions_table.add_column("Success", style="yellow")
            
            for action in recent_actions:
                time_str = action.timestamp.strftime("%H:%M:%S")
                direction_emoji = "📈" if action.direction == ScalingDirection.UP else "📉" if action.direction == ScalingDirection.DOWN else "➡️"
                success_emoji = "✅" if action.success else "❌"
                
                actions_table.add_row(
                    time_str,
                    action.resource_type.value,
                    f"{direction_emoji} {action.direction.value}",
                    action.strategy_used.value,
                    success_emoji
                )
            
            console.print(actions_table)