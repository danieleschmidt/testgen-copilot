"""
🛡️ Self-Healing System
======================

Advanced self-healing capabilities for autonomous SDLC execution.
Implements circuit breakers, automatic recovery, adaptive failure detection,
and intelligent system repair mechanisms.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json

from rich.console import Console

from .logging_config import setup_logger

logger = setup_logger(__name__)
console = Console()


class FailureType(Enum):
    """Types of failures the system can detect and heal"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    DATABASE_FAILURE = "database_failure"
    API_FAILURE = "api_failure"
    TIMEOUT = "timeout"
    MEMORY_LEAK = "memory_leak"
    DEADLOCK = "deadlock"
    CORRUPTION = "corruption"
    DEPENDENCY_FAILURE = "dependency_failure"


class HealingStrategy(Enum):
    """Available healing strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    RESOURCE_RESTART = "resource_restart"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    LOAD_BALANCING = "load_balancing"
    CACHE_INVALIDATION = "cache_invalidation"
    SERVICE_RESTART = "service_restart"
    ROLLBACK = "rollback"
    EMERGENCY_STOP = "emergency_stop"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, not allowing requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class FailureEvent:
    """Represents a detected failure"""
    failure_id: str
    failure_type: FailureType
    component: str
    severity: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    stack_trace: Optional[str] = None
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealingAction:
    """Represents a healing action taken"""
    action_id: str
    strategy: HealingStrategy
    target_component: str
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    execution_time: float = 0.0
    result_message: str = ""


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting services"""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open state
    
    # State
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    # Metrics
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class SelfHealingSystem:
    """
    🛡️ Self-Healing System for Autonomous SDLC
    
    Capabilities:
    - Automatic failure detection and classification
    - Circuit breakers for service protection
    - Intelligent healing strategy selection
    - Adaptive recovery mechanisms
    - Performance monitoring and optimization
    - Predictive failure prevention
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        
        # Healing state
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_history: List[FailureEvent] = []
        self.healing_history: List[HealingAction] = []
        self.component_health: Dict[str, float] = {}
        
        # Configuration
        self.healing_strategies: Dict[FailureType, List[HealingStrategy]] = self._initialize_healing_strategies()
        self.monitoring_interval = 30.0  # seconds
        self.max_healing_attempts = 3
        
        # Adaptive parameters
        self.failure_patterns: Dict[str, Any] = {}
        self.healing_success_rates: Dict[HealingStrategy, float] = {}
        
        # Load previous learning
        self._load_healing_knowledge()
        
        # Start background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def _initialize_healing_strategies(self) -> Dict[FailureType, List[HealingStrategy]]:
        """Initialize healing strategies for different failure types"""
        return {
            FailureType.PERFORMANCE_DEGRADATION: [
                HealingStrategy.CACHE_INVALIDATION,
                HealingStrategy.LOAD_BALANCING,
                HealingStrategy.GRACEFUL_DEGRADATION
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                HealingStrategy.RESOURCE_RESTART,
                HealingStrategy.GRACEFUL_DEGRADATION,
                HealingStrategy.LOAD_BALANCING
            ],
            FailureType.NETWORK_FAILURE: [
                HealingStrategy.RETRY,
                HealingStrategy.FALLBACK,
                HealingStrategy.CIRCUIT_BREAKER
            ],
            FailureType.DATABASE_FAILURE: [
                HealingStrategy.RETRY,
                HealingStrategy.FALLBACK,
                HealingStrategy.SERVICE_RESTART
            ],
            FailureType.API_FAILURE: [
                HealingStrategy.CIRCUIT_BREAKER,
                HealingStrategy.RETRY,
                HealingStrategy.FALLBACK
            ],
            FailureType.TIMEOUT: [
                HealingStrategy.RETRY,
                HealingStrategy.CIRCUIT_BREAKER,
                HealingStrategy.GRACEFUL_DEGRADATION
            ],
            FailureType.MEMORY_LEAK: [
                HealingStrategy.SERVICE_RESTART,
                HealingStrategy.RESOURCE_RESTART
            ],
            FailureType.DEADLOCK: [
                HealingStrategy.SERVICE_RESTART,
                HealingStrategy.EMERGENCY_STOP
            ],
            FailureType.CORRUPTION: [
                HealingStrategy.ROLLBACK,
                HealingStrategy.SERVICE_RESTART,
                HealingStrategy.EMERGENCY_STOP
            ],
            FailureType.DEPENDENCY_FAILURE: [
                HealingStrategy.FALLBACK,
                HealingStrategy.CIRCUIT_BREAKER,
                HealingStrategy.GRACEFUL_DEGRADATION
            ]
        }
    
    def _load_healing_knowledge(self) -> None:
        """Load previously learned healing knowledge"""
        knowledge_file = self.project_path / ".healing_knowledge.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file) as f:
                    data = json.load(f)
                    self.failure_patterns = data.get("failure_patterns", {})
                    self.healing_success_rates = data.get("healing_success_rates", {})
                logger.info("Loaded healing knowledge from previous runs")
            except Exception as e:
                logger.warning(f"Failed to load healing knowledge: {e}")
    
    def _save_healing_knowledge(self) -> None:
        """Save learned healing knowledge"""
        knowledge_file = self.project_path / ".healing_knowledge.json"
        try:
            data = {
                "failure_patterns": self.failure_patterns,
                "healing_success_rates": {k.value if isinstance(k, HealingStrategy) else k: v 
                                        for k, v in self.healing_success_rates.items()},
                "last_updated": datetime.now().isoformat()
            }
            with open(knowledge_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save healing knowledge: {e}")
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            return
        
        self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
        logger.info("Started self-healing monitoring system")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped self-healing monitoring system")
    
    async def _continuous_monitoring(self) -> None:
        """Continuous monitoring loop for proactive healing"""
        while True:
            try:
                await self._check_system_health()
                await self._update_circuit_breakers()
                await self._predict_failures()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retrying
    
    async def _check_system_health(self) -> None:
        """Check overall system health and detect issues"""
        components_to_check = [
            "database", "api", "cache", "memory", "cpu", "network", "disk"
        ]
        
        for component in components_to_check:
            health_score = await self._check_component_health(component)
            self.component_health[component] = health_score
            
            # Detect potential issues
            if health_score < 0.3:  # Critical
                await self._handle_detected_failure(
                    FailureType.PERFORMANCE_DEGRADATION,
                    component,
                    severity=1.0 - health_score
                )
            elif health_score < 0.6:  # Warning
                logger.warning(f"Component {component} showing degraded performance: {health_score:.2f}")
    
    async def _check_component_health(self, component: str) -> float:
        """Check health of a specific component"""
        # Simulate health checking - in practice, this would integrate with monitoring systems
        try:
            if component == "database":
                return await self._check_database_health()
            elif component == "api":
                return await self._check_api_health()
            elif component == "memory":
                return await self._check_memory_health()
            elif component == "cpu":
                return await self._check_cpu_health()
            else:
                return 0.9  # Default healthy state
        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")
            return 0.1  # Assume unhealthy if check fails
    
    async def _update_circuit_breakers(self) -> None:
        """Update circuit breaker states based on recent performance"""
        current_time = datetime.now()
        
        for breaker in self.circuit_breakers.values():
            if breaker.state == CircuitBreakerState.OPEN:
                # Check if enough time has passed to try recovery
                if (breaker.last_failure_time and 
                    (current_time - breaker.last_failure_time).total_seconds() > breaker.recovery_timeout):
                    breaker.state = CircuitBreakerState.HALF_OPEN
                    breaker.success_count = 0
                    logger.info(f"Circuit breaker {breaker.name} moved to HALF_OPEN state")
    
    async def _predict_failures(self) -> None:
        """Predict potential failures based on patterns"""
        # Analyze recent failure patterns
        recent_failures = [f for f in self.failure_history 
                         if (datetime.now() - f.timestamp).total_seconds() < 3600]
        
        if len(recent_failures) >= 3:
            # Pattern detected, take preventive action
            dominant_failure_type = max(set(f.failure_type for f in recent_failures),
                                      key=lambda x: sum(1 for f in recent_failures if f.failure_type == x))
            
            logger.warning(f"Failure pattern detected: {dominant_failure_type.value}")
            await self._take_preventive_action(dominant_failure_type)
    
    async def detect_failure(self, component: str, error: Exception, context: Dict[str, Any] = None) -> Optional[FailureEvent]:
        """Detect and classify a failure"""
        failure_type = self._classify_failure(error, context or {})
        severity = self._calculate_severity(error, context or {})
        
        failure = FailureEvent(
            failure_id=f"{component}_{int(time.time())}",
            failure_type=failure_type,
            component=component,
            severity=severity,
            error_message=str(error),
            stack_trace=str(error.__traceback__) if error.__traceback__ else None,
            context=context or {},
            metrics_snapshot=dict(self.component_health)
        )
        
        self.failure_history.append(failure)
        logger.error(f"Failure detected: {failure_type.value} in {component} (severity: {severity:.2f})")
        
        return failure
    
    def _classify_failure(self, error: Exception, context: Dict[str, Any]) -> FailureType:
        """Classify the type of failure based on error and context"""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "time" in error_str:
            return FailureType.TIMEOUT
        elif "memory" in error_str or "allocation" in error_str:
            return FailureType.MEMORY_LEAK
        elif "connection" in error_str or "network" in error_str:
            return FailureType.NETWORK_FAILURE
        elif "database" in error_str or "sql" in error_str:
            return FailureType.DATABASE_FAILURE
        elif "api" in error_str or "http" in error_str:
            return FailureType.API_FAILURE
        elif "deadlock" in error_str:
            return FailureType.DEADLOCK
        elif "corrupt" in error_str:
            return FailureType.CORRUPTION
        elif context.get("performance_degraded", False):
            return FailureType.PERFORMANCE_DEGRADATION
        else:
            return FailureType.DEPENDENCY_FAILURE
    
    def _calculate_severity(self, error: Exception, context: Dict[str, Any]) -> float:
        """Calculate failure severity (0.0 to 1.0)"""
        base_severity = 0.5
        
        # Adjust based on error type
        error_str = str(error).lower()
        if any(word in error_str for word in ["critical", "fatal", "emergency"]):
            base_severity += 0.3
        elif any(word in error_str for word in ["warning", "minor"]):
            base_severity -= 0.2
        
        # Adjust based on context
        if context.get("user_impact", False):
            base_severity += 0.2
        if context.get("data_loss_risk", False):
            base_severity += 0.3
        
        return max(0.0, min(1.0, base_severity))
    
    async def heal(self, failure: FailureEvent) -> bool:
        """Attempt to heal a detected failure"""
        logger.info(f"Starting healing process for {failure.failure_type.value} in {failure.component}")
        
        # Get available healing strategies for this failure type
        available_strategies = self.healing_strategies.get(failure.failure_type, [HealingStrategy.RETRY])
        
        # Sort strategies by success rate (adaptive learning)
        available_strategies = sorted(available_strategies, 
                                    key=lambda s: self.healing_success_rates.get(s, 0.5), 
                                    reverse=True)
        
        # Try strategies in order of preference
        for attempt in range(self.max_healing_attempts):
            for strategy in available_strategies:
                healing_action = HealingAction(
                    action_id=f"{failure.failure_id}_{strategy.value}_{attempt}",
                    strategy=strategy,
                    target_component=failure.component,
                    parameters={"attempt": attempt, "severity": failure.severity}
                )
                
                start_time = time.time()
                
                try:
                    success = await self._execute_healing_strategy(strategy, failure, healing_action)
                    healing_action.execution_time = time.time() - start_time
                    healing_action.success = success
                    
                    if success:
                        healing_action.result_message = f"Successfully healed {failure.failure_type.value}"
                        self.healing_history.append(healing_action)
                        
                        # Update success rate for adaptive learning
                        self._update_healing_success_rate(strategy, True)
                        
                        logger.info(f"Healing successful using {strategy.value}")
                        return True
                    else:
                        healing_action.result_message = f"Healing attempt failed with {strategy.value}"
                        
                except Exception as e:
                    healing_action.execution_time = time.time() - start_time
                    healing_action.success = False
                    healing_action.result_message = f"Healing error: {str(e)}"
                    logger.error(f"Healing strategy {strategy.value} failed: {e}")
                
                self.healing_history.append(healing_action)
                self._update_healing_success_rate(strategy, False)
        
        logger.error(f"All healing attempts failed for {failure.failure_type.value}")
        return False
    
    async def _execute_healing_strategy(self, strategy: HealingStrategy, failure: FailureEvent, action: HealingAction) -> bool:
        """Execute a specific healing strategy"""
        
        if strategy == HealingStrategy.RETRY:
            return await self._heal_with_retry(failure, action)
        elif strategy == HealingStrategy.FALLBACK:
            return await self._heal_with_fallback(failure, action)
        elif strategy == HealingStrategy.CIRCUIT_BREAKER:
            return await self._heal_with_circuit_breaker(failure, action)
        elif strategy == HealingStrategy.RESOURCE_RESTART:
            return await self._heal_with_resource_restart(failure, action)
        elif strategy == HealingStrategy.GRACEFUL_DEGRADATION:
            return await self._heal_with_graceful_degradation(failure, action)
        elif strategy == HealingStrategy.LOAD_BALANCING:
            return await self._heal_with_load_balancing(failure, action)
        elif strategy == HealingStrategy.CACHE_INVALIDATION:
            return await self._heal_with_cache_invalidation(failure, action)
        elif strategy == HealingStrategy.SERVICE_RESTART:
            return await self._heal_with_service_restart(failure, action)
        elif strategy == HealingStrategy.ROLLBACK:
            return await self._heal_with_rollback(failure, action)
        elif strategy == HealingStrategy.EMERGENCY_STOP:
            return await self._heal_with_emergency_stop(failure, action)
        else:
            logger.warning(f"Unknown healing strategy: {strategy}")
            return False
    
    def _update_healing_success_rate(self, strategy: HealingStrategy, success: bool) -> None:
        """Update success rate for a healing strategy (adaptive learning)"""
        if strategy not in self.healing_success_rates:
            self.healing_success_rates[strategy] = 0.5
        
        # Exponential moving average
        alpha = 0.1
        current_rate = self.healing_success_rates[strategy]
        new_rate = current_rate + alpha * (1.0 if success else 0.0 - current_rate)
        self.healing_success_rates[strategy] = new_rate
        
        # Save updated knowledge
        self._save_healing_knowledge()
    
    async def _handle_detected_failure(self, failure_type: FailureType, component: str, severity: float) -> None:
        """Handle a detected failure proactively"""
        failure = FailureEvent(
            failure_id=f"detected_{component}_{int(time.time())}",
            failure_type=failure_type,
            component=component,
            severity=severity,
            error_message=f"Proactively detected {failure_type.value}",
            context={"proactive": True}
        )
        
        success = await self.heal(failure)
        if not success:
            logger.error(f"Failed to heal proactively detected issue in {component}")
    
    async def _take_preventive_action(self, failure_type: FailureType) -> None:
        """Take preventive action based on predicted failure patterns"""
        logger.info(f"Taking preventive action for predicted {failure_type.value}")
        
        if failure_type == FailureType.PERFORMANCE_DEGRADATION:
            await self._preventive_performance_optimization()
        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            await self._preventive_resource_cleanup()
        elif failure_type == FailureType.MEMORY_LEAK:
            await self._preventive_memory_management()
    
    # Circuit breaker methods
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name=name)
        return self.circuit_breakers[name]
    
    async def call_with_circuit_breaker(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection"""
        breaker = self.get_circuit_breaker(name)
        breaker.total_requests += 1
        
        if breaker.state == CircuitBreakerState.OPEN:
            logger.warning(f"Circuit breaker {name} is OPEN, rejecting request")
            raise Exception(f"Circuit breaker {name} is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success
            breaker.total_successes += 1
            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.success_count += 1
                if breaker.success_count >= breaker.success_threshold:
                    breaker.state = CircuitBreakerState.CLOSED
                    breaker.failure_count = 0
                    logger.info(f"Circuit breaker {name} closed after recovery")
            elif breaker.state == CircuitBreakerState.CLOSED:
                breaker.failure_count = 0  # Reset failure count on success
            
            return result
            
        except Exception as e:
            # Failure
            breaker.total_failures += 1
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()
            
            if breaker.state == CircuitBreakerState.HALF_OPEN:
                # Return to open state
                breaker.state = CircuitBreakerState.OPEN
                breaker.success_count = 0
                logger.warning(f"Circuit breaker {name} reopened due to failure")
            elif (breaker.state == CircuitBreakerState.CLOSED and 
                  breaker.failure_count >= breaker.failure_threshold):
                # Open the circuit breaker
                breaker.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {name} opened due to {breaker.failure_count} failures")
            
            raise e
    
    # Placeholder healing strategy implementations
    async def _heal_with_retry(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Attempting retry healing for {failure.component}")
        await asyncio.sleep(1.0)  # Simulate retry delay
        return True
    
    async def _heal_with_fallback(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Activating fallback for {failure.component}")
        return True
    
    async def _heal_with_circuit_breaker(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Activating circuit breaker for {failure.component}")
        breaker = self.get_circuit_breaker(failure.component)
        breaker.state = CircuitBreakerState.OPEN
        return True
    
    async def _heal_with_resource_restart(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Restarting resources for {failure.component}")
        return True
    
    async def _heal_with_graceful_degradation(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Enabling graceful degradation for {failure.component}")
        return True
    
    async def _heal_with_load_balancing(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Adjusting load balancing for {failure.component}")
        return True
    
    async def _heal_with_cache_invalidation(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Invalidating cache for {failure.component}")
        return True
    
    async def _heal_with_service_restart(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Restarting service for {failure.component}")
        return True
    
    async def _heal_with_rollback(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.info(f"Rolling back changes for {failure.component}")
        return True
    
    async def _heal_with_emergency_stop(self, failure: FailureEvent, action: HealingAction) -> bool:
        logger.warning(f"Emergency stop activated for {failure.component}")
        return True
    
    # Health checking methods
    async def _check_database_health(self) -> float:
        # Simulate database health check
        return 0.9
    
    async def _check_api_health(self) -> float:
        # Simulate API health check
        return 0.85
    
    async def _check_memory_health(self) -> float:
        # Simulate memory health check
        return 0.8
    
    async def _check_cpu_health(self) -> float:
        # Simulate CPU health check
        return 0.75
    
    # Preventive action methods
    async def _preventive_performance_optimization(self) -> None:
        logger.info("Executing preventive performance optimization")
    
    async def _preventive_resource_cleanup(self) -> None:
        logger.info("Executing preventive resource cleanup")
    
    async def _preventive_memory_management(self) -> None:
        logger.info("Executing preventive memory management")