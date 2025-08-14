"""
Adaptive Intelligence Module

Self-improving patterns with machine learning-like capabilities:
- Adaptive parameter tuning based on execution history
- Pattern recognition for optimal configurations
- Predictive optimization suggestions
- Self-healing and recovery strategies
"""

import asyncio
import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle


class LearningMode(Enum):
    """Learning modes for adaptive intelligence"""
    EXPLORATION = "exploration"    # Trying new configurations
    EXPLOITATION = "exploitation"  # Using best known configurations  
    BALANCED = "balanced"         # Mix of exploration and exploitation


@dataclass
class ExecutionPattern:
    """Execution pattern for learning"""
    pattern_id: str
    context: Dict[str, Any]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    success_rate: float
    execution_count: int = 0
    last_used: Optional[datetime] = None
    confidence_score: float = 0.0


@dataclass
class AdaptiveRule:
    """Adaptive rule for parameter adjustment"""
    rule_id: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class PatternRecognizer:
    """Recognizes execution patterns and suggests optimizations"""
    
    def __init__(self, history_limit: int = 1000):
        self.history_limit = history_limit
        self.execution_history: List[Dict[str, Any]] = []
        self.recognized_patterns: Dict[str, ExecutionPattern] = {}
        
    def record_execution(self, context: Dict[str, Any], parameters: Dict[str, Any], 
                        metrics: Dict[str, float], success: bool) -> None:
        """Record execution for pattern learning"""
        
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "parameters": parameters,
            "metrics": metrics,
            "success": success
        }
        
        self.execution_history.append(execution_record)
        
        # Maintain history limit
        if len(self.execution_history) > self.history_limit:
            self.execution_history = self.execution_history[-self.history_limit:]
        
        # Update patterns
        self._update_patterns(execution_record)
    
    def _update_patterns(self, execution: Dict[str, Any]) -> None:
        """Update recognized patterns based on new execution"""
        
        # Create pattern key based on context
        context = execution["context"]
        pattern_key = self._create_pattern_key(context)
        
        if pattern_key not in self.recognized_patterns:
            # Create new pattern
            self.recognized_patterns[pattern_key] = ExecutionPattern(
                pattern_id=pattern_key,
                context=context,
                parameters=execution["parameters"].copy(),
                performance_metrics=execution["metrics"].copy(),
                success_rate=1.0 if execution["success"] else 0.0,
                execution_count=1,
                last_used=datetime.utcnow(),
                confidence_score=0.1  # Low initial confidence
            )
        else:
            # Update existing pattern
            pattern = self.recognized_patterns[pattern_key]
            pattern.execution_count += 1
            pattern.last_used = datetime.utcnow()
            
            # Update success rate (weighted average)
            weight = 0.9  # Bias toward recent executions
            new_success = 1.0 if execution["success"] else 0.0
            pattern.success_rate = (pattern.success_rate * weight + new_success * (1 - weight))
            
            # Update parameters (moving average)
            for param_name, param_value in execution["parameters"].items():
                if param_name in pattern.parameters and isinstance(param_value, (int, float)):
                    pattern.parameters[param_name] = (
                        pattern.parameters[param_name] * weight + param_value * (1 - weight)
                    )
            
            # Update confidence based on execution count and success rate
            pattern.confidence_score = min(1.0, (pattern.execution_count * pattern.success_rate) / 10)
    
    def _create_pattern_key(self, context: Dict[str, Any]) -> str:
        """Create pattern key from context"""
        
        # Use relevant context features for pattern recognition
        key_features = []
        
        for key in ["project_type", "phase", "task_type", "complexity"]:
            if key in context:
                key_features.append(f"{key}:{context[key]}")
        
        return "|".join(sorted(key_features))
    
    def get_best_pattern(self, context: Dict[str, Any]) -> Optional[ExecutionPattern]:
        """Get best matching pattern for given context"""
        
        pattern_key = self._create_pattern_key(context)
        
        if pattern_key in self.recognized_patterns:
            return self.recognized_patterns[pattern_key]
        
        # Find similar patterns
        similar_patterns = []
        for pattern in self.recognized_patterns.values():
            similarity = self._calculate_similarity(context, pattern.context)
            if similarity > 0.7:  # Threshold for similarity
                similar_patterns.append((similarity, pattern))
        
        if similar_patterns:
            # Return pattern with highest similarity * confidence
            similar_patterns.sort(key=lambda x: x[0] * x[1].confidence_score, reverse=True)
            return similar_patterns[0][1]
        
        return None
    
    def _calculate_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts"""
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def suggest_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal parameters based on learned patterns"""
        
        best_pattern = self.get_best_pattern(context)
        
        if best_pattern and best_pattern.confidence_score > 0.5:
            return best_pattern.parameters.copy()
        
        # Fallback to safe defaults
        return {
            "retry_multiplier": 1.0,
            "timeout_seconds": 30.0,
            "max_concurrent": 4,
            "cache_size": 1000
        }


class AdaptiveParameterTuner:
    """Tunes parameters adaptively based on performance feedback"""
    
    def __init__(self):
        self.parameter_ranges: Dict[str, Tuple[float, float]] = {
            "retry_multiplier": (0.5, 3.0),
            "timeout_seconds": (10.0, 120.0),
            "max_concurrent": (1, 20),
            "cache_size": (100, 5000),
            "quality_threshold": (0.6, 1.0)
        }
        
        self.current_parameters: Dict[str, float] = {
            "retry_multiplier": 1.0,
            "timeout_seconds": 30.0, 
            "max_concurrent": 4,
            "cache_size": 1000,
            "quality_threshold": 0.85
        }
        
        self.performance_history: List[Tuple[Dict[str, float], float]] = []
        self.learning_rate = 0.1
        
    def suggest_adjustment(self, parameter_name: str, current_performance: float) -> Optional[float]:
        """Suggest parameter adjustment based on performance"""
        
        if parameter_name not in self.parameter_ranges:
            return None
        
        min_val, max_val = self.parameter_ranges[parameter_name]
        current_val = self.current_parameters[parameter_name]
        
        # Simple gradient-based adjustment
        if len(self.performance_history) >= 2:
            recent_perf = [perf for _, perf in self.performance_history[-5:]]
            perf_trend = self._calculate_trend(recent_perf)
            
            if perf_trend < -0.1:  # Performance declining
                # Try adjusting parameter
                if current_performance < 0.8:  # Poor performance
                    # More aggressive adjustment
                    adjustment = (max_val - current_val) * self.learning_rate * 2
                else:
                    # Conservative adjustment
                    adjustment = (max_val - current_val) * self.learning_rate
            else:
                # Performance stable or improving, small adjustments
                adjustment = (max_val - current_val) * self.learning_rate * 0.5
            
            new_val = max(min_val, min(max_val, current_val + adjustment))
            return new_val
        
        return None
    
    def update_parameters(self, performance_metrics: Dict[str, float], overall_performance: float) -> Dict[str, float]:
        """Update parameters based on performance feedback"""
        
        # Record performance
        self.performance_history.append((self.current_parameters.copy(), overall_performance))
        
        # Keep history limited
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Suggest adjustments for each parameter
        adjustments = {}
        
        for param_name in self.current_parameters:
            suggested_val = self.suggest_adjustment(param_name, overall_performance)
            if suggested_val is not None and suggested_val != self.current_parameters[param_name]:
                adjustments[param_name] = suggested_val
                self.current_parameters[param_name] = suggested_val
        
        return adjustments
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Calculate slope (trend)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


class SelfHealingManager:
    """Manages self-healing and recovery strategies"""
    
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {
            "restart_component": self._restart_component,
            "clear_cache": self._clear_cache,
            "reduce_load": self._reduce_load,
            "failover": self._initiate_failover
        }
        
        self.failure_patterns: List[AdaptiveRule] = []
        self.recovery_history: List[Dict[str, Any]] = []
        
    async def analyze_failure(self, error_context: Dict[str, Any]) -> List[str]:
        """Analyze failure and suggest healing strategies"""
        
        suggested_strategies = []
        
        # Rule-based analysis
        error_type = error_context.get("error_type", "unknown")
        error_message = error_context.get("error_message", "")
        
        if "timeout" in error_message.lower():
            suggested_strategies.extend(["reduce_load", "restart_component"])
        elif "memory" in error_message.lower():
            suggested_strategies.extend(["clear_cache", "reduce_load"])
        elif "connection" in error_message.lower():
            suggested_strategies.extend(["restart_component", "failover"])
        else:
            suggested_strategies.append("restart_component")  # Default
        
        # Learn from similar failures
        similar_failures = self._find_similar_failures(error_context)
        for failure in similar_failures:
            if failure["recovery_success"]:
                suggested_strategies.extend(failure["strategies_used"])
        
        # Remove duplicates while preserving order
        unique_strategies = []
        for strategy in suggested_strategies:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)
        
        return unique_strategies
    
    async def execute_healing(self, strategies: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute healing strategies in order"""
        
        healing_results = {
            "strategies_attempted": [],
            "strategies_successful": [],
            "recovery_time": 0.0,
            "final_status": "failed"
        }
        
        start_time = datetime.utcnow()
        
        for strategy in strategies:
            if strategy in self.healing_strategies:
                try:
                    healing_results["strategies_attempted"].append(strategy)
                    success = await self.healing_strategies[strategy](context)
                    
                    if success:
                        healing_results["strategies_successful"].append(strategy)
                        
                        # Test if system is healthy now
                        if await self._verify_health(context):
                            healing_results["final_status"] = "recovered"
                            break
                            
                except Exception as e:
                    logging.getLogger(__name__).error(f"Healing strategy {strategy} failed: {e}")
        
        healing_results["recovery_time"] = (datetime.utcnow() - start_time).total_seconds()
        
        # Record recovery attempt for learning
        recovery_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "strategies_used": healing_results["strategies_attempted"],
            "recovery_success": healing_results["final_status"] == "recovered",
            "recovery_time": healing_results["recovery_time"]
        }
        
        self.recovery_history.append(recovery_record)
        
        return healing_results
    
    def _find_similar_failures(self, error_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar failures in history"""
        
        similar_failures = []
        
        for recovery in self.recovery_history:
            similarity = self._calculate_context_similarity(error_context, recovery["context"])
            if similarity > 0.6:  # Similarity threshold
                similar_failures.append(recovery)
        
        # Sort by similarity and recency
        similar_failures.sort(
            key=lambda x: (
                self._calculate_context_similarity(error_context, x["context"]),
                x["timestamp"]  # More recent is better
            ),
            reverse=True
        )
        
        return similar_failures[:5]  # Top 5 similar failures
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between error contexts"""
        
        # Simple keyword-based similarity
        str1 = " ".join(str(v) for v in context1.values())
        str2 = " ".join(str(v) for v in context2.values())
        
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    async def _restart_component(self, context: Dict[str, Any]) -> bool:
        """Restart failed component"""
        
        component = context.get("component", "unknown")
        logging.getLogger(__name__).info(f"🔄 Restarting component: {component}")
        
        # Simulate component restart
        await asyncio.sleep(1)
        
        return True  # Assume restart successful
    
    async def _clear_cache(self, context: Dict[str, Any]) -> bool:
        """Clear caches to free memory"""
        
        logging.getLogger(__name__).info("🧹 Clearing caches")
        
        # Simulate cache clearing
        await asyncio.sleep(0.5)
        
        return True
    
    async def _reduce_load(self, context: Dict[str, Any]) -> bool:
        """Reduce system load"""
        
        logging.getLogger(__name__).info("📉 Reducing system load")
        
        # Simulate load reduction (e.g., scaling down, throttling)
        await asyncio.sleep(0.3)
        
        return True
    
    async def _initiate_failover(self, context: Dict[str, Any]) -> bool:
        """Initiate failover to backup systems"""
        
        logging.getLogger(__name__).info("🔀 Initiating failover")
        
        # Simulate failover process
        await asyncio.sleep(2)
        
        return True
    
    async def _verify_health(self, context: Dict[str, Any]) -> bool:
        """Verify system health after healing"""
        
        # Simulate health check
        await asyncio.sleep(0.5)
        
        # Assume health check passes 80% of the time
        import random
        return random.random() > 0.2


class AdaptiveIntelligence:
    """
    Main adaptive intelligence coordinator implementing self-improving patterns.
    
    Combines pattern recognition, parameter tuning, and self-healing for
    continuous optimization and autonomous improvement.
    """
    
    def __init__(self, project_path: Path, config: Optional[Dict[str, Any]] = None):
        self.project_path = project_path
        self.config = config or {}
        
        # Core components
        self.pattern_recognizer = PatternRecognizer()
        self.parameter_tuner = AdaptiveParameterTuner()
        self.self_healing_manager = SelfHealingManager()
        
        # Learning configuration
        self.learning_mode = LearningMode.BALANCED
        self.exploration_rate = 0.2  # 20% exploration, 80% exploitation
        
        # State persistence
        self.state_file = project_path / ".adaptive_intelligence_state.pkl"
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize adaptive intelligence system"""
        
        try:
            self.logger.info("🧠 Initializing Adaptive Intelligence...")
            
            # Load previous state if available
            await self._load_state()
            
            self.logger.info("✅ Adaptive Intelligence initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize adaptive intelligence: {e}")
            return False
    
    async def optimize_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize execution using adaptive intelligence"""
        
        self.logger.info("🎯 Optimizing execution with adaptive intelligence...")
        
        optimization_results = {
            "context": context,
            "optimizations_applied": [],
            "parameter_adjustments": {},
            "performance_improvement": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # 1. Get optimal parameters from learned patterns
            suggested_params = self.pattern_recognizer.suggest_parameters(context)
            
            if suggested_params:
                optimization_results["parameter_adjustments"].update(suggested_params)
                optimization_results["optimizations_applied"].append("pattern_based_parameters")
            
            # 2. Apply adaptive parameter tuning
            current_performance = context.get("current_performance", 0.7)
            tuning_adjustments = self.parameter_tuner.update_parameters(
                context.get("performance_metrics", {}),
                current_performance
            )
            
            if tuning_adjustments:
                optimization_results["parameter_adjustments"].update(tuning_adjustments)
                optimization_results["optimizations_applied"].append("adaptive_tuning")
            
            # 3. Exploration vs exploitation decision
            if self._should_explore():
                exploration_params = await self._generate_exploration_parameters(context)
                optimization_results["parameter_adjustments"].update(exploration_params)
                optimization_results["optimizations_applied"].append("exploration_parameters")
            
            self.logger.info(f"✅ Applied {len(optimization_results['optimizations_applied'])} optimizations")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            optimization_results["error"] = str(e)
            return optimization_results
    
    async def handle_failure(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failures with adaptive self-healing"""
        
        self.logger.warning("🚨 Handling failure with adaptive intelligence...")
        
        failure_response = {
            "failure_context": failure_context,
            "healing_strategies": [],
            "recovery_status": "failed",
            "recovery_time": 0.0,
            "lessons_learned": []
        }
        
        try:
            # 1. Analyze failure pattern
            suggested_strategies = await self.self_healing_manager.analyze_failure(failure_context)
            failure_response["healing_strategies"] = suggested_strategies
            
            # 2. Execute healing strategies
            healing_results = await self.self_healing_manager.execute_healing(
                suggested_strategies, failure_context
            )
            
            failure_response.update(healing_results)
            
            # 3. Learn from the failure and recovery
            await self._learn_from_failure(failure_context, healing_results)
            
            self.logger.info(f"🔧 Recovery {'successful' if healing_results['final_status'] == 'recovered' else 'failed'}")
            
            return failure_response
            
        except Exception as e:
            self.logger.error(f"❌ Failure handling failed: {e}")
            failure_response["error"] = str(e)
            return failure_response
    
    async def record_execution_outcome(self, context: Dict[str, Any], parameters: Dict[str, Any],
                                     metrics: Dict[str, float], success: bool) -> None:
        """Record execution outcome for learning"""
        
        # Record in pattern recognizer
        self.pattern_recognizer.record_execution(context, parameters, metrics, success)
        
        # Update adaptive parameters
        overall_performance = metrics.get("overall_score", 0.7)
        self.parameter_tuner.update_parameters(metrics, overall_performance)
        
        # Persist state periodically
        await self._save_state()
        
        self.logger.debug(f"📝 Recorded execution outcome (success: {success})")
    
    def _should_explore(self) -> bool:
        """Decide whether to explore new parameters or exploit known good ones"""
        
        import random
        
        if self.learning_mode == LearningMode.EXPLORATION:
            return True
        elif self.learning_mode == LearningMode.EXPLOITATION:
            return False
        else:  # BALANCED
            return random.random() < self.exploration_rate
    
    async def _generate_exploration_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for exploration"""
        
        import random
        
        exploration_params = {}
        
        for param_name, (min_val, max_val) in self.parameter_tuner.parameter_ranges.items():
            # Generate random value within range
            if isinstance(min_val, int):
                exploration_params[param_name] = random.randint(int(min_val), int(max_val))
            else:
                exploration_params[param_name] = random.uniform(min_val, max_val)
        
        self.logger.debug("🔍 Generated exploration parameters")
        return exploration_params
    
    async def _learn_from_failure(self, failure_context: Dict[str, Any], healing_results: Dict[str, Any]) -> None:
        """Learn from failure and recovery patterns"""
        
        # Create adaptive rule based on failure pattern
        if healing_results["final_status"] == "recovered":
            successful_strategies = healing_results["strategies_successful"]
            
            # Create rule for similar failures
            rule = AdaptiveRule(
                rule_id=f"failure_rule_{datetime.utcnow().timestamp()}",
                condition={
                    "error_type": failure_context.get("error_type"),
                    "component": failure_context.get("component")
                },
                action={
                    "strategies": successful_strategies[:2]  # Top 2 successful strategies
                },
                confidence=0.7,  # Initial confidence
                success_count=1
            )
            
            self.self_healing_manager.failure_patterns.append(rule)
            
            self.logger.info(f"📚 Learned new healing pattern: {successful_strategies}")
    
    async def _save_state(self) -> None:
        """Save adaptive intelligence state to disk"""
        
        try:
            state = {
                "pattern_recognizer": {
                    "recognized_patterns": self.pattern_recognizer.recognized_patterns,
                    "execution_history": self.pattern_recognizer.execution_history[-100:]  # Last 100
                },
                "parameter_tuner": {
                    "current_parameters": self.parameter_tuner.current_parameters,
                    "performance_history": self.parameter_tuner.performance_history[-50:]  # Last 50
                },
                "self_healing_manager": {
                    "failure_patterns": self.self_healing_manager.failure_patterns,
                    "recovery_history": self.self_healing_manager.recovery_history[-50:]  # Last 50
                },
                "learning_mode": self.learning_mode.value,
                "exploration_rate": self.exploration_rate,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.debug("💾 Adaptive intelligence state saved")
            
        except Exception as e:
            self.logger.warning(f"Failed to save adaptive intelligence state: {e}")
    
    async def _load_state(self) -> None:
        """Load adaptive intelligence state from disk"""
        
        if not self.state_file.exists():
            self.logger.debug("No previous state found, starting fresh")
            return
        
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            
            # Restore pattern recognizer
            if "pattern_recognizer" in state:
                pr_state = state["pattern_recognizer"]
                self.pattern_recognizer.recognized_patterns = pr_state.get("recognized_patterns", {})
                self.pattern_recognizer.execution_history = pr_state.get("execution_history", [])
            
            # Restore parameter tuner
            if "parameter_tuner" in state:
                pt_state = state["parameter_tuner"]
                self.parameter_tuner.current_parameters.update(pt_state.get("current_parameters", {}))
                self.parameter_tuner.performance_history = pt_state.get("performance_history", [])
            
            # Restore self-healing manager
            if "self_healing_manager" in state:
                sh_state = state["self_healing_manager"]
                self.self_healing_manager.failure_patterns = sh_state.get("failure_patterns", [])
                self.self_healing_manager.recovery_history = sh_state.get("recovery_history", [])
            
            # Restore learning configuration
            if "learning_mode" in state:
                self.learning_mode = LearningMode(state["learning_mode"])
            if "exploration_rate" in state:
                self.exploration_rate = state["exploration_rate"]
            
            self.logger.info(f"📥 Loaded adaptive intelligence state from {state.get('saved_at', 'unknown')}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load adaptive intelligence state: {e}")
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get adaptive intelligence metrics and status"""
        
        return {
            "patterns_learned": len(self.pattern_recognizer.recognized_patterns),
            "executions_recorded": len(self.pattern_recognizer.execution_history),
            "current_parameters": self.parameter_tuner.current_parameters.copy(),
            "performance_history_size": len(self.parameter_tuner.performance_history),
            "failure_patterns_learned": len(self.self_healing_manager.failure_patterns),
            "recovery_attempts": len(self.self_healing_manager.recovery_history),
            "learning_mode": self.learning_mode.value,
            "exploration_rate": self.exploration_rate,
            "state_file_exists": self.state_file.exists()
        }