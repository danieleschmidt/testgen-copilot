"""Intelligent auto-scaling and load balancing for TestGen Copilot."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import deque, defaultdict
import statistics

from .logging_config import get_core_logger
from .monitoring import get_health_monitor
from .performance_optimizer import get_performance_optimizer


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CONSERVATIVE = "conservative"  # Scale slowly, prioritize stability
    AGGRESSIVE = "aggressive"     # Scale quickly, prioritize performance
    BALANCED = "balanced"         # Balanced approach
    CUSTOM = "custom"            # User-defined scaling rules


class WorkloadPattern(Enum):
    """Detected workload patterns."""
    STEADY = "steady"           # Consistent load
    BURSTY = "bursty"          # Intermittent high loads
    GRADUAL_INCREASE = "gradual_increase"  # Slowly increasing
    GRADUAL_DECREASE = "gradual_decrease"  # Slowly decreasing
    CYCLICAL = "cyclical"       # Regular patterns
    UNPREDICTABLE = "unpredictable"  # No clear pattern


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_depth: int = 0
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate_percent: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    action: str  # "scale_up", "scale_down", "no_change"
    current_capacity: int
    target_capacity: int
    reason: str
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Intelligent load balancer with health-aware routing."""
    
    def __init__(self):
        self.logger = get_core_logger()
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.health_scores = defaultdict(float)
        self._lock = threading.RLock()
        
    def register_worker(self, worker_id: str, capacity: int = 100, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new worker."""
        with self._lock:
            self.workers[worker_id] = {
                "capacity": capacity,
                "current_load": 0,
                "healthy": True,
                "last_health_check": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc)
            }
            self.health_scores[worker_id] = 1.0
            
        self.logger.info("Worker registered", {
            "worker_id": worker_id,
            "capacity": capacity,
            "metadata": metadata
        })
        
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                if worker_id in self.health_scores:
                    del self.health_scores[worker_id]
                if worker_id in self.request_counts:
                    del self.request_counts[worker_id]
                if worker_id in self.response_times:
                    del self.response_times[worker_id]
                    
        self.logger.info("Worker unregistered", {"worker_id": worker_id})
        
    def select_worker(self, algorithm: str = "least_connections") -> Optional[str]:
        """Select optimal worker based on algorithm."""
        with self._lock:
            healthy_workers = {
                wid: worker for wid, worker in self.workers.items()
                if worker["healthy"] and worker["current_load"] < worker["capacity"]
            }
            
            if not healthy_workers:
                return None
                
            if algorithm == "round_robin":
                # Simple round-robin
                return min(healthy_workers.keys(), 
                          key=lambda wid: self.request_counts[wid])
                          
            elif algorithm == "least_connections":
                # Least current connections
                return min(healthy_workers.keys(),
                          key=lambda wid: healthy_workers[wid]["current_load"])
                          
            elif algorithm == "weighted_response_time":
                # Weight by response time and health score
                def weight_function(wid):
                    recent_times = self.response_times[wid][-10:]  # Last 10 requests
                    avg_response = statistics.mean(recent_times) if recent_times else 100.0
                    health_penalty = (2.0 - self.health_scores[wid])  # Lower score = higher penalty
                    return avg_response * health_penalty
                    
                return min(healthy_workers.keys(), key=weight_function)
                
            elif algorithm == "capacity_aware":
                # Consider remaining capacity percentage
                def capacity_score(wid):
                    worker = healthy_workers[wid]
                    utilization = worker["current_load"] / worker["capacity"]
                    health_bonus = self.health_scores[wid]
                    return utilization / health_bonus  # Lower is better
                    
                return min(healthy_workers.keys(), key=capacity_score)
                
            else:
                # Default to least connections
                return min(healthy_workers.keys(),
                          key=lambda wid: healthy_workers[wid]["current_load"])
                          
    def record_request(self, worker_id: str, response_time_ms: float, 
                      success: bool = True) -> None:
        """Record request completion."""
        with self._lock:
            if worker_id in self.workers:
                self.request_counts[worker_id] += 1
                self.response_times[worker_id].append(response_time_ms)
                
                # Keep only recent response times
                if len(self.response_times[worker_id]) > 100:
                    self.response_times[worker_id] = self.response_times[worker_id][-100:]
                    
                # Update health score based on success rate
                if success:
                    self.health_scores[worker_id] = min(2.0, self.health_scores[worker_id] + 0.1)
                else:
                    self.health_scores[worker_id] = max(0.1, self.health_scores[worker_id] - 0.05)
                    
    def update_worker_load(self, worker_id: str, current_load: int) -> None:
        """Update worker's current load."""
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id]["current_load"] = current_load
                
    def mark_worker_unhealthy(self, worker_id: str, reason: str = "") -> None:
        """Mark worker as unhealthy."""
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id]["healthy"] = False
                self.health_scores[worker_id] = 0.1
                
        self.logger.warning("Worker marked unhealthy", {
            "worker_id": worker_id,
            "reason": reason
        })
        
    def mark_worker_healthy(self, worker_id: str) -> None:
        """Mark worker as healthy."""
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id]["healthy"] = True
                self.health_scores[worker_id] = max(0.5, self.health_scores[worker_id])
                
        self.logger.info("Worker marked healthy", {"worker_id": worker_id})
        
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            total_capacity = sum(w["capacity"] for w in self.workers.values())
            total_load = sum(w["current_load"] for w in self.workers.values())
            healthy_workers = sum(1 for w in self.workers.values() if w["healthy"])
            
            worker_stats = {}
            for wid, worker in self.workers.items():
                recent_times = self.response_times[wid][-10:]
                worker_stats[wid] = {
                    "capacity": worker["capacity"],
                    "current_load": worker["current_load"],
                    "utilization_percent": (worker["current_load"] / worker["capacity"] * 100) 
                                         if worker["capacity"] > 0 else 0,
                    "healthy": worker["healthy"],
                    "health_score": round(self.health_scores[wid], 2),
                    "total_requests": self.request_counts[wid],
                    "avg_response_time_ms": round(statistics.mean(recent_times), 2) 
                                          if recent_times else 0
                }
                
            return {
                "total_workers": len(self.workers),
                "healthy_workers": healthy_workers,
                "total_capacity": total_capacity,
                "total_load": total_load,
                "overall_utilization_percent": (total_load / total_capacity * 100) 
                                             if total_capacity > 0 else 0,
                "worker_stats": worker_stats
            }


class WorkloadAnalyzer:
    """Analyzes workload patterns for predictive scaling."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.pattern_cache: Dict[str, Any] = {}
        self.logger = get_core_logger()
        
    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record new metrics sample."""
        self.metrics_history.append(metrics)
        
        # Clear pattern cache when we have new data
        self.pattern_cache.clear()
        
    def detect_pattern(self) -> WorkloadPattern:
        """Detect current workload pattern."""
        if len(self.metrics_history) < 10:
            return WorkloadPattern.UNPREDICTABLE
            
        if "pattern" in self.pattern_cache:
            return self.pattern_cache["pattern"]
            
        # Analyze recent CPU utilization trends
        recent_cpu = [m.cpu_utilization for m in list(self.metrics_history)[-20:]]
        
        if len(recent_cpu) < 10:
            pattern = WorkloadPattern.UNPREDICTABLE
        else:
            # Calculate trend and variance
            x = list(range(len(recent_cpu)))
            slope = self._calculate_trend(x, recent_cpu)
            variance = statistics.variance(recent_cpu)
            mean_cpu = statistics.mean(recent_cpu)
            
            if variance < 5:  # Low variance
                pattern = WorkloadPattern.STEADY
            elif slope > 2:  # Increasing trend
                pattern = WorkloadPattern.GRADUAL_INCREASE
            elif slope < -2:  # Decreasing trend
                pattern = WorkloadPattern.GRADUAL_DECREASE
            elif variance > 20 and mean_cpu > 50:  # High variance, high load
                pattern = WorkloadPattern.BURSTY
            elif self._detect_cyclical(recent_cpu):
                pattern = WorkloadPattern.CYCLICAL
            else:
                pattern = WorkloadPattern.UNPREDICTABLE
                
        self.pattern_cache["pattern"] = pattern
        return pattern
        
    def predict_next_load(self, horizon_minutes: int = 5) -> float:
        """Predict load for next time period."""
        if len(self.metrics_history) < 5:
            return 50.0  # Default prediction
            
        pattern = self.detect_pattern()
        recent_metrics = list(self.metrics_history)[-10:]
        recent_cpu = [m.cpu_utilization for m in recent_metrics]
        
        if pattern == WorkloadPattern.STEADY:
            return statistics.mean(recent_cpu)
            
        elif pattern == WorkloadPattern.GRADUAL_INCREASE:
            slope = self._calculate_trend(list(range(len(recent_cpu))), recent_cpu)
            return min(100.0, recent_cpu[-1] + (slope * horizon_minutes))
            
        elif pattern == WorkloadPattern.GRADUAL_DECREASE:
            slope = self._calculate_trend(list(range(len(recent_cpu))), recent_cpu)
            return max(0.0, recent_cpu[-1] + (slope * horizon_minutes))
            
        elif pattern == WorkloadPattern.BURSTY:
            # Assume burst will continue, scale for peak
            return min(100.0, max(recent_cpu) * 1.2)
            
        elif pattern == WorkloadPattern.CYCLICAL:
            # Use historical pattern
            return self._predict_cyclical(recent_cpu, horizon_minutes)
            
        else:  # UNPREDICTABLE
            # Conservative prediction based on recent peak
            return min(100.0, max(recent_cpu) * 1.1)
            
    def _calculate_trend(self, x: List[int], y: List[float]) -> float:
        """Calculate linear trend slope."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
        
    def _detect_cyclical(self, values: List[float]) -> bool:
        """Detect if values show cyclical pattern."""
        if len(values) < 8:
            return False
            
        # Simple autocorrelation check
        n = len(values)
        mean_val = statistics.mean(values)
        
        for lag in [2, 3, 4, 5, 6]:  # Check common cycle lengths
            if lag >= n // 2:
                continue
                
            correlation = 0.0
            count = 0
            
            for i in range(lag, n):
                correlation += (values[i] - mean_val) * (values[i - lag] - mean_val)
                count += 1
                
            if count > 0:
                correlation /= count
                if correlation > 10:  # Arbitrary threshold
                    return True
                    
        return False
        
    def _predict_cyclical(self, values: List[float], horizon: int) -> float:
        """Predict next value for cyclical pattern."""
        if len(values) < 4:
            return statistics.mean(values)
            
        # Simple approach: assume pattern repeats every 4-6 data points
        cycle_length = 4
        cycle_position = len(values) % cycle_length
        next_position = (cycle_position + horizon) % cycle_length
        
        # Average values at similar cycle positions
        similar_values = [values[i] for i in range(next_position, len(values), cycle_length)]
        
        if similar_values:
            return statistics.mean(similar_values)
        else:
            return values[-1]
            
    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive workload analysis report."""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
            
        recent_metrics = list(self.metrics_history)[-20:]
        
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        memory_values = [m.memory_utilization for m in recent_metrics]
        response_times = [m.response_time_ms for m in recent_metrics]
        
        pattern = self.detect_pattern()
        prediction = self.predict_next_load()
        
        return {
            "current_pattern": pattern.value,
            "prediction_5min": round(prediction, 1),
            "metrics_summary": {
                "avg_cpu_percent": round(statistics.mean(cpu_values), 1),
                "max_cpu_percent": round(max(cpu_values), 1),
                "avg_memory_percent": round(statistics.mean(memory_values), 1),
                "avg_response_time_ms": round(statistics.mean(response_times), 1),
                "cpu_variance": round(statistics.variance(cpu_values), 1) if len(cpu_values) >= 2 else 0.0,
                "samples_analyzed": len(recent_metrics)
            },
            "trend_analysis": {
                "cpu_trend": self._calculate_trend(list(range(len(cpu_values))), cpu_values),
                "memory_trend": self._calculate_trend(list(range(len(memory_values))), memory_values),
                "is_cyclical": self._detect_cyclical(cpu_values)
            },
            "recommendations": self._generate_scaling_recommendations(pattern, prediction, recent_metrics)
        }
        
    def _generate_scaling_recommendations(self, pattern: WorkloadPattern, 
                                        prediction: float, 
                                        recent_metrics: List[ScalingMetrics]) -> List[str]:
        """Generate scaling recommendations based on analysis."""
        recommendations = []
        
        if pattern == WorkloadPattern.BURSTY:
            recommendations.append("Consider pre-scaling to handle burst loads effectively")
            
        elif pattern == WorkloadPattern.GRADUAL_INCREASE:
            recommendations.append("Gradual increase detected - consider proactive scaling")
            
        elif pattern == WorkloadPattern.CYCLICAL:
            recommendations.append("Cyclical pattern detected - implement scheduled scaling")
            
        if prediction > 80:
            recommendations.append("High load predicted - recommend immediate scale-up")
        elif prediction < 20:
            recommendations.append("Low load predicted - consider scale-down to optimize costs")
            
        avg_response_time = statistics.mean([m.response_time_ms for m in recent_metrics])
        if avg_response_time > 1000:
            recommendations.append("High response times - increase capacity immediately")
            
        return recommendations


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, policy: ScalingPolicy = ScalingPolicy.BALANCED):
        self.policy = policy
        self.logger = get_core_logger()
        self.load_balancer = LoadBalancer()
        self.workload_analyzer = WorkloadAnalyzer()
        
        # Scaling configuration
        self.min_workers = 1
        self.max_workers = 20
        self.scale_up_threshold = 70.0    # CPU %
        self.scale_down_threshold = 30.0  # CPU %
        self.scale_up_cooldown = 120      # seconds
        self.scale_down_cooldown = 300    # seconds
        
        self.last_scale_up = datetime.min.replace(tzinfo=timezone.utc)
        self.last_scale_down = datetime.min.replace(tzinfo=timezone.utc)
        self.scaling_decisions: List[ScalingDecision] = []
        
        self._configure_policy(policy)
        
    def _configure_policy(self, policy: ScalingPolicy) -> None:
        """Configure scaling parameters based on policy."""
        if policy == ScalingPolicy.CONSERVATIVE:
            self.scale_up_threshold = 80.0
            self.scale_down_threshold = 20.0
            self.scale_up_cooldown = 300   # 5 minutes
            self.scale_down_cooldown = 600 # 10 minutes
            
        elif policy == ScalingPolicy.AGGRESSIVE:
            self.scale_up_threshold = 60.0
            self.scale_down_threshold = 40.0
            self.scale_up_cooldown = 60    # 1 minute
            self.scale_down_cooldown = 180 # 3 minutes
            
        elif policy == ScalingPolicy.BALANCED:
            self.scale_up_threshold = 70.0
            self.scale_down_threshold = 30.0
            self.scale_up_cooldown = 120   # 2 minutes
            self.scale_down_cooldown = 300 # 5 minutes
            
    def should_scale(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Determine if scaling action is needed."""
        self.workload_analyzer.record_metrics(metrics)
        
        current_workers = len(self.load_balancer.workers)
        now = datetime.now(timezone.utc)
        
        # Check cooldown periods
        scale_up_ready = (now - self.last_scale_up).total_seconds() >= self.scale_up_cooldown
        scale_down_ready = (now - self.last_scale_down).total_seconds() >= self.scale_down_cooldown
        
        # Predictive scaling
        predicted_load = self.workload_analyzer.predict_next_load()
        pattern = self.workload_analyzer.detect_pattern()
        
        # Scale up conditions
        scale_up_reasons = []
        if metrics.cpu_utilization >= self.scale_up_threshold:
            scale_up_reasons.append(f"CPU utilization {metrics.cpu_utilization:.1f}% >= {self.scale_up_threshold:.1f}%")
            
        if metrics.response_time_ms > 2000:
            scale_up_reasons.append(f"High response time: {metrics.response_time_ms:.1f}ms")
            
        if metrics.queue_depth > current_workers * 10:
            scale_up_reasons.append(f"Queue depth {metrics.queue_depth} exceeds capacity")
            
        if predicted_load > self.scale_up_threshold:
            scale_up_reasons.append(f"Predicted load {predicted_load:.1f}% exceeds threshold")
            
        # Scale down conditions
        scale_down_reasons = []
        if metrics.cpu_utilization <= self.scale_down_threshold and current_workers > self.min_workers:
            scale_down_reasons.append(f"CPU utilization {metrics.cpu_utilization:.1f}% <= {self.scale_down_threshold:.1f}%")
            
        if metrics.response_time_ms < 500 and predicted_load < self.scale_down_threshold:
            scale_down_reasons.append("Low response time and predicted load")
            
        # Make scaling decision
        if scale_up_reasons and scale_up_ready and current_workers < self.max_workers:
            target_capacity = min(self.max_workers, current_workers + self._calculate_scale_amount(metrics, "up"))
            confidence = self._calculate_confidence(metrics, pattern, "up")
            
            return ScalingDecision(
                action="scale_up",
                current_capacity=current_workers,
                target_capacity=target_capacity,
                reason="; ".join(scale_up_reasons),
                confidence=confidence,
                metadata={"predicted_load": predicted_load, "pattern": pattern.value}
            )
            
        elif scale_down_reasons and scale_down_ready and current_workers > self.min_workers:
            target_capacity = max(self.min_workers, current_workers - self._calculate_scale_amount(metrics, "down"))
            confidence = self._calculate_confidence(metrics, pattern, "down")
            
            return ScalingDecision(
                action="scale_down",
                current_capacity=current_workers,
                target_capacity=target_capacity,
                reason="; ".join(scale_down_reasons),
                confidence=confidence,
                metadata={"predicted_load": predicted_load, "pattern": pattern.value}
            )
            
        else:
            return ScalingDecision(
                action="no_change",
                current_capacity=current_workers,
                target_capacity=current_workers,
                reason="No scaling conditions met or in cooldown period",
                confidence=0.8,
                metadata={"scale_up_ready": scale_up_ready, "scale_down_ready": scale_down_ready}
            )
            
    def _calculate_scale_amount(self, metrics: ScalingMetrics, direction: str) -> int:
        """Calculate how many workers to add/remove."""
        if direction == "up":
            # More aggressive scaling for high load or error rates
            if metrics.cpu_utilization > 90 or metrics.error_rate_percent > 5:
                return 3
            elif metrics.cpu_utilization > 80:
                return 2
            else:
                return 1
                
        else:  # scale down
            # Conservative scale down
            return 1
            
    def _calculate_confidence(self, metrics: ScalingMetrics, 
                            pattern: WorkloadPattern, direction: str) -> float:
        """Calculate confidence in scaling decision."""
        base_confidence = 0.7
        
        # Higher confidence for clear patterns
        if pattern in [WorkloadPattern.STEADY, WorkloadPattern.CYCLICAL]:
            base_confidence += 0.1
            
        # Higher confidence for extreme conditions
        if direction == "up":
            if metrics.cpu_utilization > 90:
                base_confidence += 0.2
            if metrics.response_time_ms > 3000:
                base_confidence += 0.1
        else:
            if metrics.cpu_utilization < 20:
                base_confidence += 0.1
                
        return min(1.0, base_confidence)
        
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision."""
        self.scaling_decisions.append(decision)
        
        if decision.action == "no_change":
            return True
            
        self.logger.info("Executing scaling decision", {
            "action": decision.action,
            "current_capacity": decision.current_capacity,
            "target_capacity": decision.target_capacity,
            "reason": decision.reason,
            "confidence": decision.confidence
        })
        
        try:
            if decision.action == "scale_up":
                current_workers = len(self.load_balancer.workers)
                workers_needed = decision.target_capacity - current_workers
                self._scale_up(workers_needed)
                self.last_scale_up = decision.timestamp
                
            elif decision.action == "scale_down":
                self._scale_down(decision.current_capacity - decision.target_capacity)
                self.last_scale_down = decision.timestamp
                
            return True
            
        except Exception as e:
            self.logger.error("Scaling execution failed", {
                "action": decision.action,
                "error": str(e)
            })
            return False
            
    def _scale_up(self, worker_count: int) -> None:
        """Add new workers."""
        for i in range(worker_count):
            worker_id = f"worker_{int(time.time())}_{i}"
            self.load_balancer.register_worker(worker_id, capacity=100)
            
    def _scale_down(self, worker_count: int) -> None:
        """Remove workers gracefully."""
        # Remove least loaded workers first
        workers_by_load = sorted(
            self.load_balancer.workers.items(),
            key=lambda x: x[1]["current_load"]
        )
        
        for i in range(min(worker_count, len(workers_by_load))):
            worker_id, _ = workers_by_load[i]
            self.load_balancer.unregister_worker(worker_id)
            
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling report."""
        load_stats = self.load_balancer.get_load_stats()
        analysis_report = self.workload_analyzer.get_analysis_report()
        
        recent_decisions = self.scaling_decisions[-10:] if self.scaling_decisions else []
        
        return {
            "auto_scaling_status": {
                "policy": self.policy.value,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "last_scale_up": self.last_scale_up.isoformat(),
                "last_scale_down": self.last_scale_down.isoformat()
            },
            "load_balancing": load_stats,
            "workload_analysis": analysis_report,
            "recent_scaling_decisions": [
                {
                    "action": d.action,
                    "current_capacity": d.current_capacity,
                    "target_capacity": d.target_capacity,
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in recent_decisions
            ],
            "scaling_effectiveness": self._calculate_scaling_effectiveness()
        }
        
    def _calculate_scaling_effectiveness(self) -> Dict[str, Any]:
        """Calculate how effective recent scaling decisions have been."""
        if not self.scaling_decisions:
            return {"status": "insufficient_data"}
            
        recent_decisions = self.scaling_decisions[-20:]  # Last 20 decisions
        
        scale_up_count = sum(1 for d in recent_decisions if d.action == "scale_up")
        scale_down_count = sum(1 for d in recent_decisions if d.action == "scale_down")
        no_change_count = sum(1 for d in recent_decisions if d.action == "no_change")
        
        avg_confidence = statistics.mean([d.confidence for d in recent_decisions])
        
        return {
            "total_decisions": len(recent_decisions),
            "scale_up_count": scale_up_count,
            "scale_down_count": scale_down_count,
            "no_change_count": no_change_count,
            "average_confidence": round(avg_confidence, 2),
            "scaling_frequency": len([d for d in recent_decisions if d.action != "no_change"]) / len(recent_decisions),
            "status": "effective" if avg_confidence > 0.7 else "needs_tuning"
        }


# Global auto-scaler instance
_auto_scaler: Optional[AutoScaler] = None

def get_auto_scaler(policy: ScalingPolicy = ScalingPolicy.BALANCED) -> AutoScaler:
    """Get or create global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(policy)
    return _auto_scaler