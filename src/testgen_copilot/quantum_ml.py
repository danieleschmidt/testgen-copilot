"""Quantum machine learning for intelligent task prediction and optimization."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .quantum_planner import QuantumTask, ResourceQuantum, TaskPriority, TaskState


@dataclass
class TaskFeatures:
    """Feature representation of quantum tasks for ML."""

    # Basic features
    priority_level: float
    estimated_hours: float
    dependency_count: int
    resource_cpu: float
    resource_memory: float
    resource_io: float

    # Temporal features
    days_to_deadline: float
    hour_of_day: int
    day_of_week: int

    # Quantum features
    entanglement_count: int
    urgency_score: float
    quantum_uncertainty: float

    # Historical features
    avg_completion_time: float = 0.0
    success_rate: float = 1.0
    complexity_score: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector."""
        return np.array([
            self.priority_level,
            self.estimated_hours,
            self.dependency_count,
            self.resource_cpu,
            self.resource_memory,
            self.resource_io,
            self.days_to_deadline,
            self.hour_of_day / 24.0,  # Normalize to 0-1
            self.day_of_week / 7.0,   # Normalize to 0-1
            self.entanglement_count,
            self.urgency_score,
            self.quantum_uncertainty,
            self.avg_completion_time,
            self.success_rate,
            self.complexity_score
        ])

    @classmethod
    def feature_names(cls) -> List[str]:
        """Get feature names for model interpretation."""
        return [
            "priority_level", "estimated_hours", "dependency_count",
            "resource_cpu", "resource_memory", "resource_io",
            "days_to_deadline", "hour_of_day_norm", "day_of_week_norm",
            "entanglement_count", "urgency_score", "quantum_uncertainty",
            "avg_completion_time", "success_rate", "complexity_score"
        ]


class QuantumTaskPredictor:
    """ML model for predicting task completion and optimization."""

    def __init__(self, learning_rate: float = 0.01, regularization: float = 0.001):
        """Initialize quantum task predictor."""
        self.learning_rate = learning_rate
        self.regularization = regularization

        # Model weights (initialized randomly)
        self.feature_count = len(TaskFeatures.feature_names())
        self.weights = np.random.normal(0, 0.1, self.feature_count)
        self.bias = 0.0

        # Training data
        self.training_features: List[np.ndarray] = []
        self.training_targets: List[float] = []

        # Model performance tracking
        self.training_history: List[Dict[str, float]] = []
        self.prediction_cache: Dict[str, Tuple[float, datetime]] = {}

        self.logger = logging.getLogger(__name__)

    def extract_features(self, task: QuantumTask) -> TaskFeatures:
        """Extract ML features from quantum task."""

        # Calculate temporal features
        now = datetime.now(timezone.utc)
        days_to_deadline = 365.0  # Default far future

        if task.deadline:
            days_to_deadline = max(0.0, (task.deadline - now).total_seconds() / 86400)

        hour_of_day = now.hour
        day_of_week = now.weekday()

        # Calculate quantum uncertainty (simulated)
        quantum_uncertainty = len(task.entangled_tasks) * 0.1 + random.uniform(0.0, 0.2)

        # Extract resource requirements
        cpu_req = task.resources_required.get('cpu', 1.0)
        memory_req = task.resources_required.get('memory', 1.0)
        io_req = task.resources_required.get('io', 0.0)

        return TaskFeatures(
            priority_level=float(task.priority.value),
            estimated_hours=task.estimated_duration.total_seconds() / 3600,
            dependency_count=len(task.dependencies),
            resource_cpu=cpu_req,
            resource_memory=memory_req,
            resource_io=io_req,
            days_to_deadline=days_to_deadline,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            entanglement_count=len(task.entangled_tasks),
            urgency_score=task.calculate_urgency_score(),
            quantum_uncertainty=quantum_uncertainty
        )

    def predict_completion_time(self, task: QuantumTask) -> Tuple[float, float]:
        """Predict task completion time with confidence interval."""

        # Check cache first
        cache_key = f"{task.id}_{task.state.value}"
        if cache_key in self.prediction_cache:
            cached_prediction, cache_time = self.prediction_cache[cache_key]
            # Use cache if less than 5 minutes old
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < 300:
                return cached_prediction, 0.9  # High confidence for cached results

        # Extract features
        features = self.extract_features(task)
        feature_vector = features.to_vector()

        # Make prediction using neural network approximation
        prediction = self._forward_pass(feature_vector)

        # Apply quantum uncertainty
        base_uncertainty = features.quantum_uncertainty
        model_uncertainty = 0.1  # Base model uncertainty
        total_uncertainty = math.sqrt(base_uncertainty**2 + model_uncertainty**2)

        # Confidence based on training data similarity
        confidence = self._calculate_prediction_confidence(feature_vector)

        # Cache prediction
        self.prediction_cache[cache_key] = (prediction, datetime.now(timezone.utc))

        # Cleanup old cache entries
        self._cleanup_prediction_cache()

        return prediction, confidence

    def _forward_pass(self, features: np.ndarray) -> float:
        """Forward pass through quantum neural network."""

        # Linear transformation
        linear_output = np.dot(features, self.weights) + self.bias

        # Apply quantum activation function (modified sigmoid)
        quantum_activation = self._quantum_sigmoid(linear_output)

        # Scale to reasonable time range (0.1 to 10 hours)
        scaled_output = 0.1 + (quantum_activation * 9.9)

        return scaled_output

    def _quantum_sigmoid(self, x: float) -> float:
        """Quantum-enhanced sigmoid activation function."""

        # Standard sigmoid with quantum uncertainty
        base_sigmoid = 1.0 / (1.0 + math.exp(-x))

        # Add quantum fluctuations
        quantum_noise = random.uniform(-0.05, 0.05)

        return max(0.0, min(1.0, base_sigmoid + quantum_noise))

    def train_on_completion(
        self,
        task: QuantumTask,
        actual_completion_time: float,
        success: bool = True
    ):
        """Train model based on actual task completion."""

        features = self.extract_features(task)
        feature_vector = features.to_vector()

        # Add to training data
        self.training_features.append(feature_vector)
        self.training_targets.append(actual_completion_time)

        # Perform online learning update
        prediction = self._forward_pass(feature_vector)
        error = actual_completion_time - prediction

        # Gradient descent update with quantum regularization
        gradient = error * feature_vector
        quantum_regularization = self.regularization * self.weights

        self.weights += self.learning_rate * (gradient - quantum_regularization)
        self.bias += self.learning_rate * error

        # Track performance
        training_loss = error**2
        self.training_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loss": training_loss,
            "prediction": prediction,
            "actual": actual_completion_time,
            "task_id": task.id,
            "success": success
        })

        # Limit training history size
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-1000:]

        self.logger.debug(f"Model updated: loss={training_loss:.4f}, prediction={prediction:.2f}h, actual={actual_completion_time:.2f}h")

    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in prediction based on training data similarity."""

        if not self.training_features:
            return 0.5  # Low confidence with no training data

        # Find most similar training examples
        similarities = []
        for training_features in self.training_features[-50:]:  # Use recent 50 examples
            similarity = self._cosine_similarity(features, training_features)
            similarities.append(similarity)

        # Confidence based on maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        confidence = 0.5 + (max_similarity * 0.5)  # Scale to 0.5-1.0 range

        return confidence

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _cleanup_prediction_cache(self):
        """Remove old entries from prediction cache."""

        if len(self.prediction_cache) <= 100:
            return

        current_time = datetime.now(timezone.utc)

        # Remove entries older than 1 hour
        keys_to_remove = []
        for key, (_, cache_time) in self.prediction_cache.items():
            if (current_time - cache_time).total_seconds() > 3600:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.prediction_cache[key]

    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""

        if not self.training_history:
            return {"status": "no_training_data"}

        recent_history = self.training_history[-100:]  # Last 100 predictions

        # Calculate metrics
        losses = [entry["loss"] for entry in recent_history]
        predictions = [entry["prediction"] for entry in recent_history]
        actuals = [entry["actual"] for entry in recent_history]

        mean_loss = sum(losses) / len(losses)

        # Mean Absolute Error
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)

        # R-squared approximation
        actual_mean = sum(actuals) / len(actuals)
        ss_res = sum((a - p)**2 for p, a in zip(predictions, actuals))
        ss_tot = sum((a - actual_mean)**2 for a in actuals)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "training_samples": len(self.training_features),
            "recent_predictions": len(recent_history),
            "mean_squared_error": mean_loss,
            "mean_absolute_error": mae,
            "r_squared": r_squared,
            "model_weights_norm": np.linalg.norm(self.weights),
            "cache_size": len(self.prediction_cache)
        }

    def save_model(self, model_path: str):
        """Save trained model to disk."""

        model_data = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "training_history": self.training_history[-100:],  # Save recent history
            "feature_names": TaskFeatures.feature_names(),
            "model_version": "1.0.0",
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)

        self.logger.info(f"Quantum ML model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load trained model from disk."""

        if not Path(model_path).exists():
            self.logger.warning(f"Model file not found: {model_path}")
            return

        with open(model_path) as f:
            model_data = json.load(f)

        self.weights = np.array(model_data["weights"])
        self.bias = model_data["bias"]
        self.learning_rate = model_data.get("learning_rate", self.learning_rate)
        self.regularization = model_data.get("regularization", self.regularization)
        self.training_history = model_data.get("training_history", [])

        self.logger.info(f"Quantum ML model loaded from {model_path}")


class QuantumResourcePredictor:
    """Predicts optimal resource allocation using quantum ML."""

    def __init__(self, history_size: int = 1000):
        """Initialize quantum resource predictor."""
        self.history_size = history_size

        # Resource usage history
        self.resource_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.allocation_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Quantum learning parameters
        self.quantum_learning_rate = 0.05
        self.adaptation_weights = np.array([0.3, 0.3, 0.2, 0.2])  # [recent, trend, quantum, pattern]

        self.logger = logging.getLogger(__name__)

    def record_resource_usage(
        self,
        resource_name: str,
        usage_data: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """Record resource usage for learning."""

        timestamp = timestamp or datetime.now(timezone.utc)

        usage_record = {
            "timestamp": timestamp,
            "utilization": usage_data.get("utilization", 0.0),
            "efficiency": usage_data.get("efficiency", 1.0),
            "quantum_speedup": usage_data.get("quantum_speedup", 1.0),
            "coherence_time": usage_data.get("coherence_time", 30.0)
        }

        self.resource_history[resource_name].append(usage_record)

        # Update allocation patterns
        self._update_allocation_patterns(resource_name, usage_record)

    def predict_optimal_allocation(
        self,
        tasks: List[QuantumTask],
        current_resources: List[ResourceQuantum],
        prediction_horizon: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Predict optimal resource allocation for upcoming tasks."""

        prediction_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "horizon_hours": prediction_horizon.total_seconds() / 3600,
            "resource_predictions": {},
            "allocation_recommendations": [],
            "quantum_insights": {},
            "confidence_score": 0.0
        }

        total_confidence = 0.0

        for resource in current_resources:
            resource_prediction = self._predict_single_resource(
                resource, tasks, prediction_horizon
            )

            prediction_result["resource_predictions"][resource.name] = resource_prediction
            total_confidence += resource_prediction["confidence"]

        # Calculate average confidence
        prediction_result["confidence_score"] = total_confidence / len(current_resources) if current_resources else 0.0

        # Generate allocation recommendations
        prediction_result["allocation_recommendations"] = self._generate_allocation_recommendations(
            prediction_result["resource_predictions"], tasks
        )

        # Quantum insights
        prediction_result["quantum_insights"] = self._analyze_quantum_patterns(
            prediction_result["resource_predictions"]
        )

        return prediction_result

    def _predict_single_resource(
        self,
        resource: ResourceQuantum,
        tasks: List[QuantumTask],
        horizon: timedelta
    ) -> Dict[str, Any]:
        """Predict usage for a single resource."""

        history = list(self.resource_history[resource.name])

        if not history:
            # No history - use conservative estimates
            return {
                "predicted_utilization": 0.5,
                "predicted_efficiency": resource.quantum_efficiency,
                "confidence": 0.3,
                "trend": "unknown",
                "quantum_forecast": {
                    "coherence_stability": 0.8,
                    "expected_speedup": resource.quantum_efficiency
                }
            }

        # Recent usage trend
        recent_utilizations = [record["utilization"] for record in history[-10:]]
        trend_direction = self._calculate_trend(recent_utilizations)

        # Quantum-weighted prediction
        weights = self._calculate_quantum_weights(len(history))
        weighted_utilization = sum(
            record["utilization"] * weight
            for record, weight in zip(history, weights)
        )

        # Factor in upcoming task load
        upcoming_load = self._calculate_upcoming_load(resource, tasks, horizon)

        # Combine predictions with quantum superposition
        base_prediction = weighted_utilization
        load_adjusted = base_prediction + upcoming_load * 0.5

        # Apply quantum uncertainty
        quantum_uncertainty = random.uniform(0.95, 1.05)
        final_prediction = load_adjusted * quantum_uncertainty

        # Calculate confidence based on data quality
        confidence = self._calculate_prediction_confidence_resource(history, upcoming_load)

        return {
            "predicted_utilization": min(final_prediction, 1.0),
            "predicted_efficiency": self._predict_efficiency(resource, history),
            "confidence": confidence,
            "trend": trend_direction,
            "quantum_forecast": {
                "coherence_stability": self._predict_coherence_stability(history),
                "expected_speedup": self._predict_quantum_speedup(resource, upcoming_load)
            },
            "contributing_factors": {
                "historical_weight": 0.6,
                "upcoming_load_weight": 0.3,
                "quantum_uncertainty_weight": 0.1
            }
        }

    def _calculate_quantum_weights(self, history_length: int) -> List[float]:
        """Calculate quantum-inspired weights for historical data."""

        # Exponential decay with quantum oscillations
        weights = []
        for i in range(history_length):
            # More recent data gets higher weight
            recency_weight = math.exp(-i * 0.1)

            # Add quantum oscillations to prevent overfitting to recent data
            quantum_oscillation = 1.0 + 0.1 * math.sin(i * 0.5)

            weight = recency_weight * quantum_oscillation
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        return weights

    def _calculate_upcoming_load(
        self,
        resource: ResourceQuantum,
        tasks: List[QuantumTask],
        horizon: timedelta
    ) -> float:
        """Calculate predicted resource load from upcoming tasks."""

        total_load = 0.0
        horizon_hours = horizon.total_seconds() / 3600

        for task in tasks:
            if task.state in [TaskState.SUPERPOSITION, TaskState.COLLAPSED]:
                # Extract resource requirement for this resource type
                resource_type = resource.name.split('_')[1] if '_' in resource.name else 'cpu'
                task_requirement = task.resources_required.get(resource_type, 0.0)

                # Weight by task priority and urgency
                priority_weight = (4 - task.priority.value) / 4.0
                urgency_weight = task.calculate_urgency_score()

                # Estimate probability of task running in horizon
                if task.deadline:
                    deadline_hours = (task.deadline - datetime.now(timezone.utc)).total_seconds() / 3600
                    deadline_probability = 1.0 if deadline_hours <= horizon_hours else 0.3
                else:
                    deadline_probability = 0.7  # Default probability

                weighted_load = task_requirement * priority_weight * urgency_weight * deadline_probability
                total_load += weighted_load

        # Normalize by resource capacity and horizon
        normalized_load = total_load / (resource.total_capacity * horizon_hours / 24)

        return min(normalized_load, 1.0)

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values."""

        if len(values) < 3:
            return "stable"

        # Simple linear regression
        n = len(values)
        x_values = list(range(n))

        # Calculate slope
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def _predict_efficiency(
        self,
        resource: ResourceQuantum,
        history: List[Dict[str, Any]]
    ) -> float:
        """Predict quantum efficiency for resource."""

        if not history:
            return resource.quantum_efficiency

        recent_efficiencies = [record["efficiency"] for record in history[-5:]]
        avg_efficiency = sum(recent_efficiencies) / len(recent_efficiencies)

        # Apply quantum enhancement based on usage patterns
        usage_stability = 1.0 - np.std([record["utilization"] for record in history[-10:]])
        quantum_enhancement = 1.0 + (usage_stability * 0.2)

        return avg_efficiency * quantum_enhancement

    def _predict_coherence_stability(self, history: List[Dict[str, Any]]) -> float:
        """Predict quantum coherence stability."""

        if not history:
            return 0.8  # Default stability

        coherence_times = [record["coherence_time"] for record in history[-10:]]
        avg_coherence = sum(coherence_times) / len(coherence_times)

        # Stability based on coherence time variance
        coherence_variance = np.var(coherence_times)
        stability = 1.0 / (1.0 + coherence_variance * 0.1)

        return min(stability, 1.0)

    def _predict_quantum_speedup(self, resource: ResourceQuantum, upcoming_load: float) -> float:
        """Predict quantum speedup factor."""

        base_speedup = resource.quantum_efficiency

        # Speedup decreases with higher load (quantum decoherence)
        load_factor = 1.0 - (upcoming_load * 0.2)

        # Add quantum fluctuations
        quantum_fluctuation = random.uniform(0.95, 1.05)

        predicted_speedup = base_speedup * load_factor * quantum_fluctuation

        return max(1.0, predicted_speedup)  # Speedup is always >= 1.0

    def _calculate_prediction_confidence_resource(
        self,
        history: List[Dict[str, Any]],
        upcoming_load: float
    ) -> float:
        """Calculate confidence in resource prediction."""

        # Base confidence from data quantity
        data_confidence = min(len(history) / 50.0, 1.0)  # Full confidence with 50+ samples

        # Confidence decreases with high upcoming load (more uncertainty)
        load_confidence = 1.0 - (upcoming_load * 0.3)

        # Historical variance penalty
        if len(history) >= 5:
            utilizations = [record["utilization"] for record in history[-10:]]
            variance_penalty = np.var(utilizations) * 2.0
        else:
            variance_penalty = 0.5  # High penalty for low data

        confidence = data_confidence * load_confidence * (1.0 - variance_penalty)

        return max(0.1, min(confidence, 1.0))

    def _update_allocation_patterns(self, resource_name: str, usage_record: Dict[str, Any]):
        """Update allocation patterns for machine learning."""

        patterns = self.allocation_patterns[resource_name]

        # Extract pattern features
        pattern = {
            "timestamp": usage_record["timestamp"].isoformat(),
            "hour_of_day": usage_record["timestamp"].hour,
            "day_of_week": usage_record["timestamp"].weekday(),
            "utilization": usage_record["utilization"],
            "efficiency": usage_record["efficiency"],
            "quantum_coherence": usage_record.get("coherence_time", 30.0) / 60.0  # Normalize to hours
        }

        patterns.append(pattern)

        # Limit pattern history
        if len(patterns) > self.history_size:
            patterns[:] = patterns[-self.history_size:]

    def _generate_allocation_recommendations(
        self,
        resource_predictions: Dict[str, Dict[str, Any]],
        tasks: List[QuantumTask]
    ) -> List[str]:
        """Generate intelligent allocation recommendations."""

        recommendations = []

        # Analyze predicted bottlenecks
        bottleneck_resources = []
        for resource_name, prediction in resource_predictions.items():
            if prediction["predicted_utilization"] > 0.85:
                bottleneck_resources.append((resource_name, prediction["predicted_utilization"]))

        if bottleneck_resources:
            bottleneck_resources.sort(key=lambda x: x[1], reverse=True)
            worst_bottleneck = bottleneck_resources[0]

            recommendations.append(
                f"🚨 Critical: {worst_bottleneck[0]} predicted to reach {worst_bottleneck[1]:.1%} utilization"
            )
            recommendations.append(
                f"💡 Recommend scaling {worst_bottleneck[0]} by {math.ceil((worst_bottleneck[1] - 0.8) * 10)} units"
            )

        # Quantum efficiency recommendations
        low_efficiency_resources = [
            (name, pred) for name, pred in resource_predictions.items()
            if pred["quantum_forecast"]["expected_speedup"] < 1.2
        ]

        if low_efficiency_resources:
            recommendations.append(
                "⚡ Consider quantum coherence optimization for improved speedup"
            )

        # Load balancing recommendations
        utilizations = [pred["predicted_utilization"] for pred in resource_predictions.values()]
        if len(utilizations) > 1:
            utilization_variance = np.var(utilizations)
            if utilization_variance > 0.1:
                recommendations.append(
                    "⚖️ Implement quantum load balancing to reduce utilization variance"
                )

        # Task scheduling recommendations based on predictions
        high_priority_tasks = [t for t in tasks if t.priority in [TaskPriority.GROUND_STATE, TaskPriority.EXCITED_1]]
        if len(high_priority_tasks) > 5:
            recommendations.append(
                f"📋 {len(high_priority_tasks)} high-priority tasks detected - consider priority-based resource reservation"
            )

        return recommendations

    def _analyze_quantum_patterns(self, resource_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quantum patterns in resource predictions."""

        insights = {
            "coherence_trends": {},
            "entanglement_opportunities": [],
            "quantum_optimization_potential": 0.0
        }

        # Analyze coherence trends
        for resource_name, prediction in resource_predictions.items():
            coherence_stability = prediction["quantum_forecast"]["coherence_stability"]
            insights["coherence_trends"][resource_name] = {
                "stability": coherence_stability,
                "trend": "stable" if coherence_stability > 0.8 else "degrading"
            }

        # Identify entanglement opportunities
        resource_names = list(resource_predictions.keys())
        for i, res1 in enumerate(resource_names):
            for res2 in resource_names[i+1:]:
                pred1 = resource_predictions[res1]
                pred2 = resource_predictions[res2]

                # Check for complementary usage patterns
                util_diff = abs(pred1["predicted_utilization"] - pred2["predicted_utilization"])
                if util_diff > 0.3:  # Significant difference
                    insights["entanglement_opportunities"].append({
                        "high_util_resource": res1 if pred1["predicted_utilization"] > pred2["predicted_utilization"] else res2,
                        "low_util_resource": res2 if pred1["predicted_utilization"] > pred2["predicted_utilization"] else res1,
                        "load_balance_potential": util_diff
                    })

        # Calculate quantum optimization potential
        avg_speedup = np.mean([
            pred["quantum_forecast"]["expected_speedup"]
            for pred in resource_predictions.values()
        ])
        optimization_potential = max(0.0, (avg_speedup - 1.0) * 0.5)
        insights["quantum_optimization_potential"] = optimization_potential

        return insights


class QuantumAdaptiveLearning:
    """Adaptive learning system that improves quantum planning over time."""

    def __init__(self):
        """Initialize quantum adaptive learning system."""
        self.task_predictor = QuantumTaskPredictor()
        self.resource_predictor = QuantumResourcePredictor()

        # Learning metrics
        self.learning_sessions: List[Dict[str, Any]] = []
        self.adaptation_rate = 0.1

        self.logger = logging.getLogger(__name__)

    async def learn_from_execution(
        self,
        executed_tasks: List[Dict[str, Any]],
        resource_usage: Dict[str, Dict[str, float]],
        execution_metadata: Dict[str, Any]
    ):
        """Learn from task execution results."""

        self.logger.info(f"Learning from execution of {len(executed_tasks)} tasks")

        learning_session = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tasks_analyzed": len(executed_tasks),
            "performance_improvements": []
        }

        # Learn from individual tasks
        for task_data in executed_tasks:
            await self._learn_from_task_execution(task_data, learning_session)

        # Learn from resource usage patterns
        for resource_name, usage_data in resource_usage.items():
            self.resource_predictor.record_resource_usage(resource_name, usage_data)
            learning_session["performance_improvements"].append(
                f"Updated {resource_name} usage patterns"
            )

        # Adapt prediction models
        adaptation_improvements = await self._adapt_prediction_models(execution_metadata)
        learning_session["performance_improvements"].extend(adaptation_improvements)

        self.learning_sessions.append(learning_session)

        # Limit session history
        if len(self.learning_sessions) > 100:
            self.learning_sessions = self.learning_sessions[-100:]

        self.logger.info(f"Learning session completed with {len(learning_session['performance_improvements'])} improvements")

    async def _learn_from_task_execution(
        self,
        task_data: Dict[str, Any],
        learning_session: Dict[str, Any]
    ):
        """Learn from individual task execution."""

        # Reconstruct task for feature extraction
        task = self._reconstruct_task_from_data(task_data)

        if not task:
            return

        # Extract execution metrics
        actual_duration = task_data.get("execution_time_hours", 1.0)
        success = task_data.get("success", True)

        # Train prediction model
        self.task_predictor.train_on_completion(task, actual_duration, success)

        learning_session["performance_improvements"].append(
            f"Updated prediction model with task {task.id} (actual: {actual_duration:.2f}h)"
        )

    def _reconstruct_task_from_data(self, task_data: Dict[str, Any]) -> Optional[QuantumTask]:
        """Reconstruct QuantumTask from execution data."""

        try:
            # Parse priority
            priority_name = task_data.get("priority", "EXCITED_2")
            priority = TaskPriority[priority_name] if hasattr(TaskPriority, priority_name) else TaskPriority.EXCITED_2

            # Parse deadline
            deadline = None
            if task_data.get("deadline"):
                deadline = datetime.fromisoformat(task_data["deadline"])

            # Create task
            task = QuantumTask(
                id=task_data["task_id"],
                name=task_data.get("name", "Unknown Task"),
                description=task_data.get("description", ""),
                priority=priority,
                estimated_duration=timedelta(hours=task_data.get("estimated_duration_hours", 1.0)),
                dependencies=set(task_data.get("dependencies", [])),
                resources_required=task_data.get("resources_required", {"cpu": 1.0}),
                deadline=deadline
            )

            # Set entangled tasks
            task.entangled_tasks = set(task_data.get("entangled_tasks", []))

            return task

        except Exception as e:
            self.logger.error(f"Failed to reconstruct task from data: {e}")
            return None

    async def _adapt_prediction_models(self, execution_metadata: Dict[str, Any]) -> List[str]:
        """Adapt prediction models based on execution results."""

        improvements = []

        # Analyze prediction accuracy
        prediction_accuracy = execution_metadata.get("prediction_accuracy", 0.8)

        if prediction_accuracy < 0.7:
            # Low accuracy - increase learning rate
            old_lr = self.task_predictor.learning_rate
            self.task_predictor.learning_rate *= 1.2
            improvements.append(
                f"Increased learning rate from {old_lr:.4f} to {self.task_predictor.learning_rate:.4f}"
            )

        elif prediction_accuracy > 0.95:
            # Very high accuracy - might be overfitting, reduce learning rate
            old_lr = self.task_predictor.learning_rate
            self.task_predictor.learning_rate *= 0.9
            improvements.append(
                f"Reduced learning rate from {old_lr:.4f} to {self.task_predictor.learning_rate:.4f}"
            )

        # Adapt quantum parameters based on execution efficiency
        execution_efficiency = execution_metadata.get("quantum_efficiency", 1.0)

        if execution_efficiency > 1.5:
            # High quantum efficiency - increase quantum uncertainty sampling
            self.adaptation_rate = min(self.adaptation_rate * 1.1, 0.5)
            improvements.append("Increased quantum adaptation rate due to high efficiency")

        return improvements

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""

        return {
            "learning_sessions": len(self.learning_sessions),
            "total_tasks_learned": sum(
                session.get("tasks_analyzed", 0) for session in self.learning_sessions
            ),
            "model_performance": self.task_predictor.get_model_performance(),
            "resource_prediction_confidence": self._calculate_avg_resource_confidence(),
            "adaptation_rate": self.adaptation_rate,
            "quantum_learning_insights": self._generate_learning_insights()
        }

    def _calculate_avg_resource_confidence(self) -> float:
        """Calculate average resource prediction confidence."""

        total_confidence = 0.0
        total_resources = 0

        for resource_patterns in self.resource_predictor.allocation_patterns.values():
            if resource_patterns:
                # Simulate confidence calculation
                confidence = min(len(resource_patterns) / 20.0, 1.0)
                total_confidence += confidence
                total_resources += 1

        return total_confidence / total_resources if total_resources > 0 else 0.5

    def _generate_learning_insights(self) -> List[str]:
        """Generate insights from quantum learning process."""

        insights = []

        # Task prediction insights
        model_performance = self.task_predictor.get_model_performance()
        if model_performance.get("r_squared", 0) > 0.8:
            insights.append("🧠 Task prediction model achieving high accuracy")

        # Resource pattern insights
        pattern_count = sum(
            len(patterns) for patterns in self.resource_predictor.allocation_patterns.values()
        )
        if pattern_count > 100:
            insights.append("📊 Rich resource usage patterns enabling advanced prediction")

        # Quantum adaptation insights
        if self.adaptation_rate > 0.2:
            insights.append("⚡ High quantum adaptation rate - system learning rapidly")

        # Learning velocity insights
        recent_sessions = self.learning_sessions[-10:]
        if len(recent_sessions) >= 5:
            avg_tasks_per_session = sum(
                session.get("tasks_analyzed", 0) for session in recent_sessions
            ) / len(recent_sessions)

            if avg_tasks_per_session > 10:
                insights.append("🚀 High learning velocity from frequent task execution")

        return insights


# Factory functions
def create_quantum_task_predictor() -> QuantumTaskPredictor:
    """Create quantum task predictor with optimal configuration."""
    return QuantumTaskPredictor(learning_rate=0.01, regularization=0.001)


def create_quantum_resource_predictor() -> QuantumResourcePredictor:
    """Create quantum resource predictor."""
    return QuantumResourcePredictor()


def create_adaptive_learning_system() -> QuantumAdaptiveLearning:
    """Create quantum adaptive learning system."""
    return QuantumAdaptiveLearning()


# Demo function
async def demo_quantum_ml():
    """Demonstrate quantum machine learning capabilities."""

    from .quantum_planner import create_quantum_planner

    print("🧠 Starting Quantum ML Demo...")

    # Create learning system
    learning_system = create_adaptive_learning_system()

    # Create sample tasks for prediction
    planner = create_quantum_planner()

    sample_tasks = [
        ("ml_training", "Train ML Model", TaskPriority.GROUND_STATE, 4.0),
        ("data_processing", "Process Training Data", TaskPriority.EXCITED_1, 2.0),
        ("model_validation", "Validate Model Performance", TaskPriority.EXCITED_2, 1.5),
        ("deployment", "Deploy to Production", TaskPriority.EXCITED_1, 3.0)
    ]

    for task_id, name, priority, hours in sample_tasks:
        planner.add_task(
            task_id=task_id,
            name=name,
            description=f"Task: {name}",
            priority=priority,
            estimated_duration=timedelta(hours=hours),
            resources_required={"cpu": random.uniform(1.0, 3.0), "memory": random.uniform(1.0, 4.0)}
        )

    # Test task prediction
    print("🔮 Testing task prediction...")
    for task in planner.tasks.values():
        predicted_time, confidence = learning_system.task_predictor.predict_completion_time(task)
        print(f"   Task {task.name}: {predicted_time:.2f}h (confidence: {confidence:.2f})")

    # Test resource prediction
    print("📊 Testing resource prediction...")
    resource_prediction = learning_system.resource_predictor.predict_optimal_allocation(
        list(planner.tasks.values()),
        planner.resources
    )

    print(f"   Overall confidence: {resource_prediction['confidence_score']:.2f}")
    print(f"   Recommendations: {len(resource_prediction['allocation_recommendations'])}")

    # Simulate learning from execution
    print("📚 Simulating learning from execution...")

    executed_tasks = []
    for task in planner.tasks.values():
        executed_tasks.append({
            "task_id": task.id,
            "name": task.name,
            "priority": task.priority.name,
            "estimated_duration_hours": task.estimated_duration.total_seconds() / 3600,
            "execution_time_hours": random.uniform(0.8, 1.2) * task.estimated_duration.total_seconds() / 3600,
            "success": random.random() > 0.1,  # 90% success rate
            "dependencies": list(task.dependencies),
            "resources_required": task.resources_required
        })

    resource_usage = {
        resource.name: {
            "utilization": random.uniform(0.4, 0.8),
            "efficiency": random.uniform(1.1, 2.0),
            "quantum_speedup": random.uniform(1.2, 2.5)
        }
        for resource in planner.resources
    }

    execution_metadata = {
        "prediction_accuracy": 0.85,
        "quantum_efficiency": 1.8
    }

    await learning_system.learn_from_execution(executed_tasks, resource_usage, execution_metadata)

    # Get learning statistics
    stats = learning_system.get_learning_statistics()
    print("✅ Learning completed:")
    print(f"   📈 Model performance R²: {stats['model_performance'].get('r_squared', 0):.3f}")
    print(f"   🎯 Resource confidence: {stats['resource_prediction_confidence']:.2f}")
    print(f"   🧠 Learning insights: {len(stats['quantum_learning_insights'])}")

    return stats


if __name__ == "__main__":
    asyncio.run(demo_quantum_ml())
