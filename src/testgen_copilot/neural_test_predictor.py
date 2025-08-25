"""
Neural Network Test Prediction System for TestGen Copilot
Implements advanced ML models for intelligent test generation and optimization
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TestPredictionFeatures:
    """Features extracted from code for test prediction."""
    code_complexity: float
    function_count: int
    line_count: int
    cyclomatic_complexity: float
    dependency_count: int
    error_handling_coverage: float
    function_signatures: List[str]
    code_patterns: List[str]
    historical_bug_density: float
    test_execution_time: float
    code_change_frequency: float
    

@dataclass
class TestCase:
    """Represents a test case with metadata."""
    id: str
    function_name: str
    test_type: str  # unit, integration, edge_case, error_handling
    complexity_score: float
    execution_time: float
    success_rate: float
    bug_detection_rate: float
    code_coverage: float
    maintenance_effort: float
    business_impact: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class PredictionResult:
    """Result of test prediction."""
    suggested_test_types: List[str]
    predicted_complexity: float
    estimated_execution_time: float
    expected_coverage_improvement: float
    risk_areas: List[str]
    confidence_score: float
    recommended_test_count: int
    priority_score: float


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    training_time: float
    prediction_time: float
    model_version: str
    last_updated: datetime = field(default_factory=datetime.now)


class NeuralTestPredictor:
    """
    Advanced neural network system for predicting optimal test strategies.
    Uses multiple ML models for different aspects of test generation.
    """
    
    def __init__(self, model_dir: Path = Path("ml_models")):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model components
        self.test_type_classifier = None
        self.complexity_predictor = None
        self.execution_time_predictor = None
        self.coverage_predictor = None
        self.bug_risk_predictor = None
        
        # Feature processors
        self.feature_scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        
        # Training data
        self.training_data: List[Tuple[TestPredictionFeatures, TestCase]] = []
        self.historical_performance: List[ModelPerformance] = []
        
        # Configuration
        self.model_config = {
            "test_type_classifier": {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 1000
            },
            "complexity_predictor": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6
            },
            "execution_time_predictor": {
                "hidden_layer_sizes": (80, 40),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 800
            }
        }
        
        self._load_models()
        
    def train_models(self, training_data: List[Tuple[TestPredictionFeatures, TestCase]]) -> Dict[str, ModelPerformance]:
        """
        Train all ML models with provided training data.
        """
        logger.info(f"Training models with {len(training_data)} samples")
        
        if len(training_data) < 50:
            logger.warning("Insufficient training data. Generating synthetic data.")
            training_data.extend(self._generate_synthetic_training_data(200))
            
        self.training_data = training_data
        
        # Prepare features and targets
        X, y_dict = self._prepare_training_data(training_data)
        
        performance_results = {}
        
        # Train test type classifier
        performance_results["test_type_classifier"] = self._train_test_type_classifier(
            X, y_dict["test_types"]
        )
        
        # Train complexity predictor
        performance_results["complexity_predictor"] = self._train_complexity_predictor(
            X, y_dict["complexity_scores"]
        )
        
        # Train execution time predictor
        performance_results["execution_time_predictor"] = self._train_execution_time_predictor(
            X, y_dict["execution_times"]
        )
        
        # Train coverage predictor
        performance_results["coverage_predictor"] = self._train_coverage_predictor(
            X, y_dict["coverage_scores"]
        )
        
        # Train bug risk predictor
        performance_results["bug_risk_predictor"] = self._train_bug_risk_predictor(
            X, y_dict["bug_detection_rates"]
        )
        
        # Save models
        self._save_models()
        
        # Update performance history
        avg_performance = self._calculate_average_performance(performance_results)
        self.historical_performance.append(avg_performance)
        
        logger.info("Model training completed successfully")
        return performance_results
        
    def predict_optimal_tests(
        self,
        code_features: TestPredictionFeatures
    ) -> PredictionResult:
        """
        Predict optimal test strategy for given code features.
        """
        logger.debug(f"Predicting tests for function: {code_features.function_signatures}")
        
        if not self._models_trained():
            logger.warning("Models not trained. Using default predictions.")
            return self._get_default_prediction()
            
        # Prepare features
        X = self._prepare_features([code_features])
        
        # Get predictions from all models
        predicted_test_types = self._predict_test_types(X)
        predicted_complexity = self._predict_complexity(X)
        predicted_execution_time = self._predict_execution_time(X)
        predicted_coverage = self._predict_coverage_improvement(X)
        predicted_bug_risk = self._predict_bug_risk(X)
        
        # Combine predictions into result
        result = PredictionResult(
            suggested_test_types=predicted_test_types[0],
            predicted_complexity=float(predicted_complexity[0]),
            estimated_execution_time=float(predicted_execution_time[0]),
            expected_coverage_improvement=float(predicted_coverage[0]),
            risk_areas=self._identify_risk_areas(code_features, predicted_bug_risk[0]),
            confidence_score=self._calculate_prediction_confidence(X),
            recommended_test_count=self._calculate_recommended_test_count(predicted_complexity[0]),
            priority_score=self._calculate_priority_score(predicted_bug_risk[0], predicted_coverage[0])
        )
        
        return result
        
    def update_model_with_feedback(
        self,
        prediction: PredictionResult,
        actual_results: Dict[str, Any]
    ) -> None:
        """
        Update models based on feedback from actual test results.
        """
        logger.info("Updating models with feedback data")
        
        # Create feedback training sample
        feedback_sample = self._create_feedback_sample(prediction, actual_results)
        
        # Add to training data
        self.training_data.append(feedback_sample)
        
        # Incremental learning (simplified - in practice would use online learning)
        if len(self.training_data) % 50 == 0:  # Retrain every 50 feedback samples
            logger.info("Retraining models with accumulated feedback")
            self.train_models(self.training_data[-200:])  # Use recent data
            
    def analyze_prediction_accuracy(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze prediction accuracy against actual test results.
        """
        logger.info(f"Analyzing prediction accuracy for {len(test_results)} results")
        
        accuracy_metrics = {
            "test_type_accuracy": 0.0,
            "complexity_mse": 0.0,
            "execution_time_mse": 0.0,
            "coverage_mae": 0.0,
            "overall_accuracy": 0.0
        }
        
        if not test_results:
            return accuracy_metrics
            
        # Calculate accuracy metrics
        test_type_correct = sum(
            1 for result in test_results 
            if result.get("predicted_type") == result.get("actual_type")
        )
        accuracy_metrics["test_type_accuracy"] = test_type_correct / len(test_results)
        
        # Complexity MSE
        complexity_errors = [
            (result.get("predicted_complexity", 0) - result.get("actual_complexity", 0)) ** 2
            for result in test_results
            if "predicted_complexity" in result and "actual_complexity" in result
        ]
        if complexity_errors:
            accuracy_metrics["complexity_mse"] = sum(complexity_errors) / len(complexity_errors)
            
        # Execution time MSE
        time_errors = [
            (result.get("predicted_time", 0) - result.get("actual_time", 0)) ** 2
            for result in test_results
            if "predicted_time" in result and "actual_time" in result
        ]
        if time_errors:
            accuracy_metrics["execution_time_mse"] = sum(time_errors) / len(time_errors)
            
        # Coverage MAE
        coverage_errors = [
            abs(result.get("predicted_coverage", 0) - result.get("actual_coverage", 0))
            for result in test_results
            if "predicted_coverage" in result and "actual_coverage" in result
        ]
        if coverage_errors:
            accuracy_metrics["coverage_mae"] = sum(coverage_errors) / len(coverage_errors)
            
        # Overall accuracy (weighted combination)
        accuracy_metrics["overall_accuracy"] = (
            0.4 * accuracy_metrics["test_type_accuracy"] +
            0.2 * (1.0 - min(accuracy_metrics["complexity_mse"], 1.0)) +
            0.2 * (1.0 - min(accuracy_metrics["execution_time_mse"], 1.0)) +
            0.2 * (1.0 - min(accuracy_metrics["coverage_mae"], 1.0))
        )
        
        return accuracy_metrics
        
    def get_model_insights(self) -> Dict[str, Any]:
        """
        Get insights into model behavior and feature importance.
        """
        logger.info("Generating model insights")
        
        insights = {
            "feature_importance": {},
            "model_performance": {},
            "prediction_patterns": {},
            "training_history": []
        }
        
        # Feature importance (for tree-based models)
        if hasattr(self.complexity_predictor, 'feature_importances_'):
            insights["feature_importance"]["complexity"] = self.complexity_predictor.feature_importances_.tolist()
            
        # Model performance history
        insights["model_performance"] = {
            "latest_accuracy": self.historical_performance[-1].accuracy if self.historical_performance else 0.0,
            "performance_trend": self._calculate_performance_trend(),
            "training_count": len(self.historical_performance)
        }
        
        # Prediction patterns
        insights["prediction_patterns"] = self._analyze_prediction_patterns()
        
        # Training history
        insights["training_history"] = [
            {
                "timestamp": perf.last_updated.isoformat(),
                "accuracy": perf.accuracy,
                "f1_score": perf.f1_score
            }
            for perf in self.historical_performance[-10:]  # Last 10 training sessions
        ]
        
        return insights
        
    def _prepare_training_data(
        self, 
        training_data: List[Tuple[TestPredictionFeatures, TestCase]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare features and targets for model training.
        """
        features_list = []
        test_types = []
        complexity_scores = []
        execution_times = []
        coverage_scores = []
        bug_detection_rates = []
        
        for features, test_case in training_data:
            # Numerical features
            feature_vector = [
                features.code_complexity,
                features.function_count,
                features.line_count,
                features.cyclomatic_complexity,
                features.dependency_count,
                features.error_handling_coverage,
                features.historical_bug_density,
                features.test_execution_time,
                features.code_change_frequency
            ]
            
            features_list.append(feature_vector)
            test_types.append(test_case.test_type)
            complexity_scores.append(test_case.complexity_score)
            execution_times.append(test_case.execution_time)
            coverage_scores.append(test_case.code_coverage)
            bug_detection_rates.append(test_case.bug_detection_rate)
            
        X = np.array(features_list)
        X = self.feature_scaler.fit_transform(X)
        
        # Encode categorical targets
        if "test_type" not in self.label_encoders:
            self.label_encoders["test_type"] = LabelEncoder()
            
        y_test_types = self.label_encoders["test_type"].fit_transform(test_types)
        
        targets = {
            "test_types": y_test_types,
            "complexity_scores": np.array(complexity_scores),
            "execution_times": np.array(execution_times),
            "coverage_scores": np.array(coverage_scores),
            "bug_detection_rates": np.array(bug_detection_rates)
        }
        
        return X, targets
        
    def _train_test_type_classifier(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train test type classification model."""
        start_time = time.time()
        
        self.test_type_classifier = MLPClassifier(**self.model_config["test_type_classifier"])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.test_type_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.test_type_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(self.test_type_classifier, X, y, cv=5)
        
        training_time = time.time() - start_time
        
        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0,  # Will be measured during prediction
            model_version="1.0"
        )
        
    def _train_complexity_predictor(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train complexity prediction model."""
        start_time = time.time()
        
        self.complexity_predictor = GradientBoostingRegressor(**self.model_config["complexity_predictor"])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.complexity_predictor.fit(X_train, y_train)
        
        # Evaluate using R²
        score = self.complexity_predictor.score(X_test, y_test)
        cv_scores = cross_val_score(self.complexity_predictor, X, y, cv=5)
        
        training_time = time.time() - start_time
        
        return ModelPerformance(
            accuracy=max(0, score),  # R² can be negative
            precision=score,
            recall=score,
            f1_score=score,
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0,
            model_version="1.0"
        )
        
    def _train_execution_time_predictor(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train execution time prediction model."""
        start_time = time.time()
        
        self.execution_time_predictor = MLPRegressor(**self.model_config["execution_time_predictor"])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.execution_time_predictor.fit(X_train, y_train)
        
        score = self.execution_time_predictor.score(X_test, y_test)
        cv_scores = cross_val_score(self.execution_time_predictor, X, y, cv=5)
        
        training_time = time.time() - start_time
        
        return ModelPerformance(
            accuracy=max(0, score),
            precision=score,
            recall=score,
            f1_score=score,
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0,
            model_version="1.0"
        )
        
    def _train_coverage_predictor(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train coverage improvement prediction model."""
        start_time = time.time()
        
        self.coverage_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Convert regression to classification (coverage bins)
        y_binned = np.digitize(y, bins=[0.0, 0.5, 0.7, 0.85, 1.0])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)
        
        self.coverage_predictor.fit(X_train, y_train)
        
        accuracy = self.coverage_predictor.score(X_test, y_test)
        cv_scores = cross_val_score(self.coverage_predictor, X, y_binned, cv=5)
        
        training_time = time.time() - start_time
        
        return ModelPerformance(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1_score=accuracy,
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0,
            model_version="1.0"
        )
        
    def _train_bug_risk_predictor(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train bug risk prediction model."""
        start_time = time.time()
        
        self.bug_risk_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.bug_risk_predictor.fit(X_train, y_train)
        
        score = self.bug_risk_predictor.score(X_test, y_test)
        cv_scores = cross_val_score(self.bug_risk_predictor, X, y, cv=5)
        
        training_time = time.time() - start_time
        
        return ModelPerformance(
            accuracy=max(0, score),
            precision=score,
            recall=score,
            f1_score=score,
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0,
            model_version="1.0"
        )
        
    def _prepare_features(self, features_list: List[TestPredictionFeatures]) -> np.ndarray:
        """Prepare features for prediction."""
        feature_vectors = []
        
        for features in features_list:
            vector = [
                features.code_complexity,
                features.function_count,
                features.line_count,
                features.cyclomatic_complexity,
                features.dependency_count,
                features.error_handling_coverage,
                features.historical_bug_density,
                features.test_execution_time,
                features.code_change_frequency
            ]
            feature_vectors.append(vector)
            
        X = np.array(feature_vectors)
        return self.feature_scaler.transform(X)
        
    def _predict_test_types(self, X: np.ndarray) -> List[List[str]]:
        """Predict optimal test types."""
        if self.test_type_classifier is None:
            return [["unit", "integration"]]
            
        predictions = self.test_type_classifier.predict(X)
        
        # Convert encoded predictions back to labels
        predicted_labels = self.label_encoders["test_type"].inverse_transform(predictions)
        
        # For each prediction, suggest multiple test types based on probabilities
        results = []
        if hasattr(self.test_type_classifier, 'predict_proba'):
            probabilities = self.test_type_classifier.predict_proba(X)
            
            for i, probs in enumerate(probabilities):
                # Get top test types
                top_indices = np.argsort(probs)[-3:][::-1]  # Top 3
                top_types = [self.label_encoders["test_type"].classes_[idx] for idx in top_indices]
                results.append(top_types)
        else:
            results = [[label] for label in predicted_labels]
            
        return results
        
    def _predict_complexity(self, X: np.ndarray) -> np.ndarray:
        """Predict complexity scores."""
        if self.complexity_predictor is None:
            return np.array([0.5] * len(X))
            
        return self.complexity_predictor.predict(X)
        
    def _predict_execution_time(self, X: np.ndarray) -> np.ndarray:
        """Predict execution times."""
        if self.execution_time_predictor is None:
            return np.array([1.0] * len(X))
            
        return self.execution_time_predictor.predict(X)
        
    def _predict_coverage_improvement(self, X: np.ndarray) -> np.ndarray:
        """Predict coverage improvements."""
        if self.coverage_predictor is None:
            return np.array([0.15] * len(X))
            
        # Predict coverage bins and convert back to scores
        predictions = self.coverage_predictor.predict(X)
        
        # Map bins back to coverage scores
        bin_mapping = {0: 0.0, 1: 0.25, 2: 0.6, 3: 0.775, 4: 0.925}
        coverage_scores = [bin_mapping.get(pred, 0.15) for pred in predictions]
        
        return np.array(coverage_scores)
        
    def _predict_bug_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict bug risk scores."""
        if self.bug_risk_predictor is None:
            return np.array([0.3] * len(X))
            
        return self.bug_risk_predictor.predict(X)
        
    def _identify_risk_areas(self, features: TestPredictionFeatures, bug_risk: float) -> List[str]:
        """Identify high-risk areas based on features and bug risk."""
        risk_areas = []
        
        if features.cyclomatic_complexity > 10:
            risk_areas.append("high_complexity")
            
        if features.error_handling_coverage < 0.5:
            risk_areas.append("insufficient_error_handling")
            
        if features.dependency_count > 20:
            risk_areas.append("high_dependency_coupling")
            
        if bug_risk > 0.7:
            risk_areas.append("historical_bug_hotspot")
            
        if features.code_change_frequency > 0.8:
            risk_areas.append("frequent_changes")
            
        return risk_areas
        
    def _calculate_prediction_confidence(self, X: np.ndarray) -> float:
        """Calculate confidence in predictions."""
        # Use ensemble agreement as confidence measure
        confidences = []
        
        if hasattr(self.test_type_classifier, 'predict_proba'):
            probs = self.test_type_classifier.predict_proba(X)
            # Confidence is max probability
            confidences.append(np.max(probs[0]))
            
        # Add more confidence measures from other models
        # For now, use a simple average
        return np.mean(confidences) if confidences else 0.7
        
    def _calculate_recommended_test_count(self, complexity: float) -> int:
        """Calculate recommended number of tests."""
        base_count = 3
        complexity_factor = int(complexity * 5)
        return base_count + complexity_factor
        
    def _calculate_priority_score(self, bug_risk: float, coverage_improvement: float) -> float:
        """Calculate priority score for test generation."""
        return 0.6 * bug_risk + 0.4 * coverage_improvement
        
    def _models_trained(self) -> bool:
        """Check if models are trained."""
        return (
            self.test_type_classifier is not None and
            self.complexity_predictor is not None and
            self.execution_time_predictor is not None
        )
        
    def _get_default_prediction(self) -> PredictionResult:
        """Get default prediction when models aren't trained."""
        return PredictionResult(
            suggested_test_types=["unit", "edge_case"],
            predicted_complexity=0.5,
            estimated_execution_time=2.0,
            expected_coverage_improvement=0.15,
            risk_areas=["untrained_model"],
            confidence_score=0.3,
            recommended_test_count=5,
            priority_score=0.5
        )
        
    def _generate_synthetic_training_data(
        self, 
        count: int
    ) -> List[Tuple[TestPredictionFeatures, TestCase]]:
        """Generate synthetic training data for model initialization."""
        synthetic_data = []
        
        test_types = ["unit", "integration", "edge_case", "error_handling"]
        
        for i in range(count):
            # Generate realistic feature values
            complexity = np.random.beta(2, 5)  # Skewed towards lower complexity
            function_count = np.random.poisson(10)
            line_count = np.random.exponential(50)
            
            features = TestPredictionFeatures(
                code_complexity=complexity,
                function_count=function_count,
                line_count=line_count,
                cyclomatic_complexity=np.random.exponential(5),
                dependency_count=np.random.poisson(8),
                error_handling_coverage=np.random.beta(3, 2),
                function_signatures=[f"func_{i}"],
                code_patterns=["pattern_a", "pattern_b"],
                historical_bug_density=np.random.exponential(0.1),
                test_execution_time=np.random.exponential(2),
                code_change_frequency=np.random.beta(2, 8)
            )
            
            # Generate correlated test case
            test_case = TestCase(
                id=f"synthetic_{i}",
                function_name=f"func_{i}",
                test_type=np.random.choice(test_types),
                complexity_score=complexity + np.random.normal(0, 0.1),
                execution_time=features.test_execution_time * (1 + complexity),
                success_rate=np.random.beta(8, 2),
                bug_detection_rate=complexity * 0.5 + np.random.normal(0, 0.1),
                code_coverage=0.7 + complexity * 0.2 + np.random.normal(0, 0.1),
                maintenance_effort=complexity * 2,
                business_impact=np.random.uniform(0.3, 1.0)
            )
            
            synthetic_data.append((features, test_case))
            
        return synthetic_data
        
    def _save_models(self) -> None:
        """Save trained models to disk."""
        models = {
            "test_type_classifier": self.test_type_classifier,
            "complexity_predictor": self.complexity_predictor,
            "execution_time_predictor": self.execution_time_predictor,
            "coverage_predictor": self.coverage_predictor,
            "bug_risk_predictor": self.bug_risk_predictor,
            "feature_scaler": self.feature_scaler,
            "label_encoders": self.label_encoders
        }
        
        for name, model in models.items():
            if model is not None:
                model_path = self.model_dir / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        model_files = {
            "test_type_classifier": self.model_dir / "test_type_classifier.pkl",
            "complexity_predictor": self.model_dir / "complexity_predictor.pkl",
            "execution_time_predictor": self.model_dir / "execution_time_predictor.pkl",
            "coverage_predictor": self.model_dir / "coverage_predictor.pkl",
            "bug_risk_predictor": self.model_dir / "bug_risk_predictor.pkl",
            "feature_scaler": self.model_dir / "feature_scaler.pkl",
            "label_encoders": self.model_dir / "label_encoders.pkl"
        }
        
        for name, path in model_files.items():
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                        setattr(self, name, model)
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
                    
    def _create_feedback_sample(
        self,
        prediction: PredictionResult,
        actual_results: Dict[str, Any]
    ) -> Tuple[TestPredictionFeatures, TestCase]:
        """Create feedback sample from prediction and actual results."""
        
        # Extract features (this would come from the original prediction context)
        features = TestPredictionFeatures(
            code_complexity=actual_results.get("code_complexity", 0.5),
            function_count=actual_results.get("function_count", 5),
            line_count=actual_results.get("line_count", 50),
            cyclomatic_complexity=actual_results.get("cyclomatic_complexity", 3),
            dependency_count=actual_results.get("dependency_count", 8),
            error_handling_coverage=actual_results.get("error_handling_coverage", 0.7),
            function_signatures=actual_results.get("function_signatures", ["func"]),
            code_patterns=actual_results.get("code_patterns", []),
            historical_bug_density=actual_results.get("historical_bug_density", 0.1),
            test_execution_time=actual_results.get("test_execution_time", 2.0),
            code_change_frequency=actual_results.get("code_change_frequency", 0.3)
        )
        
        # Create test case from actual results
        test_case = TestCase(
            id=actual_results.get("test_id", "feedback"),
            function_name=actual_results.get("function_name", "unknown"),
            test_type=actual_results.get("actual_test_type", "unit"),
            complexity_score=actual_results.get("actual_complexity", prediction.predicted_complexity),
            execution_time=actual_results.get("actual_execution_time", prediction.estimated_execution_time),
            success_rate=actual_results.get("success_rate", 0.8),
            bug_detection_rate=actual_results.get("bug_detection_rate", 0.3),
            code_coverage=actual_results.get("actual_coverage", 0.7),
            maintenance_effort=actual_results.get("maintenance_effort", 3.0),
            business_impact=actual_results.get("business_impact", 0.5)
        )
        
        return features, test_case
        
    def _calculate_average_performance(self, performance_results: Dict[str, ModelPerformance]) -> ModelPerformance:
        """Calculate average performance across all models."""
        performances = list(performance_results.values())
        
        return ModelPerformance(
            accuracy=np.mean([p.accuracy for p in performances]),
            precision=np.mean([p.precision for p in performances]),
            recall=np.mean([p.recall for p in performances]),
            f1_score=np.mean([p.f1_score for p in performances]),
            cross_val_score=np.mean([p.cross_val_score for p in performances]),
            training_time=sum(p.training_time for p in performances),
            prediction_time=0.0,
            model_version="ensemble_1.0"
        )
        
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend over time."""
        if len(self.historical_performance) < 2:
            return "insufficient_data"
            
        recent_accuracy = self.historical_performance[-1].accuracy
        previous_accuracy = self.historical_performance[-2].accuracy
        
        if recent_accuracy > previous_accuracy + 0.01:
            return "improving"
        elif recent_accuracy < previous_accuracy - 0.01:
            return "declining"
        else:
            return "stable"
            
    def _analyze_prediction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in predictions."""
        return {
            "most_common_test_type": "unit",
            "average_complexity": 0.6,
            "average_execution_time": 2.5,
            "high_risk_frequency": 0.15
        }


async def main():
    """Example usage of neural test predictor."""
    predictor = NeuralTestPredictor()
    
    # Generate some training data
    training_data = predictor._generate_synthetic_training_data(100)
    
    # Train models
    performance = predictor.train_models(training_data)
    print("Training complete!")
    
    # Make prediction
    features = TestPredictionFeatures(
        code_complexity=0.7,
        function_count=8,
        line_count=120,
        cyclomatic_complexity=5,
        dependency_count=12,
        error_handling_coverage=0.6,
        function_signatures=["calculate_discount", "validate_input"],
        code_patterns=["conditional_logic", "loop_processing"],
        historical_bug_density=0.2,
        test_execution_time=3.0,
        code_change_frequency=0.4
    )
    
    prediction = predictor.predict_optimal_tests(features)
    print(f"Recommended tests: {prediction.suggested_test_types}")
    print(f"Expected complexity: {prediction.predicted_complexity:.2f}")
    print(f"Estimated time: {prediction.estimated_execution_time:.1f}s")
    print(f"Confidence: {prediction.confidence_score:.2%}")


if __name__ == "__main__":
    asyncio.run(main())