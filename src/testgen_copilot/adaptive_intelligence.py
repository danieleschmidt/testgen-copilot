"""
🧠 Adaptive Intelligence System
===============================

Self-learning system that adapts development strategies based on execution patterns,
success rates, and environmental feedback. Implements quantum-inspired learning
algorithms for continuous improvement.
"""

import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .logging_config import setup_logger

logger = setup_logger(__name__)


class AdaptiveStrategy(Enum):
    """Types of adaptive strategies the system can employ"""
    CONSERVATIVE = "conservative"        # Low risk, proven methods
    BALANCED = "balanced"               # Mix of proven and experimental
    AGGRESSIVE = "aggressive"           # High innovation, higher risk
    EXPERIMENTAL = "experimental"       # Cutting-edge, research-focused


@dataclass
class LearningPattern:
    """Represents a learned pattern with success metrics"""
    pattern_id: str
    description: str
    context: Dict[str, Any]
    success_rate: float = 0.0
    confidence: float = 0.0
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    effectiveness_score: float = 0.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EnvironmentContext:
    """Current environment context for adaptive decisions"""
    project_type: str
    technology_stack: List[str]
    team_size: int
    deadline_pressure: float  # 0.0 to 1.0
    risk_tolerance: float     # 0.0 to 1.0
    performance_requirements: Dict[str, float]
    resource_constraints: Dict[str, float]


class AdaptiveIntelligenceSystem:
    """
    🧠 Adaptive Intelligence System for Autonomous SDLC
    
    Features:
    - Pattern recognition and learning
    - Strategy adaptation based on context
    - Performance optimization through feedback loops
    - Risk-aware decision making
    - Quantum-inspired optimization algorithms
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.current_strategy = AdaptiveStrategy.BALANCED
        self.adaptation_threshold = 0.1  # Minimum improvement needed to adapt
        
        # Learning parameters
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.decay_rate = 0.95
        
        # Load existing patterns
        self._load_learned_patterns()
    
    def _load_learned_patterns(self) -> None:
        """Load previously learned patterns for continuous learning"""
        patterns_file = self.project_path / ".adaptive_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    data = json.load(f)
                    for pattern_data in data.get("patterns", []):
                        pattern = LearningPattern(**pattern_data)
                        self.learning_patterns[pattern.pattern_id] = pattern
                logger.info(f"Loaded {len(self.learning_patterns)} adaptive patterns")
            except Exception as e:
                logger.warning(f"Failed to load adaptive patterns: {e}")
    
    def _save_learned_patterns(self) -> None:
        """Save learned patterns for future runs"""
        patterns_file = self.project_path / ".adaptive_patterns.json"
        try:
            data = {
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "description": p.description,
                        "context": p.context,
                        "success_rate": p.success_rate,
                        "confidence": p.confidence,
                        "usage_count": p.usage_count,
                        "last_used": p.last_used.isoformat(),
                        "effectiveness_score": p.effectiveness_score,
                        "adaptation_history": p.adaptation_history
                    }
                    for p in self.learning_patterns.values()
                ]
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved adaptive patterns for future learning")
        except Exception as e:
            logger.error(f"Failed to save adaptive patterns: {e}")
    
    async def analyze_context(self, project_info: Dict[str, Any]) -> EnvironmentContext:
        """Analyze current project context for adaptive decision making"""
        context = EnvironmentContext(
            project_type=project_info.get("type", "unknown"),
            technology_stack=project_info.get("technologies", []),
            team_size=project_info.get("team_size", 1),
            deadline_pressure=project_info.get("deadline_pressure", 0.5),
            risk_tolerance=project_info.get("risk_tolerance", 0.5),
            performance_requirements=project_info.get("performance_requirements", {}),
            resource_constraints=project_info.get("resource_constraints", {})
        )
        
        logger.info(f"Analyzed context: {context.project_type} with {len(context.technology_stack)} technologies")
        return context
    
    async def select_optimal_strategy(self, context: EnvironmentContext) -> AdaptiveStrategy:
        """Select optimal strategy based on context and learned patterns"""
        
        # Calculate strategy scores based on context
        strategy_scores = {}
        
        for strategy in AdaptiveStrategy:
            score = await self._calculate_strategy_score(strategy, context)
            strategy_scores[strategy] = score
        
        # Select strategy with highest score
        optimal_strategy = max(strategy_scores, key=strategy_scores.get)
        
        logger.info(f"Selected strategy: {optimal_strategy.value} (score: {strategy_scores[optimal_strategy]:.3f})")
        return optimal_strategy
    
    async def _calculate_strategy_score(self, strategy: AdaptiveStrategy, context: EnvironmentContext) -> float:
        """Calculate score for a strategy based on context and learned patterns"""
        base_score = 0.5
        
        # Adjust based on risk tolerance
        if strategy == AdaptiveStrategy.CONSERVATIVE:
            base_score += (1.0 - context.risk_tolerance) * 0.3
        elif strategy == AdaptiveStrategy.AGGRESSIVE:
            base_score += context.risk_tolerance * 0.3
        elif strategy == AdaptiveStrategy.EXPERIMENTAL:
            base_score += context.risk_tolerance * 0.5 - 0.2  # Higher risk
        
        # Adjust based on deadline pressure
        if strategy == AdaptiveStrategy.CONSERVATIVE and context.deadline_pressure > 0.7:
            base_score += 0.2  # Conservative is safer under pressure
        elif strategy == AdaptiveStrategy.EXPERIMENTAL and context.deadline_pressure > 0.5:
            base_score -= 0.3  # Experimental is risky under pressure
        
        # Apply learned pattern bonuses
        for pattern in self.learning_patterns.values():
            if self._pattern_matches_context(pattern, context):
                if strategy.value in pattern.context.get("successful_strategies", []):
                    base_score += pattern.effectiveness_score * 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _pattern_matches_context(self, pattern: LearningPattern, context: EnvironmentContext) -> bool:
        """Check if a learned pattern matches the current context"""
        pattern_context = pattern.context
        
        # Check project type match
        if pattern_context.get("project_type") == context.project_type:
            return True
        
        # Check technology stack overlap
        pattern_tech = set(pattern_context.get("technology_stack", []))
        context_tech = set(context.technology_stack)
        if pattern_tech.intersection(context_tech):
            return True
        
        return False
    
    async def adapt_from_execution(self, execution_result: Dict[str, Any]) -> None:
        """Adapt intelligence based on execution results"""
        
        # Extract key metrics from execution
        success_rate = execution_result.get("success_rate", 0.0)
        performance_metrics = execution_result.get("performance_metrics", {})
        strategy_used = execution_result.get("strategy_used", self.current_strategy.value)
        context = execution_result.get("context", {})
        
        # Update execution history
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "success_rate": success_rate,
            "strategy": strategy_used,
            "context": context,
            "performance": performance_metrics
        })
        
        # Learn from successful patterns
        if success_rate > 0.8:  # High success rate
            await self._reinforce_successful_pattern(execution_result)
        elif success_rate < 0.5:  # Low success rate
            await self._learn_from_failure(execution_result)
        
        # Adapt strategy if needed
        await self._adapt_strategy_selection()
        
        # Clean up old patterns
        await self._cleanup_obsolete_patterns()
        
        # Save updated patterns
        self._save_learned_patterns()
    
    async def _reinforce_successful_pattern(self, execution_result: Dict[str, Any]) -> None:
        """Reinforce patterns that led to successful execution"""
        pattern_id = self._generate_pattern_id(execution_result)
        
        if pattern_id in self.learning_patterns:
            # Update existing pattern
            pattern = self.learning_patterns[pattern_id]
            old_success_rate = pattern.success_rate
            pattern.success_rate = (pattern.success_rate * pattern.usage_count + execution_result["success_rate"]) / (pattern.usage_count + 1)
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            pattern.confidence = min(1.0, pattern.confidence + self.learning_rate * (execution_result["success_rate"] - old_success_rate))
            pattern.effectiveness_score = pattern.success_rate * pattern.confidence
            
            # Record adaptation
            pattern.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "old_success_rate": old_success_rate,
                "new_success_rate": pattern.success_rate,
                "improvement": pattern.success_rate - old_success_rate
            })
        else:
            # Create new pattern
            pattern = LearningPattern(
                pattern_id=pattern_id,
                description=f"Successful pattern for {execution_result.get('strategy_used', 'unknown')} strategy",
                context=execution_result.get("context", {}),
                success_rate=execution_result["success_rate"],
                confidence=0.7,  # Start with medium confidence
                usage_count=1,
                effectiveness_score=execution_result["success_rate"] * 0.7
            )
            self.learning_patterns[pattern_id] = pattern
        
        logger.info(f"Reinforced successful pattern: {pattern_id}")
    
    async def _learn_from_failure(self, execution_result: Dict[str, Any]) -> None:
        """Learn from failed execution to avoid similar issues"""
        pattern_id = self._generate_pattern_id(execution_result)
        
        if pattern_id in self.learning_patterns:
            # Decrease confidence in failed pattern
            pattern = self.learning_patterns[pattern_id]
            pattern.confidence *= self.decay_rate
            pattern.effectiveness_score *= self.decay_rate
            
            # Record failure
            pattern.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "failure": True,
                "success_rate": execution_result["success_rate"],
                "confidence_adjustment": -0.1
            })
        
        # Create anti-pattern to avoid similar failures
        anti_pattern_id = f"avoid_{pattern_id}"
        if anti_pattern_id not in self.learning_patterns:
            anti_pattern = LearningPattern(
                pattern_id=anti_pattern_id,
                description=f"Anti-pattern to avoid: {execution_result.get('failure_reason', 'unknown failure')}",
                context=execution_result.get("context", {}),
                success_rate=0.0,  # Explicitly mark as unsuccessful
                confidence=0.8,  # High confidence in avoiding this
                effectiveness_score=-0.5  # Negative effectiveness
            )
            self.learning_patterns[anti_pattern_id] = anti_pattern
        
        logger.warning(f"Learned from failure, created anti-pattern: {anti_pattern_id}")
    
    def _generate_pattern_id(self, execution_result: Dict[str, Any]) -> str:
        """Generate unique pattern ID based on execution context"""
        context = execution_result.get("context", {})
        strategy = execution_result.get("strategy_used", "unknown")
        project_type = context.get("project_type", "unknown")
        
        # Create hash-like ID from key characteristics
        pattern_components = [
            strategy,
            project_type,
            str(sorted(context.get("technology_stack", []))),
            str(context.get("team_size", 1))
        ]
        
        pattern_id = "_".join(pattern_components).replace(" ", "_").lower()
        return pattern_id[:50]  # Limit length
    
    async def _adapt_strategy_selection(self) -> None:
        """Adapt strategy selection based on recent execution history"""
        if len(self.execution_history) < 5:
            return  # Need sufficient data for adaptation
        
        # Analyze recent performance
        recent_executions = self.execution_history[-5:]
        strategy_performance = {}
        
        for execution in recent_executions:
            strategy = execution["strategy"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(execution["success_rate"])
        
        # Calculate average performance per strategy
        strategy_avg_performance = {
            strategy: np.mean(performances)
            for strategy, performances in strategy_performance.items()
        }
        
        # Adapt current strategy if significant improvement available
        if strategy_avg_performance:
            best_strategy = max(strategy_avg_performance, key=strategy_avg_performance.get)
            current_performance = strategy_avg_performance.get(self.current_strategy.value, 0.0)
            best_performance = strategy_avg_performance[best_strategy]
            
            if best_performance > current_performance + self.adaptation_threshold:
                old_strategy = self.current_strategy
                self.current_strategy = AdaptiveStrategy(best_strategy)
                logger.info(f"Adapted strategy from {old_strategy.value} to {self.current_strategy.value}")
    
    async def _cleanup_obsolete_patterns(self) -> None:
        """Remove patterns that are no longer relevant or effective"""
        current_time = datetime.now()
        patterns_to_remove = []
        
        for pattern_id, pattern in self.learning_patterns.items():
            # Remove if not used in last 30 days and low effectiveness
            days_since_use = (current_time - pattern.last_used).days
            if days_since_use > 30 and pattern.effectiveness_score < 0.3:
                patterns_to_remove.append(pattern_id)
            
            # Remove anti-patterns with very low confidence
            if pattern_id.startswith("avoid_") and pattern.confidence < 0.2:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.learning_patterns[pattern_id]
        
        if patterns_to_remove:
            logger.info(f"Cleaned up {len(patterns_to_remove)} obsolete patterns")
    
    async def get_recommendation_for_task(self, task: Dict[str, Any], context: EnvironmentContext) -> Dict[str, Any]:
        """Get AI recommendation for executing a specific task"""
        
        # Find matching patterns
        matching_patterns = [
            pattern for pattern in self.learning_patterns.values()
            if self._pattern_matches_context(pattern, context) and pattern.effectiveness_score > 0.5
        ]
        
        # Sort by effectiveness
        matching_patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
        
        recommendation = {
            "confidence": 0.5,
            "approach": "standard",
            "risk_level": "medium",
            "estimated_success_rate": 0.7,
            "specific_guidance": [],
            "patterns_used": []
        }
        
        if matching_patterns:
            best_pattern = matching_patterns[0]
            recommendation.update({
                "confidence": best_pattern.confidence,
                "estimated_success_rate": best_pattern.success_rate,
                "patterns_used": [best_pattern.pattern_id],
                "specific_guidance": self._extract_guidance_from_pattern(best_pattern, task)
            })
        
        return recommendation
    
    def _extract_guidance_from_pattern(self, pattern: LearningPattern, task: Dict[str, Any]) -> List[str]:
        """Extract specific guidance from a learned pattern"""
        guidance = []
        
        if pattern.success_rate > 0.8:
            guidance.append(f"High success pattern: {pattern.description}")
        
        if "successful_strategies" in pattern.context:
            strategies = pattern.context["successful_strategies"]
            guidance.append(f"Recommended strategies: {', '.join(strategies)}")
        
        if "optimization_techniques" in pattern.context:
            techniques = pattern.context["optimization_techniques"]
            guidance.append(f"Apply optimizations: {', '.join(techniques)}")
        
        return guidance
    
    async def predict_execution_outcome(self, planned_execution: Dict[str, Any]) -> Dict[str, float]:
        """Predict likely outcome of planned execution based on learned patterns"""
        
        context = planned_execution.get("context", {})
        strategy = planned_execution.get("strategy", self.current_strategy.value)
        
        # Find relevant patterns
        relevant_patterns = [
            pattern for pattern in self.learning_patterns.values()
            if self._matches_execution_plan(pattern, planned_execution)
        ]
        
        if not relevant_patterns:
            # Return default prediction
            return {
                "success_probability": 0.7,
                "completion_time_hours": 2.0,
                "quality_score": 0.8,
                "risk_score": 0.3
            }
        
        # Weight patterns by relevance and confidence
        total_weight = sum(p.confidence * p.effectiveness_score for p in relevant_patterns)
        
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero
        
        weighted_success = sum(
            p.success_rate * p.confidence * p.effectiveness_score
            for p in relevant_patterns
        ) / total_weight
        
        return {
            "success_probability": weighted_success,
            "completion_time_hours": 2.0,  # Could be learned from patterns
            "quality_score": 0.8 + (weighted_success - 0.5) * 0.4,
            "risk_score": 1.0 - weighted_success
        }
    
    def _matches_execution_plan(self, pattern: LearningPattern, planned_execution: Dict[str, Any]) -> bool:
        """Check if pattern matches the planned execution"""
        # Simple matching logic - can be enhanced
        pattern_context = pattern.context
        plan_context = planned_execution.get("context", {})
        
        # Check technology stack overlap
        pattern_tech = set(pattern_context.get("technology_stack", []))
        plan_tech = set(plan_context.get("technology_stack", []))
        
        return bool(pattern_tech.intersection(plan_tech))