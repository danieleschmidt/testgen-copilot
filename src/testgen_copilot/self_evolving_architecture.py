"""
Self-Evolving Code Architecture for TestGen Copilot
Implements autonomous code modification and optimization capabilities
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CodeMutation:
    """Represents a potential code modification."""
    id: str
    file_path: Path
    original_code: str
    mutated_code: str
    mutation_type: str  # optimization, refactor, enhancement, bug_fix
    confidence_score: float
    impact_estimation: Dict[str, float]
    safety_analysis: Dict[str, Any]
    performance_prediction: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class EvolutionMetrics:
    """Metrics for tracking code evolution performance."""
    generation: int
    fitness_score: float
    performance_metrics: Dict[str, float]
    code_quality_metrics: Dict[str, float]
    test_coverage: float
    security_score: float
    maintainability_score: float
    mutation_success_rate: float
    rollback_count: int = 0
    

@dataclass
class SafetyConstraints:
    """Safety constraints for code evolution."""
    max_mutations_per_cycle: int = 5
    min_test_coverage: float = 0.85
    max_performance_regression: float = 0.05
    required_security_score: float = 0.9
    mandatory_tests: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    rollback_threshold: float = 0.7


class SelfEvolvingArchitecture:
    """
    Autonomous code evolution system that safely modifies and optimizes code.
    Implements genetic programming principles with safety constraints.
    """
    
    def __init__(
        self, 
        codebase_path: Path,
        evolution_log_path: Path = Path("evolution_log"),
        safety_constraints: SafetyConstraints = None
    ):
        self.codebase_path = Path(codebase_path)
        self.evolution_log_path = Path(evolution_log_path)
        self.evolution_log_path.mkdir(exist_ok=True)
        
        self.safety_constraints = safety_constraints or SafetyConstraints()
        
        # Evolution state
        self.current_generation = 0
        self.evolution_history: List[EvolutionMetrics] = []
        self.active_mutations: Dict[str, CodeMutation] = {}
        self.successful_mutations: List[CodeMutation] = []
        self.failed_mutations: List[CodeMutation] = []
        
        # Code analysis
        self.code_analyzer = CodeAnalyzer()
        self.performance_profiler = PerformanceProfiler()
        self.safety_checker = SafetyChecker(self.safety_constraints)
        
        # Backup management
        self.backup_dir = self.evolution_log_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def evolve_codebase(self, cycles: int = 10) -> List[EvolutionMetrics]:
        """
        Run autonomous code evolution for specified number of cycles.
        """
        logger.info(f"Starting autonomous code evolution for {cycles} cycles")
        
        results = []
        
        for cycle in range(cycles):
            logger.info(f"Starting evolution cycle {cycle + 1}/{cycles}")
            
            # Create backup
            backup_path = self._create_backup()
            
            try:
                # Analyze current state
                current_metrics = self._analyze_current_state()
                
                # Generate mutations
                mutations = self._generate_mutations()
                
                # Apply mutations safely
                applied_mutations = self._apply_mutations_safely(mutations)
                
                # Evaluate fitness
                fitness_metrics = self._evaluate_fitness()
                
                # Decide on keeping changes
                if self._should_keep_changes(current_metrics, fitness_metrics):
                    self._commit_generation()
                    results.append(fitness_metrics)
                    logger.info(f"Cycle {cycle + 1} successful: fitness {fitness_metrics.fitness_score:.3f}")
                else:
                    self._rollback_to_backup(backup_path)
                    fitness_metrics.rollback_count += 1
                    logger.info(f"Cycle {cycle + 1} rolled back due to fitness degradation")
                    
                self.current_generation += 1
                
            except Exception as e:
                logger.error(f"Error in evolution cycle {cycle + 1}: {e}")
                self._rollback_to_backup(backup_path)
                
        logger.info(f"Evolution complete: {len(results)} successful cycles")
        return results
        
    def _analyze_current_state(self) -> EvolutionMetrics:
        """Analyze current codebase state and metrics."""
        logger.debug("Analyzing current codebase state")
        
        # Code quality analysis
        quality_metrics = self.code_analyzer.analyze_quality(self.codebase_path)
        
        # Performance profiling
        performance_metrics = self.performance_profiler.profile_codebase(self.codebase_path)
        
        # Test coverage
        coverage = self._calculate_test_coverage()
        
        # Security analysis
        security_score = self._calculate_security_score()
        
        # Maintainability
        maintainability = self.code_analyzer.calculate_maintainability(self.codebase_path)
        
        # Overall fitness
        fitness = self._calculate_fitness_score(
            performance_metrics, quality_metrics, coverage, security_score, maintainability
        )
        
        metrics = EvolutionMetrics(
            generation=self.current_generation,
            fitness_score=fitness,
            performance_metrics=performance_metrics,
            code_quality_metrics=quality_metrics,
            test_coverage=coverage,
            security_score=security_score,
            maintainability_score=maintainability,
            mutation_success_rate=self._calculate_mutation_success_rate()
        )
        
        return metrics
        
    def _generate_mutations(self) -> List[CodeMutation]:
        """Generate potential code mutations based on analysis."""
        logger.debug("Generating code mutations")
        
        mutations = []
        
        # Identify optimization opportunities
        optimization_targets = self.code_analyzer.identify_optimization_opportunities(
            self.codebase_path
        )
        
        for target in optimization_targets[:self.safety_constraints.max_mutations_per_cycle]:
            mutation = self._create_optimization_mutation(target)
            if mutation:
                mutations.append(mutation)
                
        # Generate refactoring mutations
        refactor_opportunities = self.code_analyzer.identify_refactoring_opportunities(
            self.codebase_path
        )
        
        for opportunity in refactor_opportunities[:2]:  # Limit refactoring mutations
            mutation = self._create_refactoring_mutation(opportunity)
            if mutation:
                mutations.append(mutation)
                
        # Generate enhancement mutations
        enhancement_opportunities = self._identify_enhancement_opportunities()
        
        for enhancement in enhancement_opportunities[:2]:
            mutation = self._create_enhancement_mutation(enhancement)
            if mutation:
                mutations.append(mutation)
                
        logger.info(f"Generated {len(mutations)} potential mutations")
        return mutations
        
    def _create_optimization_mutation(self, target: Dict[str, Any]) -> Optional[CodeMutation]:
        """Create an optimization mutation for a specific target."""
        file_path = Path(target["file_path"])
        
        if not file_path.exists():
            return None
            
        with open(file_path, 'r') as f:
            original_code = f.read()
            
        # Apply optimization transformation
        optimized_code = self._apply_optimization_transform(original_code, target)
        
        if optimized_code == original_code:
            return None
            
        mutation_id = f"opt_{int(time.time())}_{hash(str(file_path)) % 10000}"
        
        mutation = CodeMutation(
            id=mutation_id,
            file_path=file_path,
            original_code=original_code,
            mutated_code=optimized_code,
            mutation_type="optimization",
            confidence_score=target.get("confidence", 0.7),
            impact_estimation=target.get("impact", {}),
            safety_analysis=self.safety_checker.analyze_mutation_safety(
                original_code, optimized_code
            ),
            performance_prediction=target.get("performance_prediction", {})
        )
        
        return mutation
        
    def _create_refactoring_mutation(self, opportunity: Dict[str, Any]) -> Optional[CodeMutation]:
        """Create a refactoring mutation."""
        file_path = Path(opportunity["file_path"])
        
        with open(file_path, 'r') as f:
            original_code = f.read()
            
        refactored_code = self._apply_refactoring_transform(original_code, opportunity)
        
        if refactored_code == original_code:
            return None
            
        mutation_id = f"ref_{int(time.time())}_{hash(str(file_path)) % 10000}"
        
        mutation = CodeMutation(
            id=mutation_id,
            file_path=file_path,
            original_code=original_code,
            mutated_code=refactored_code,
            mutation_type="refactor",
            confidence_score=opportunity.get("confidence", 0.8),
            impact_estimation=opportunity.get("impact", {}),
            safety_analysis=self.safety_checker.analyze_mutation_safety(
                original_code, refactored_code
            ),
            performance_prediction={}
        )
        
        return mutation
        
    def _create_enhancement_mutation(self, enhancement: Dict[str, Any]) -> Optional[CodeMutation]:
        """Create an enhancement mutation."""
        file_path = Path(enhancement["file_path"])
        
        with open(file_path, 'r') as f:
            original_code = f.read()
            
        enhanced_code = self._apply_enhancement_transform(original_code, enhancement)
        
        if enhanced_code == original_code:
            return None
            
        mutation_id = f"enh_{int(time.time())}_{hash(str(file_path)) % 10000}"
        
        mutation = CodeMutation(
            id=mutation_id,
            file_path=file_path,
            original_code=original_code,
            mutated_code=enhanced_code,
            mutation_type="enhancement",
            confidence_score=enhancement.get("confidence", 0.6),
            impact_estimation=enhancement.get("impact", {}),
            safety_analysis=self.safety_checker.analyze_mutation_safety(
                original_code, enhanced_code
            ),
            performance_prediction=enhancement.get("performance_prediction", {})
        )
        
        return mutation
        
    def _apply_mutations_safely(self, mutations: List[CodeMutation]) -> List[CodeMutation]:
        """Apply mutations with safety checks."""
        logger.debug(f"Applying {len(mutations)} mutations with safety checks")
        
        applied_mutations = []
        
        for mutation in mutations:
            # Safety pre-check
            if not self.safety_checker.is_mutation_safe(mutation):
                logger.warning(f"Mutation {mutation.id} failed safety check")
                self.failed_mutations.append(mutation)
                continue
                
            # Apply mutation
            try:
                self._apply_single_mutation(mutation)
                
                # Verify application
                if self._verify_mutation_applied(mutation):
                    applied_mutations.append(mutation)
                    self.active_mutations[mutation.id] = mutation
                else:
                    logger.warning(f"Mutation {mutation.id} verification failed")
                    self.failed_mutations.append(mutation)
                    
            except Exception as e:
                logger.error(f"Failed to apply mutation {mutation.id}: {e}")
                self.failed_mutations.append(mutation)
                
        logger.info(f"Successfully applied {len(applied_mutations)} mutations")
        return applied_mutations
        
    def _apply_single_mutation(self, mutation: CodeMutation) -> None:
        """Apply a single code mutation."""
        # Write mutated code to file
        with open(mutation.file_path, 'w') as f:
            f.write(mutation.mutated_code)
            
    def _verify_mutation_applied(self, mutation: CodeMutation) -> bool:
        """Verify that mutation was applied correctly."""
        try:
            with open(mutation.file_path, 'r') as f:
                current_code = f.read()
                
            # Check if the code matches expected mutation
            return current_code.strip() == mutation.mutated_code.strip()
            
        except Exception:
            return False
            
    def _evaluate_fitness(self) -> EvolutionMetrics:
        """Evaluate fitness of current generation."""
        logger.debug("Evaluating generation fitness")
        
        # Run tests to ensure nothing is broken
        test_results = self._run_test_suite()
        
        if not test_results["all_passed"]:
            logger.warning("Some tests failed in fitness evaluation")
            
        # Calculate new metrics
        return self._analyze_current_state()
        
    def _should_keep_changes(
        self, 
        previous_metrics: EvolutionMetrics, 
        current_metrics: EvolutionMetrics
    ) -> bool:
        """Determine if current changes should be kept."""
        
        # Check minimum requirements
        if current_metrics.test_coverage < self.safety_constraints.min_test_coverage:
            logger.info("Rejecting changes: test coverage below threshold")
            return False
            
        if current_metrics.security_score < self.safety_constraints.required_security_score:
            logger.info("Rejecting changes: security score below threshold")
            return False
            
        # Check performance regression
        perf_regression = (
            previous_metrics.fitness_score - current_metrics.fitness_score
        ) / previous_metrics.fitness_score
        
        if perf_regression > self.safety_constraints.max_performance_regression:
            logger.info(f"Rejecting changes: performance regression {perf_regression:.3f}")
            return False
            
        # If fitness improved, keep changes
        if current_metrics.fitness_score > previous_metrics.fitness_score:
            logger.info("Accepting changes: fitness improved")
            return True
            
        # If fitness is similar but other metrics improved
        fitness_diff = abs(current_metrics.fitness_score - previous_metrics.fitness_score)
        if fitness_diff < 0.01:  # Similar fitness
            if (current_metrics.maintainability_score > previous_metrics.maintainability_score or
                current_metrics.code_quality_metrics.get("complexity_score", 0) > 
                previous_metrics.code_quality_metrics.get("complexity_score", 0)):
                logger.info("Accepting changes: metrics improved with similar fitness")
                return True
                
        logger.info("Rejecting changes: no significant improvement")
        return False
        
    def _commit_generation(self) -> None:
        """Commit current generation and update history."""
        # Move active mutations to successful
        for mutation in self.active_mutations.values():
            self.successful_mutations.append(mutation)
            
        self.active_mutations.clear()
        
        # Log generation
        self._log_generation()
        
    def _rollback_to_backup(self, backup_path: Path) -> None:
        """Rollback codebase to backup state."""
        logger.info("Rolling back to previous state")
        
        # Restore from backup
        if backup_path.exists():
            # Remove current codebase
            temp_current = Path(str(self.codebase_path) + "_temp")
            self.codebase_path.rename(temp_current)
            
            # Restore backup
            shutil.copytree(backup_path, self.codebase_path)
            
            # Remove temp
            shutil.rmtree(temp_current, ignore_errors=True)
            
        # Move active mutations to failed
        for mutation in self.active_mutations.values():
            self.failed_mutations.append(mutation)
            
        self.active_mutations.clear()
        
    def _create_backup(self) -> Path:
        """Create backup of current codebase state."""
        backup_name = f"backup_gen_{self.current_generation}_{int(time.time())}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copytree(self.codebase_path, backup_path, ignore=shutil.ignore_patterns(
            '__pycache__', '*.pyc', '.git', 'venv', '.venv', 'node_modules'
        ))
        
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
        
    def _calculate_fitness_score(
        self,
        performance_metrics: Dict[str, float],
        quality_metrics: Dict[str, float], 
        coverage: float,
        security_score: float,
        maintainability: float
    ) -> float:
        """Calculate overall fitness score."""
        
        # Weighted combination of metrics
        weights = {
            "performance": 0.3,
            "quality": 0.25,
            "coverage": 0.2,
            "security": 0.15,
            "maintainability": 0.1
        }
        
        # Normalize metrics to 0-1 scale
        perf_score = np.mean(list(performance_metrics.values()))
        quality_score = np.mean(list(quality_metrics.values()))
        
        fitness = (
            weights["performance"] * perf_score +
            weights["quality"] * quality_score +
            weights["coverage"] * coverage +
            weights["security"] * security_score +
            weights["maintainability"] * maintainability
        )
        
        return float(fitness)
        
    def _calculate_test_coverage(self) -> float:
        """Calculate current test coverage."""
        try:
            # Run coverage analysis
            result = subprocess.run([
                "python", "-m", "pytest", "--cov=src", "--cov-report=json", "-q"
            ], cwd=self.codebase_path, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse coverage report
                coverage_file = self.codebase_path / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0) / 100
                    
        except Exception as e:
            logger.warning(f"Coverage calculation failed: {e}")
            
        return 0.5  # Default assumption
        
    def _calculate_security_score(self) -> float:
        """Calculate security score."""
        try:
            # Run security analysis
            result = subprocess.run([
                "python", "-m", "bandit", "-r", "src", "-f", "json", "-o", "security_report.json"
            ], cwd=self.codebase_path, capture_output=True, text=True, timeout=60)
            
            security_file = self.codebase_path / "security_report.json"
            if security_file.exists():
                with open(security_file) as f:
                    security_data = json.load(f)
                    
                # Calculate score based on issues
                metrics = security_data.get("metrics", {})
                total_lines = metrics.get("_totals", {}).get("loc", 1)
                high_issues = metrics.get("_totals", {}).get("SEVERITY.HIGH", 0)
                medium_issues = metrics.get("_totals", {}).get("SEVERITY.MEDIUM", 0)
                
                # Score based on issue density
                issue_density = (high_issues * 2 + medium_issues) / total_lines
                security_score = max(0, 1.0 - issue_density * 100)
                
                return security_score
                
        except Exception as e:
            logger.warning(f"Security score calculation failed: {e}")
            
        return 0.8  # Default assumption
        
    def _calculate_mutation_success_rate(self) -> float:
        """Calculate success rate of mutations."""
        total_mutations = len(self.successful_mutations) + len(self.failed_mutations)
        if total_mutations == 0:
            return 0.0
            
        return len(self.successful_mutations) / total_mutations
        
    def _run_test_suite(self) -> Dict[str, Any]:
        """Run test suite and return results."""
        try:
            result = subprocess.run([
                "python", "-m", "pytest", "-v", "--tb=short"
            ], cwd=self.codebase_path, capture_output=True, text=True, timeout=300)
            
            return {
                "all_passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"all_passed": False, "output": "", "errors": "Test timeout"}
        except Exception as e:
            return {"all_passed": False, "output": "", "errors": str(e)}
            
    def _log_generation(self) -> None:
        """Log current generation data."""
        generation_log = {
            "generation": self.current_generation,
            "timestamp": datetime.now().isoformat(),
            "successful_mutations": len(self.successful_mutations),
            "failed_mutations": len(self.failed_mutations),
            "mutation_success_rate": self._calculate_mutation_success_rate()
        }
        
        log_file = self.evolution_log_path / f"generation_{self.current_generation}.json"
        with open(log_file, 'w') as f:
            json.dump(generation_log, f, indent=2)
            
    def _apply_optimization_transform(self, code: str, target: Dict[str, Any]) -> str:
        """Apply optimization transformation to code."""
        # Placeholder for actual optimization logic
        # In real implementation, this would use AST manipulation
        
        optimization_type = target.get("type", "general")
        
        if optimization_type == "loop_optimization":
            return self._optimize_loops(code)
        elif optimization_type == "memory_optimization":
            return self._optimize_memory_usage(code)
        elif optimization_type == "algorithm_improvement":
            return self._improve_algorithms(code)
        else:
            return code
            
    def _apply_refactoring_transform(self, code: str, opportunity: Dict[str, Any]) -> str:
        """Apply refactoring transformation."""
        refactor_type = opportunity.get("type", "general")
        
        if refactor_type == "extract_function":
            return self._extract_functions(code)
        elif refactor_type == "simplify_conditionals":
            return self._simplify_conditionals(code)
        elif refactor_type == "remove_duplication":
            return self._remove_code_duplication(code)
        else:
            return code
            
    def _apply_enhancement_transform(self, code: str, enhancement: Dict[str, Any]) -> str:
        """Apply enhancement transformation."""
        enhancement_type = enhancement.get("type", "general")
        
        if enhancement_type == "add_error_handling":
            return self._add_error_handling(code)
        elif enhancement_type == "improve_logging":
            return self._improve_logging(code)
        elif enhancement_type == "add_type_hints":
            return self._add_type_hints(code)
        else:
            return code
            
    def _optimize_loops(self, code: str) -> str:
        """Optimize loop structures in code."""
        # Placeholder for loop optimization
        # Could implement list comprehension conversion, etc.
        return code
        
    def _optimize_memory_usage(self, code: str) -> str:
        """Optimize memory usage patterns."""
        # Placeholder for memory optimization
        return code
        
    def _improve_algorithms(self, code: str) -> str:
        """Improve algorithmic efficiency.""" 
        # Placeholder for algorithm improvement
        return code
        
    def _extract_functions(self, code: str) -> str:
        """Extract repeated code into functions."""
        # Placeholder for function extraction
        return code
        
    def _simplify_conditionals(self, code: str) -> str:
        """Simplify complex conditional statements."""
        # Placeholder for conditional simplification
        return code
        
    def _remove_code_duplication(self, code: str) -> str:
        """Remove code duplication."""
        # Placeholder for duplication removal
        return code
        
    def _add_error_handling(self, code: str) -> str:
        """Add comprehensive error handling."""
        # Placeholder for error handling enhancement
        return code
        
    def _improve_logging(self, code: str) -> str:
        """Improve logging statements."""
        # Placeholder for logging improvement
        return code
        
    def _add_type_hints(self, code: str) -> str:
        """Add type hints to code."""
        # Placeholder for type hint addition
        return code
        
    def _identify_enhancement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for code enhancement."""
        # Placeholder - would analyze codebase for enhancement opportunities
        return [
            {
                "file_path": self.codebase_path / "src" / "testgen_copilot" / "generator.py",
                "type": "add_error_handling",
                "confidence": 0.8,
                "impact": {"maintainability": 0.2}
            }
        ]


class CodeAnalyzer:
    """Analyzes code for optimization and refactoring opportunities."""
    
    def analyze_quality(self, codebase_path: Path) -> Dict[str, float]:
        """Analyze code quality metrics."""
        return {
            "complexity_score": 0.8,
            "maintainability_score": 0.7,
            "readability_score": 0.85
        }
        
    def calculate_maintainability(self, codebase_path: Path) -> float:
        """Calculate maintainability score."""
        return 0.75
        
    def identify_optimization_opportunities(self, codebase_path: Path) -> List[Dict[str, Any]]:
        """Identify code optimization opportunities."""
        return [
            {
                "file_path": codebase_path / "src" / "testgen_copilot" / "generator.py",
                "type": "loop_optimization",
                "confidence": 0.7,
                "impact": {"performance": 0.15},
                "performance_prediction": {"speed_improvement": 0.1}
            }
        ]
        
    def identify_refactoring_opportunities(self, codebase_path: Path) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities."""
        return [
            {
                "file_path": codebase_path / "src" / "testgen_copilot" / "cli.py",
                "type": "extract_function",
                "confidence": 0.8,
                "impact": {"maintainability": 0.2}
            }
        ]


class PerformanceProfiler:
    """Profiles code performance."""
    
    def profile_codebase(self, codebase_path: Path) -> Dict[str, float]:
        """Profile codebase performance."""
        return {
            "execution_time": 0.8,
            "memory_efficiency": 0.75,
            "cpu_utilization": 0.7
        }


class SafetyChecker:
    """Checks safety of code mutations."""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        
    def is_mutation_safe(self, mutation: CodeMutation) -> bool:
        """Check if mutation is safe to apply."""
        safety_analysis = mutation.safety_analysis
        
        # Check confidence threshold
        if mutation.confidence_score < 0.5:
            return False
            
        # Check for forbidden patterns
        for pattern in self.constraints.forbidden_patterns:
            if pattern in mutation.mutated_code:
                return False
                
        # Check safety analysis results
        if safety_analysis.get("syntax_valid", False) and safety_analysis.get("no_dangerous_patterns", True):
            return True
            
        return False
        
    def analyze_mutation_safety(self, original_code: str, mutated_code: str) -> Dict[str, Any]:
        """Analyze safety of a code mutation."""
        try:
            # Check syntax validity
            ast.parse(mutated_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
            
        # Check for dangerous patterns
        dangerous_patterns = ["exec(", "eval(", "os.system(", "__import__"]
        no_dangerous_patterns = not any(pattern in mutated_code for pattern in dangerous_patterns)
        
        return {
            "syntax_valid": syntax_valid,
            "no_dangerous_patterns": no_dangerous_patterns,
            "code_size_change": len(mutated_code) - len(original_code),
            "complexity_change": 0  # Placeholder
        }


async def main():
    """Example usage of self-evolving architecture."""
    architecture = SelfEvolvingArchitecture(
        codebase_path=Path("src"),
        safety_constraints=SafetyConstraints(
            max_mutations_per_cycle=3,
            min_test_coverage=0.8,
            max_performance_regression=0.1
        )
    )
    
    results = architecture.evolve_codebase(cycles=5)
    
    print(f"Evolution complete! {len(results)} successful cycles")
    for i, metrics in enumerate(results):
        print(f"Cycle {i+1}: Fitness {metrics.fitness_score:.3f}, Coverage {metrics.test_coverage:.2%}")


if __name__ == "__main__":
    asyncio.run(main())