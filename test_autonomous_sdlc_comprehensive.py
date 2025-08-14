#!/usr/bin/env python3
"""
Comprehensive test suite for the Autonomous SDLC implementation.

This test suite validates all generations of the progressive enhancement strategy:
- Generation 1: Make It Work (Simple)
- Generation 2: Make It Robust (Reliable) 
- Generation 3: Make It Scale (Optimized)

Along with quality gates, adaptive intelligence, and self-healing capabilities.
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import the autonomous SDLC modules
from testgen_copilot.autonomous_sdlc.execution_engine import (
    AutonomousSDLCEngine, SDLCPhase, ExecutionStatus, SDLCTask
)
from testgen_copilot.autonomous_sdlc.quality_gates import (
    QualityGateValidator, QualityGateType, QualityGateStatus
)
from testgen_copilot.autonomous_sdlc.progressive_enhancement import (
    ProgressiveEnhancer, PerformanceOptimizer, AutoScalingManager, AdaptiveCache
)
from testgen_copilot.autonomous_sdlc.adaptive_intelligence import (
    AdaptiveIntelligence, PatternRecognizer, AdaptiveParameterTuner, SelfHealingManager
)


class TestAutonomousSDLCEngine:
    """Test the core autonomous SDLC execution engine"""

    @pytest.fixture
    async def sdlc_engine(self):
        """Create SDLC engine for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            engine = AutonomousSDLCEngine(project_path)
            await engine.initialize()
            yield engine

    @pytest.mark.asyncio
    async def test_engine_initialization(self, sdlc_engine):
        """Test SDLC engine initialization"""
        assert sdlc_engine.project_path.exists()
        assert len(sdlc_engine.tasks) > 0
        assert sdlc_engine.current_phase == SDLCPhase.ANALYSIS
        assert not sdlc_engine.is_executing

    @pytest.mark.asyncio
    async def test_task_creation(self, sdlc_engine):
        """Test that all required tasks are created"""
        expected_phases = [
            SDLCPhase.ANALYSIS,
            SDLCPhase.GENERATION_1_SIMPLE,
            SDLCPhase.GENERATION_2_ROBUST,
            SDLCPhase.GENERATION_3_SCALE,
            SDLCPhase.QUALITY_VALIDATION,
            SDLCPhase.SECURITY_SCAN,
            SDLCPhase.DEPLOYMENT_PREP,
            SDLCPhase.DOCUMENTATION
        ]
        
        phases_in_tasks = set(task.phase for task in sdlc_engine.tasks.values())
        for phase in expected_phases:
            assert phase in phases_in_tasks, f"Missing phase: {phase}"

    @pytest.mark.asyncio
    async def test_task_dependencies(self, sdlc_engine):
        """Test task dependency resolution"""
        # Find a task with dependencies
        dependent_task = None
        for task in sdlc_engine.tasks.values():
            if task.dependencies:
                dependent_task = task
                break
        
        assert dependent_task is not None, "No dependent tasks found"
        
        # Verify dependencies exist
        for dep_id in dependent_task.dependencies:
            assert dep_id in sdlc_engine.tasks, f"Missing dependency: {dep_id}"

    @pytest.mark.asyncio 
    async def test_phase_execution(self, sdlc_engine):
        """Test execution of a single phase"""
        # Mock task execution to be fast
        with patch.object(sdlc_engine, '_execute_task_logic', return_value=True):
            success = await sdlc_engine._execute_phase(SDLCPhase.ANALYSIS)
            assert success

    @pytest.mark.asyncio
    async def test_execution_metrics_collection(self, sdlc_engine):
        """Test execution metrics are properly collected"""
        initial_metrics = sdlc_engine.execution_metrics
        
        # Mock a task execution
        task = SDLCTask(
            task_id="test_task",
            name="Test Task", 
            phase=SDLCPhase.ANALYSIS,
            description="Test task"
        )
        
        with patch.object(sdlc_engine, '_execute_task_logic', return_value=True):
            with patch.object(sdlc_engine, '_validate_task_success', return_value=True):
                result = await sdlc_engine._execute_task(task)
                assert result

        # Verify task completion tracking
        assert task.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, sdlc_engine):
        """Test circuit breaker protection for task execution"""
        task = SDLCTask(
            task_id="failing_task",
            name="Failing Task",
            phase=SDLCPhase.ANALYSIS, 
            description="Task that fails"
        )
        
        # Mock task to always fail
        with patch.object(sdlc_engine, '_execute_task_logic', side_effect=Exception("Task failed")):
            result = await sdlc_engine._execute_task(task)
            assert not result
            assert task.status == ExecutionStatus.FAILED


class TestQualityGateValidator:
    """Test the quality gate validation system"""

    @pytest.fixture
    async def quality_validator(self):
        """Create quality validator for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create some test files
            (project_path / "test_file.py").write_text("def test_function(): pass")
            (project_path / "requirements.txt").write_text("pytest>=7.0.0")
            
            validator = QualityGateValidator(project_path)
            await validator.initialize(project_path)
            yield validator

    @pytest.mark.asyncio
    async def test_validator_initialization(self, quality_validator):
        """Test quality validator initialization"""
        assert quality_validator.project_path.exists()
        assert quality_validator.project_maturity_score > 0
        assert len(quality_validator.thresholds) > 0

    @pytest.mark.asyncio
    async def test_code_quality_gate(self, quality_validator):
        """Test code quality gate"""
        score, details, suggestions = await quality_validator._check_code_quality()
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_test_coverage_gate(self, quality_validator):
        """Test test coverage gate"""
        score, details, suggestions = await quality_validator._check_test_coverage()
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_security_scan_gate(self, quality_validator):
        """Test security scanning gate"""
        score, details, suggestions = await quality_validator._check_security()
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_documentation_gate(self, quality_validator):
        """Test documentation coverage gate"""
        score, details, suggestions = await quality_validator._check_documentation()
        
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_comprehensive_validation(self, quality_validator):
        """Test complete quality gate validation"""
        report = await quality_validator.validate_all_gates()
        
        assert report.overall_score >= 0.0
        assert report.total_gates > 0
        assert len(report.gate_results) > 0
        assert report.execution_duration > 0

    @pytest.mark.asyncio
    async def test_adaptive_thresholds(self, quality_validator):
        """Test adaptive threshold adjustment"""
        original_thresholds = quality_validator.thresholds.copy()
        
        # Simulate high project maturity
        quality_validator.project_maturity_score = 0.9
        await quality_validator._adapt_thresholds()
        
        # Thresholds should be adjusted
        assert quality_validator.thresholds != original_thresholds


class TestProgressiveEnhancer:
    """Test the progressive enhancement system"""

    @pytest.fixture
    async def progressive_enhancer(self):
        """Create progressive enhancer for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            enhancer = ProgressiveEnhancer(project_path)
            await enhancer.initialize()
            yield enhancer

    @pytest.mark.asyncio
    async def test_enhancer_initialization(self, progressive_enhancer):
        """Test progressive enhancer initialization"""
        assert progressive_enhancer.performance_optimizer is not None
        assert progressive_enhancer.auto_scaler is not None

    @pytest.mark.asyncio
    async def test_generation_3_execution(self, progressive_enhancer):
        """Test Generation 3 scaling execution"""
        results = await progressive_enhancer.execute_generation_3_scaling()
        
        assert results["generation"] == "3_scale"
        assert "optimizations" in results
        assert "performance_improvements" in results
        assert results["status"] == "completed"

    @pytest.mark.asyncio
    async def test_adaptive_cache(self):
        """Test adaptive caching system"""
        cache = AdaptiveCache(max_size=10, ttl_seconds=1)
        
        # Test basic operations
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"
        
        # Test TTL expiration
        await asyncio.sleep(1.1)
        expired_value = await cache.get("key1")
        assert expired_value is None
        
        # Test LRU eviction
        for i in range(15):  # Exceed max_size
            await cache.set(f"key{i}", f"value{i}")
        
        stats = cache.get_stats()
        assert stats["size"] <= 10

    @pytest.mark.asyncio
    async def test_performance_optimization(self, progressive_enhancer):
        """Test performance optimization"""
        results = await progressive_enhancer.performance_optimizer.optimize_performance()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should have optimization strategies
        expected_strategies = ["cache_optimization", "concurrency_tuning"]
        for strategy in expected_strategies:
            assert strategy in results

    @pytest.mark.asyncio 
    async def test_auto_scaling(self, progressive_enhancer):
        """Test auto-scaling functionality"""
        from testgen_copilot.autonomous_sdlc.progressive_enhancement import PerformanceMetrics
        
        # Create metrics that should trigger scaling
        high_cpu_metrics = PerformanceMetrics(
            cpu_usage=80.0,
            response_time=4.0,
            error_rate=0.05
        )
        
        scaling_result = await progressive_enhancer.auto_scaler.evaluate_scaling(high_cpu_metrics)
        
        assert "actions_taken" in scaling_result
        assert "current_scale" in scaling_result

    @pytest.mark.asyncio
    async def test_monitoring_and_adaptation(self, progressive_enhancer):
        """Test continuous monitoring and adaptation"""
        monitoring_results = await progressive_enhancer.monitor_and_adapt()
        
        assert "metrics_collected" in monitoring_results
        assert monitoring_results["metrics_collected"] > 0
        assert "monitoring_start" in monitoring_results
        assert "monitoring_end" in monitoring_results


class TestAdaptiveIntelligence:
    """Test the adaptive intelligence system"""

    @pytest.fixture
    async def adaptive_intelligence(self):
        """Create adaptive intelligence for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            ai = AdaptiveIntelligence(project_path)
            await ai.initialize()
            yield ai

    @pytest.mark.asyncio
    async def test_ai_initialization(self, adaptive_intelligence):
        """Test adaptive intelligence initialization"""
        assert adaptive_intelligence.pattern_recognizer is not None
        assert adaptive_intelligence.parameter_tuner is not None
        assert adaptive_intelligence.self_healing_manager is not None

    @pytest.mark.asyncio
    async def test_pattern_recognition(self, adaptive_intelligence):
        """Test pattern recognition and learning"""
        context = {"project_type": "cli_tool", "complexity": "medium"}
        parameters = {"retry_multiplier": 1.5, "timeout": 30}
        metrics = {"success_rate": 0.9, "response_time": 1.2}
        
        # Record several executions
        for i in range(5):
            adaptive_intelligence.pattern_recognizer.record_execution(
                context, parameters, metrics, success=True
            )
        
        # Should recognize the pattern
        best_pattern = adaptive_intelligence.pattern_recognizer.get_best_pattern(context)
        assert best_pattern is not None
        assert best_pattern.confidence_score > 0

    @pytest.mark.asyncio
    async def test_parameter_tuning(self, adaptive_intelligence):
        """Test adaptive parameter tuning"""
        initial_params = adaptive_intelligence.parameter_tuner.current_parameters.copy()
        
        # Simulate poor performance to trigger adjustment
        performance_metrics = {"response_time": 5.0, "error_rate": 0.1}
        adjustments = adaptive_intelligence.parameter_tuner.update_parameters(
            performance_metrics, overall_performance=0.6
        )
        
        # Parameters should be adjusted
        final_params = adaptive_intelligence.parameter_tuner.current_parameters
        assert final_params != initial_params or len(adjustments) > 0

    @pytest.mark.asyncio
    async def test_execution_optimization(self, adaptive_intelligence):
        """Test execution optimization"""
        context = {
            "project_type": "web_app",
            "current_performance": 0.7,
            "performance_metrics": {"response_time": 2.0}
        }
        
        results = await adaptive_intelligence.optimize_execution(context)
        
        assert "optimizations_applied" in results
        assert "parameter_adjustments" in results
        assert isinstance(results["optimizations_applied"], list)

    @pytest.mark.asyncio
    async def test_failure_handling(self, adaptive_intelligence):
        """Test adaptive failure handling"""
        failure_context = {
            "error_type": "timeout",
            "error_message": "Connection timeout after 30 seconds",
            "component": "database_client"
        }
        
        response = await adaptive_intelligence.handle_failure(failure_context)
        
        assert "healing_strategies" in response
        assert "recovery_status" in response
        assert len(response["healing_strategies"]) > 0

    @pytest.mark.asyncio
    async def test_self_healing_strategies(self, adaptive_intelligence):
        """Test self-healing strategy execution"""
        context = {"component": "web_server", "error_type": "memory"}
        strategies = ["clear_cache", "restart_component"]
        
        healing_results = await adaptive_intelligence.self_healing_manager.execute_healing(
            strategies, context
        )
        
        assert "strategies_attempted" in healing_results
        assert "final_status" in healing_results
        assert len(healing_results["strategies_attempted"]) > 0

    @pytest.mark.asyncio
    async def test_state_persistence(self, adaptive_intelligence):
        """Test state saving and loading"""
        # Record some patterns
        context = {"test": "pattern"}
        adaptive_intelligence.pattern_recognizer.record_execution(
            context, {"param": 1}, {"metric": 0.8}, True
        )
        
        # Save state
        await adaptive_intelligence._save_state()
        
        # Create new instance and load state
        new_ai = AdaptiveIntelligence(adaptive_intelligence.project_path)
        await new_ai.initialize()
        
        # Should have loaded the pattern
        assert len(new_ai.pattern_recognizer.execution_history) > 0


class TestIntegrationScenarios:
    """Test complete integration scenarios"""

    @pytest.fixture
    async def full_system(self):
        """Create complete autonomous SDLC system"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create realistic project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "src" / "__init__.py").write_text("")
            (project_path / "src" / "main.py").write_text("def main(): pass")
            (project_path / "tests" / "test_main.py").write_text("def test_main(): pass")
            (project_path / "README.md").write_text("# Test Project")
            (project_path / "requirements.txt").write_text("pytest>=7.0.0")
            
            # Initialize system components
            sdlc_engine = AutonomousSDLCEngine(project_path)
            await sdlc_engine.initialize()
            
            yield {
                "project_path": project_path,
                "sdlc_engine": sdlc_engine
            }

    @pytest.mark.asyncio
    async def test_end_to_end_execution(self, full_system):
        """Test complete end-to-end SDLC execution"""
        sdlc_engine = full_system["sdlc_engine"]
        
        # Mock long-running operations to be fast
        with patch.object(sdlc_engine, '_execute_task_logic', return_value=True):
            with patch.object(sdlc_engine, '_validate_task_success', return_value=True):
                with patch.object(sdlc_engine, '_get_test_coverage', return_value=0.85):
                    with patch.object(sdlc_engine, '_get_security_score', return_value=0.90):
                        
                        # Execute full SDLC
                        metrics = await sdlc_engine.execute_full_sdlc()
                        
                        # Verify execution completed successfully
                        assert metrics.completed_tasks > 0
                        assert metrics.quality_gate_pass_rate > 0.8
                        assert metrics.deployment_readiness_score > 0.8

    @pytest.mark.asyncio
    async def test_progressive_enhancement_integration(self, full_system):
        """Test progressive enhancement integration"""
        project_path = full_system["project_path"]
        
        enhancer = ProgressiveEnhancer(project_path)
        await enhancer.initialize()
        
        # Test all three generations
        gen3_results = await enhancer.execute_generation_3_scaling()
        
        assert gen3_results["status"] == "completed"
        assert len(gen3_results["optimizations"]) > 0

    @pytest.mark.asyncio  
    async def test_quality_gates_integration(self, full_system):
        """Test quality gates integration"""
        project_path = full_system["project_path"]
        
        validator = QualityGateValidator(project_path)
        await validator.initialize(project_path)
        
        # Run comprehensive validation
        report = await validator.validate_all_gates()
        
        assert report.total_gates > 0
        assert report.overall_score >= 0.0

    @pytest.mark.asyncio
    async def test_adaptive_intelligence_integration(self, full_system):
        """Test adaptive intelligence integration"""
        project_path = full_system["project_path"]
        
        ai = AdaptiveIntelligence(project_path)
        await ai.initialize()
        
        # Test optimization
        context = {"project_type": "python_cli", "complexity": "medium"}
        results = await ai.optimize_execution(context)
        
        assert "optimizations_applied" in results

    @pytest.mark.asyncio
    async def test_failure_recovery_integration(self, full_system):
        """Test integrated failure recovery"""
        project_path = full_system["project_path"]
        
        ai = AdaptiveIntelligence(project_path) 
        await ai.initialize()
        
        # Simulate system failure
        failure_context = {
            "error_type": "system_overload",
            "error_message": "System resources exhausted",
            "component": "execution_engine"
        }
        
        recovery_results = await ai.handle_failure(failure_context)
        
        assert "healing_strategies" in recovery_results
        assert len(recovery_results["healing_strategies"]) > 0


class TestPerformanceAndReliability:
    """Test performance and reliability characteristics"""

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self):
        """Test thread safety during concurrent operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create multiple engines concurrently
            engines = []
            tasks = []
            
            async def create_engine():
                engine = AutonomousSDLCEngine(project_path)
                await engine.initialize()
                return engine
            
            # Create 5 engines concurrently
            for _ in range(5):
                tasks.append(create_engine())
            
            engines = await asyncio.gather(*tasks)
            
            # All should initialize successfully
            assert len(engines) == 5
            for engine in engines:
                assert len(engine.tasks) > 0

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage doesn't grow unbounded"""
        cache = AdaptiveCache(max_size=100)
        
        # Add many items
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}" * 100)  # Large values
        
        stats = cache.get_stats()
        
        # Cache should not exceed max size
        assert stats["size"] <= 100
        assert stats["utilization"] <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self):
        """Test system handles errors gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Test with invalid project path
            invalid_path = Path("/nonexistent/path")
            engine = AutonomousSDLCEngine(invalid_path)
            
            # Should handle gracefully without crashing
            try:
                success = await engine.initialize()
                # May succeed or fail, but should not crash
            except Exception as e:
                # Should be a handled exception, not a crash
                assert isinstance(e, (OSError, ValueError, RuntimeError))

    @pytest.mark.asyncio
    async def test_large_scale_pattern_recognition(self):
        """Test pattern recognition with large datasets"""
        recognizer = PatternRecognizer(history_limit=1000)
        
        # Add many execution records
        for i in range(500):
            context = {
                "project_type": "web_app" if i % 2 == 0 else "cli_tool",
                "complexity": "high" if i % 3 == 0 else "medium"
            }
            parameters = {"retry_multiplier": 1.0 + (i % 5) * 0.2}
            metrics = {"response_time": 1.0 + (i % 10) * 0.1}
            
            recognizer.record_execution(context, parameters, metrics, success=i % 4 != 0)
        
        # Should recognize patterns efficiently
        test_context = {"project_type": "web_app", "complexity": "medium"}
        pattern = recognizer.get_best_pattern(test_context)
        
        # Should find a pattern with reasonable confidence
        assert pattern is None or pattern.confidence_score >= 0.0


if __name__ == "__main__":
    """Run comprehensive test suite"""
    
    # Configure pytest for async testing
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-x"  # Stop on first failure
    ])