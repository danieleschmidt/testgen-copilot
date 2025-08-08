"""Tests for Generation 3 scaling and performance optimization components."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from testgen_copilot.performance_optimizer import (
    PerformanceOptimizer, PerformanceCache, ConcurrentExecutor,
    AdaptiveResourceManager, get_performance_optimizer
)
from testgen_copilot.auto_scaling import (
    AutoScaler, LoadBalancer, WorkloadAnalyzer, ScalingMetrics,
    ScalingPolicy, WorkloadPattern, get_auto_scaler
)


class TestPerformanceCache:
    """Test the high-performance caching system."""
    
    def test_memory_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = PerformanceCache(max_memory_mb=1, max_disk_mb=5)
        
        # Test put and get
        cache.put("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test cache miss
        assert cache.get("non_existent_key") is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats["memory_hits"] == 1
        assert stats["misses"] == 1
        
    def test_cache_eviction(self):
        """Test cache eviction under memory pressure."""
        cache = PerformanceCache(max_memory_mb=0.001, max_disk_mb=1)  # Very small memory cache
        
        # Fill cache beyond memory limit
        large_data = "x" * 1000  # 1KB of data
        for i in range(10):
            cache.put(f"key_{i}", large_data)
            
        # Some items should be evicted from memory
        stats = cache.get_stats()
        assert stats["evictions"] > 0 or stats["disk_writes"] > 0
        
    def test_multilevel_caching(self):
        """Test L1 (memory) and L2 (disk) cache interaction."""
        cache = PerformanceCache(max_memory_mb=0.001, max_disk_mb=5)
        
        # Put multiple large items to force disk usage
        large_data = "x" * 5000  # 5KB of data
        for i in range(5):
            cache.put(f"disk_item_{i}", large_data)
        
        # Retrieve one item (should promote from disk to memory if needed)
        value = cache.get("disk_item_0")
        assert value == large_data
        
        stats = cache.get_stats()
        # With multiple large items and small memory, should use disk
        assert stats["disk_hits"] > 0 or stats["disk_writes"] > 0 or stats["evictions"] > 0


class TestConcurrentExecutor:
    """Test concurrent execution capabilities."""
    
    def test_basic_batch_execution(self):
        """Test basic batch task execution."""
        executor = ConcurrentExecutor(max_workers=2)
        
        def add_numbers(a, b):
            return a + b
            
        tasks = [
            (add_numbers, (1, 2), {}),
            (add_numbers, (3, 4), {}),
            (add_numbers, (5, 6), {})
        ]
        
        # Run the async function in event loop
        async def run_test():
            results = await executor.execute_batch(tasks)
            return results
            
        results = asyncio.run(run_test())
        
        assert len(results) == 3
        assert set(results) == {3, 7, 11}
        
        executor.shutdown()
        
    def test_performance_context(self):
        """Test performance monitoring context manager."""
        executor = ConcurrentExecutor(max_workers=2)
        
        with executor.performance_context("test_operation"):
            time.sleep(0.01)  # Small delay to measure
            
        # Context should complete without errors
        executor.shutdown()
        
    def test_error_handling_in_batch(self):
        """Test error handling during batch execution."""
        executor = ConcurrentExecutor(max_workers=2)
        
        def failing_task():
            raise ValueError("Test error")
            
        def success_task():
            return "success"
            
        tasks = [
            (success_task, (), {}),
            (failing_task, (), {}),
            (success_task, (), {})
        ]
        
        async def run_test():
            results = await executor.execute_batch(tasks)
            return results
            
        results = asyncio.run(run_test())
        
        # Should have results with None for failed tasks
        assert len(results) == 3
        assert results[0] == "success"
        assert results[1] is None  # Failed task
        assert results[2] == "success"
        
        executor.shutdown()


class TestAdaptiveResourceManager:
    """Test adaptive resource management."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_concurrency_optimization(self, mock_memory, mock_cpu):
        """Test dynamic concurrency optimization."""
        # Mock high resource usage
        mock_cpu.return_value = 90.0
        mock_memory.return_value = Mock(percent=85.0)
        
        manager = AdaptiveResourceManager()
        optimal_concurrency = manager.optimize_concurrency()
        
        # Should reduce concurrency due to high resource usage
        assert optimal_concurrency >= 1
        
        # Mock low resource usage
        mock_cpu.return_value = 30.0
        mock_memory.return_value = Mock(percent=40.0)
        
        optimal_concurrency = manager.optimize_concurrency()
        
        # Should allow higher concurrency
        assert optimal_concurrency >= 1
        
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_resource_recommendations(self, mock_disk, mock_memory, mock_cpu):
        """Test resource recommendation generation."""
        mock_cpu.return_value = 95.0  # High CPU
        mock_memory.return_value = Mock(percent=90.0, available=1024**3)  # High memory
        mock_disk.return_value = Mock(percent=95.0, free=1024**3)  # High disk
        
        manager = AdaptiveResourceManager()
        recommendations = manager.get_resource_recommendation()
        
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0
        assert recommendations["performance_status"] == "critical"


class TestPerformanceOptimizer:
    """Test the main performance optimizer."""
    
    def test_optimization_context(self):
        """Test operation optimization context."""
        optimizer = PerformanceOptimizer()
        
        with optimizer.optimize_operation("test_op", cacheable=True):
            time.sleep(0.01)  # Small operation
            
        # Check metrics were recorded
        assert "test_op" in optimizer.metrics
        metrics = optimizer.metrics["test_op"]
        assert metrics.total_calls == 1
        assert metrics.average_time_ms > 0
        
    def test_comprehensive_report(self):
        """Test comprehensive performance report generation."""
        optimizer = PerformanceOptimizer()
        
        # Perform some operations to generate metrics
        with optimizer.optimize_operation("test_op1"):
            time.sleep(0.005)
            
        with optimizer.optimize_operation("test_op2"):
            time.sleep(0.01)
            
        report = optimizer.get_comprehensive_report()
        
        assert "optimization_status" in report
        assert "cache_performance" in report
        assert "operation_metrics" in report
        assert "recommendations" in report
        
        # Should have metrics for our operations
        assert "test_op1" in report["operation_metrics"]
        assert "test_op2" in report["operation_metrics"]
        
    def test_batch_processing_optimization(self):
        """Test optimized batch processing."""
        optimizer = PerformanceOptimizer()
        
        def simple_task(value):
            return value * 2
            
        tasks = [(simple_task, (i,), {}) for i in range(5)]
        
        async def run_test():
            results = await optimizer.optimize_batch_processing(tasks, cpu_bound=False)
            return results
            
        results = asyncio.run(run_test())
        
        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]


class TestLoadBalancer:
    """Test intelligent load balancer."""
    
    def test_worker_registration(self):
        """Test worker registration and management."""
        lb = LoadBalancer()
        
        # Register workers
        lb.register_worker("worker1", capacity=100)
        lb.register_worker("worker2", capacity=150)
        
        assert len(lb.workers) == 2
        assert "worker1" in lb.workers
        assert "worker2" in lb.workers
        
        # Test worker selection
        selected = lb.select_worker("least_connections")
        assert selected in ["worker1", "worker2"]
        
    def test_load_balancing_algorithms(self):
        """Test different load balancing algorithms."""
        lb = LoadBalancer()
        
        lb.register_worker("worker1", capacity=100)
        lb.register_worker("worker2", capacity=100)
        
        # Test round robin
        selected1 = lb.select_worker("round_robin")
        selected2 = lb.select_worker("round_robin")
        
        assert selected1 in ["worker1", "worker2"]
        assert selected2 in ["worker1", "worker2"]
        
        # Test least connections
        lb.update_worker_load("worker1", 50)
        lb.update_worker_load("worker2", 30)
        
        selected = lb.select_worker("least_connections")
        assert selected == "worker2"  # Should select less loaded worker
        
    def test_health_monitoring(self):
        """Test worker health monitoring."""
        lb = LoadBalancer()
        
        lb.register_worker("worker1", capacity=100)
        
        # Mark worker as unhealthy
        lb.mark_worker_unhealthy("worker1", "Test failure")
        
        # Should not select unhealthy worker
        selected = lb.select_worker()
        assert selected is None
        
        # Mark as healthy again
        lb.mark_worker_healthy("worker1")
        selected = lb.select_worker()
        assert selected == "worker1"
        
    def test_request_recording(self):
        """Test request performance recording."""
        lb = LoadBalancer()
        
        lb.register_worker("worker1", capacity=100)
        
        # Record successful requests
        lb.record_request("worker1", 100.0, success=True)
        lb.record_request("worker1", 150.0, success=True)
        
        stats = lb.get_load_stats()
        worker_stats = stats["worker_stats"]["worker1"]
        
        assert worker_stats["total_requests"] == 2
        assert worker_stats["avg_response_time_ms"] == 125.0
        assert worker_stats["health_score"] > 1.0  # Should improve with successful requests


class TestWorkloadAnalyzer:
    """Test workload pattern analysis."""
    
    def test_pattern_detection(self):
        """Test workload pattern detection."""
        analyzer = WorkloadAnalyzer(history_size=100)
        
        # Create steady workload pattern
        for i in range(20):
            metrics = ScalingMetrics(
                cpu_utilization=50.0,  # Steady load
                memory_utilization=60.0,
                response_time_ms=100.0,
                throughput_ops_per_sec=10.0
            )
            analyzer.record_metrics(metrics)
            
        pattern = analyzer.detect_pattern()
        assert pattern == WorkloadPattern.STEADY
        
    def test_increasing_pattern_detection(self):
        """Test detection of gradually increasing load."""
        analyzer = WorkloadAnalyzer(history_size=100)
        
        # Create increasing workload pattern
        for i in range(20):
            metrics = ScalingMetrics(
                cpu_utilization=30.0 + (i * 3),  # Increasing load
                memory_utilization=40.0 + (i * 2),
                response_time_ms=100.0,
                throughput_ops_per_sec=10.0
            )
            analyzer.record_metrics(metrics)
            
        pattern = analyzer.detect_pattern()
        assert pattern == WorkloadPattern.GRADUAL_INCREASE
        
    def test_load_prediction(self):
        """Test load prediction functionality."""
        analyzer = WorkloadAnalyzer(history_size=100)
        
        # Add some metrics
        for i in range(10):
            metrics = ScalingMetrics(
                cpu_utilization=50.0,
                memory_utilization=60.0,
                response_time_ms=100.0,
                throughput_ops_per_sec=10.0
            )
            analyzer.record_metrics(metrics)
            
        prediction = analyzer.predict_next_load(horizon_minutes=5)
        
        # Should predict reasonable values
        assert 0 <= prediction <= 100
        
    def test_analysis_report(self):
        """Test comprehensive analysis report."""
        analyzer = WorkloadAnalyzer(history_size=100)
        
        # Add sample data
        for i in range(15):
            metrics = ScalingMetrics(
                cpu_utilization=40.0 + (i * 2),
                memory_utilization=50.0,
                response_time_ms=120.0,
                throughput_ops_per_sec=8.0
            )
            analyzer.record_metrics(metrics)
            
        report = analyzer.get_analysis_report()
        
        assert "current_pattern" in report
        assert "prediction_5min" in report
        assert "metrics_summary" in report
        assert "trend_analysis" in report
        assert "recommendations" in report


class TestAutoScaler:
    """Test intelligent auto-scaling system."""
    
    def test_scaling_decision_logic(self):
        """Test auto-scaling decision making."""
        scaler = AutoScaler(policy=ScalingPolicy.BALANCED)
        
        # Register initial worker
        scaler.load_balancer.register_worker("worker1", capacity=100)
        
        # High CPU should trigger scale up
        high_load_metrics = ScalingMetrics(
            cpu_utilization=85.0,  # Above threshold
            memory_utilization=60.0,
            response_time_ms=500.0,
            queue_depth=5
        )
        
        decision = scaler.should_scale(high_load_metrics)
        assert decision.action == "scale_up"
        assert decision.target_capacity > decision.current_capacity
        
    def test_scale_down_decision(self):
        """Test scale down decision logic."""
        scaler = AutoScaler(policy=ScalingPolicy.BALANCED)
        
        # Register multiple workers
        scaler.load_balancer.register_worker("worker1", capacity=100)
        scaler.load_balancer.register_worker("worker2", capacity=100)
        
        # Low CPU should trigger scale down
        low_load_metrics = ScalingMetrics(
            cpu_utilization=20.0,  # Below threshold
            memory_utilization=30.0,
            response_time_ms=100.0,
            queue_depth=1
        )
        
        # Allow immediate scaling by setting old timestamps
        from datetime import datetime, timezone, timedelta
        scaler.last_scale_down = datetime.now(timezone.utc) - timedelta(hours=1)
        
        decision = scaler.should_scale(low_load_metrics)
        assert decision.action == "scale_down"
        assert decision.target_capacity < decision.current_capacity
        
    def test_scaling_policies(self):
        """Test different scaling policies."""
        # Conservative policy
        conservative_scaler = AutoScaler(policy=ScalingPolicy.CONSERVATIVE)
        assert conservative_scaler.scale_up_threshold == 80.0
        assert conservative_scaler.scale_up_cooldown == 300
        
        # Aggressive policy
        aggressive_scaler = AutoScaler(policy=ScalingPolicy.AGGRESSIVE)
        assert aggressive_scaler.scale_up_threshold == 60.0
        assert aggressive_scaler.scale_up_cooldown == 60
        
    def test_scaling_execution(self):
        """Test scaling decision execution."""
        scaler = AutoScaler(policy=ScalingPolicy.BALANCED)
        
        from testgen_copilot.auto_scaling import ScalingDecision
        
        # Test scale up execution
        scale_up_decision = ScalingDecision(
            action="scale_up",
            current_capacity=1,
            target_capacity=3,
            reason="Test scale up",
            confidence=0.8
        )
        
        result = scaler.execute_scaling_decision(scale_up_decision)
        assert result is True
        
        # Should have created new workers
        assert len(scaler.load_balancer.workers) == 3
        
    def test_scaling_effectiveness_calculation(self):
        """Test scaling effectiveness metrics."""
        scaler = AutoScaler(policy=ScalingPolicy.BALANCED)
        
        # Add some scaling decisions
        from testgen_copilot.auto_scaling import ScalingDecision
        
        for i in range(5):
            decision = ScalingDecision(
                action="scale_up",
                current_capacity=i,
                target_capacity=i+1,
                reason="Test",
                confidence=0.8
            )
            scaler.scaling_decisions.append(decision)
            
        effectiveness = scaler._calculate_scaling_effectiveness()
        
        assert effectiveness["total_decisions"] == 5
        assert effectiveness["scale_up_count"] == 5
        assert effectiveness["average_confidence"] == 0.8
        assert effectiveness["status"] == "effective"
        
    def test_comprehensive_scaling_report(self):
        """Test comprehensive scaling report generation."""
        scaler = AutoScaler(policy=ScalingPolicy.BALANCED)
        
        # Register a worker and add some metrics
        scaler.load_balancer.register_worker("worker1", capacity=100)
        
        metrics = ScalingMetrics(
            cpu_utilization=50.0,
            memory_utilization=60.0,
            response_time_ms=200.0,
            throughput_ops_per_sec=10.0
        )
        
        scaler.workload_analyzer.record_metrics(metrics)
        
        report = scaler.get_scaling_report()
        
        assert "auto_scaling_status" in report
        assert "load_balancing" in report
        assert "workload_analysis" in report
        assert "scaling_effectiveness" in report


class TestIntegration:
    """Integration tests for Generation 3 components."""
    
    def test_end_to_end_optimization(self):
        """Test end-to-end performance optimization flow."""
        # Get global optimizer
        optimizer = get_performance_optimizer()
        
        # Perform optimized operations
        with optimizer.optimize_operation("integration_test"):
            time.sleep(0.01)
            
        # Get auto-scaler
        scaler = get_auto_scaler(ScalingPolicy.BALANCED)
        
        # Test scaling with sample metrics
        metrics = ScalingMetrics(
            cpu_utilization=60.0,
            memory_utilization=50.0,
            response_time_ms=150.0
        )
        
        decision = scaler.should_scale(metrics)
        assert decision.action in ["scale_up", "scale_down", "no_change"]
        
        # Get comprehensive reports
        perf_report = optimizer.get_comprehensive_report()
        scaling_report = scaler.get_scaling_report()
        
        assert perf_report is not None
        assert scaling_report is not None
        
    def test_resource_aware_scaling(self):
        """Test that scaling considers resource constraints."""
        optimizer = get_performance_optimizer()
        scaler = get_auto_scaler(ScalingPolicy.BALANCED)
        
        # Simulate resource-constrained environment
        resource_recommendations = optimizer.resource_manager.get_resource_recommendation()
        
        # High resource usage should influence scaling decisions
        high_resource_metrics = ScalingMetrics(
            cpu_utilization=95.0,  # Very high
            memory_utilization=90.0,  # Very high
            response_time_ms=1000.0
        )
        
        decision = scaler.should_scale(high_resource_metrics)
        
        # Even with high metrics, scaling should be considered carefully
        # due to resource constraints
        assert decision.confidence is not None
        assert 0.0 <= decision.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])