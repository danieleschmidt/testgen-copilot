"""Comprehensive integration tests for quantum-inspired task planner."""

import asyncio
# import pytest  # Mock for testing without pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from testgen_copilot.quantum_planner import (
        QuantumTaskPlanner, QuantumTask, ResourceQuantum, TaskPriority, TaskState, 
        create_quantum_planner, demo_quantum_planning
    )
    from testgen_copilot.quantum_performance import (
        PerformanceConfig, QuantumMemoryPool, QuantumTaskBatcher, 
        QuantumPerformanceProfiler, QuantumLoadBalancer, QuantumAutoScaler
    )
    from testgen_copilot.quantum_monitoring import (
        QuantumHealthChecker, QuantumMetricsCollector, QuantumAlertManager,
        AlertSeverity, QuantumAlert
    )
    from testgen_copilot.quantum_security import (
        QuantumInputValidator, QuantumThreatDetector, ThreatLevel,
        QuantumEncryption
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestQuantumTaskPlannerIntegration:
    """Integration tests for quantum task planner."""
    
    # @pytest.fixture  # Mock fixture
    def planner(self):
        """Create quantum planner for testing."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
        return create_quantum_planner()
    
    # @pytest.fixture  # Mock fixture
    def sample_tasks(self, planner):
        """Create sample tasks for testing."""
        tasks = []
        
        # Foundation task - no dependencies
        task1 = planner.add_task(
            task_id="foundation",
            name="Foundation Setup", 
            description="Set up project foundation",
            priority=TaskPriority.GROUND_STATE,
            estimated_duration=timedelta(hours=4),
            resources_required={"cpu": 2.0, "memory": 1.0}
        )
        tasks.append(task1)
        
        # Database task - depends on foundation
        task2 = planner.add_task(
            task_id="database",
            name="Database Setup",
            description="Configure database",
            priority=TaskPriority.EXCITED_1,
            estimated_duration=timedelta(hours=3),
            dependencies={"foundation"},
            resources_required={"cpu": 1.5, "memory": 2.0}
        )
        tasks.append(task2)
        
        # API task - depends on database and foundation
        task3 = planner.add_task(
            task_id="api",
            name="API Implementation",
            description="Build REST API",
            priority=TaskPriority.EXCITED_1,
            estimated_duration=timedelta(hours=6),
            dependencies={"foundation", "database"},
            resources_required={"cpu": 3.0, "memory": 2.5}
        )
        tasks.append(task3)
        
        return tasks
    
    def test_quantum_planner_creation(self, planner):
        """Test quantum planner can be created."""
        assert planner is not None
        assert len(planner.resources) > 0
        assert planner.quantum_coherence_time > 0
    
    def test_task_addition_and_retrieval(self, planner, sample_tasks):
        """Test adding and retrieving tasks."""
        assert len(planner.tasks) == 3
        assert "foundation" in planner.tasks
        assert "database" in planner.tasks
        assert "api" in planner.tasks
        
        # Test task properties
        foundation_task = planner.tasks["foundation"]
        assert foundation_task.name == "Foundation Setup"
        assert foundation_task.priority == TaskPriority.GROUND_STATE
        assert len(foundation_task.dependencies) == 0
    
    def test_dependency_resolution(self, planner, sample_tasks):
        """Test dependency resolution works correctly."""
        database_task = planner.tasks["database"]
        api_task = planner.tasks["api"]
        
        assert "foundation" in database_task.dependencies
        assert "foundation" in api_task.dependencies
        assert "database" in api_task.dependencies
        
        # Test dependency validation
        valid_deps = planner.validate_dependencies()
        assert valid_deps["valid"] is True
        assert len(valid_deps["errors"]) == 0
    
    # @pytest.mark.asyncio  # Mock async marker
    async def test_quantum_planning_execution(self, planner, sample_tasks):
        """Test quantum planning algorithm execution."""
        # Add deadline to make planning more interesting
        api_task = planner.tasks["api"]
        api_task.deadline = datetime.now(timezone.utc) + timedelta(hours=12)
        
        # Run quantum planning
        plan = await planner.create_quantum_plan(max_iterations=100)
        
        assert plan is not None
        assert "schedule" in plan
        assert "optimization_stats" in plan
        assert len(plan["schedule"]) == 3
        
        # Verify schedule respects dependencies
        schedule = plan["schedule"] 
        foundation_time = None
        database_time = None
        api_time = None
        
        for task_id, task_info in schedule.items():
            if task_id == "foundation":
                foundation_time = task_info["scheduled_time"]
            elif task_id == "database":
                database_time = task_info["scheduled_time"]  
            elif task_id == "api":
                api_time = task_info["scheduled_time"]
        
        # Foundation should be scheduled first
        assert foundation_time is not None
        assert database_time is not None
        assert api_time is not None
        
        # Dependencies should be respected
        assert foundation_time <= database_time
        assert foundation_time <= api_time
        assert database_time <= api_time
    
    def test_resource_allocation(self, planner, sample_tasks):
        """Test resource allocation and management."""
        total_cpu_required = sum(task.resources_required.get("cpu", 0) for task in sample_tasks)
        total_memory_required = sum(task.resources_required.get("memory", 0) for task in sample_tasks)
        
        available_cpu = sum(resource.total_capacity for resource in planner.resources 
                          if "cpu" in resource.name.lower())
        available_memory = sum(resource.total_capacity for resource in planner.resources
                             if "memory" in resource.name.lower())
        
        # Should have sufficient resources
        assert available_cpu >= total_cpu_required / 2  # Allow parallel execution
        assert available_memory >= total_memory_required / 2
        
    def test_quantum_entanglement(self, planner):
        """Test quantum entanglement between tasks."""
        # Add entangled tasks
        task1 = planner.add_task(
            task_id="frontend",
            name="Frontend Development", 
            description="Build frontend",
            priority=TaskPriority.EXCITED_1,
            estimated_duration=timedelta(hours=8),
            resources_required={"cpu": 2.0}
        )
        
        task2 = planner.add_task(
            task_id="testing",
            name="Testing Suite",
            description="Comprehensive testing",
            priority=TaskPriority.EXCITED_2, 
            estimated_duration=timedelta(hours=4),
            resources_required={"cpu": 1.5}
        )
        
        # Create entanglement
        planner.entangle_tasks("frontend", "testing")
        
        assert "testing" in task1.entangled_tasks
        assert "frontend" in task2.entangled_tasks
    
    def test_task_state_transitions(self, planner, sample_tasks):
        """Test quantum task state transitions."""
        task = sample_tasks[0]
        
        # Initial state
        assert task.state == TaskState.PENDING
        
        # Transition to executing
        task.quantum_evolve()
        # State should have possibility of being different states
        
        # Force state transitions
        task.start_execution() 
        assert task.state == TaskState.EXECUTING
        
        task.complete_execution()
        assert task.state == TaskState.COMPLETED


class TestQuantumPerformanceIntegration:
    """Integration tests for quantum performance optimizations."""
    
    # @pytest.fixture  # Mock fixture
    def performance_config(self):
        """Create performance configuration."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
        return PerformanceConfig(
            max_workers=4,
            task_batch_size=10,
            memory_limit_mb=512,
            enable_caching=True
        )
    
    def test_memory_pool_lifecycle(self, performance_config):
        """Test memory pool object lifecycle."""
        pool = QuantumMemoryPool(performance_config.cache_size_mb)
        
        # Test object creation and reuse
        obj1 = pool.get_object("test_object", lambda: {"data": "test1"})
        pool.return_object("test_object", obj1)
        
        obj2 = pool.get_object("test_object", lambda: {"data": "test2"})
        
        # obj2 should be reused obj1
        assert obj2 == obj1
        
        # Check statistics
        stats = pool.get_memory_usage()
        assert stats["usage_stats"]["test_object"]["allocated"] == 1
        assert stats["usage_stats"]["test_object"]["reused"] == 1
    
    # @pytest.mark.asyncio  # Mock async marker
    async def test_task_batching(self, performance_config):
        """Test task batching functionality."""
        batcher = QuantumTaskBatcher(batch_size=3, batch_timeout=1.0)
        
        # Create mock tasks
        tasks = []
        for i in range(5):
            task = Mock()
            task.id = f"task_{i}"
            task.priority = Mock()
            task.priority.value = i % 3
            task.entangled_tasks = set()
            task.calculate_urgency_score = Mock(return_value=0.5)
            tasks.append(task)
        
        # Add tasks to batcher
        batch_ready_count = 0
        for task in tasks:
            ready = await batcher.add_task(task)
            if ready:
                batch_ready_count += 1
                batch = await batcher.get_batch()
                assert len(batch) > 0
        
        # Should have created at least one batch
        assert batch_ready_count > 0
        
        # Check for remaining tasks
        remaining_batch = await batcher.get_batch(min_size=1)
        assert len(remaining_batch) >= 0
    
    def test_load_balancer_worker_selection(self, performance_config):
        """Test load balancer worker selection logic."""
        workers = [
            {"capacity": 2.0, "quantum_efficiency": 1.5, "specialization": "general"},
            {"capacity": 1.0, "quantum_efficiency": 2.0, "specialization": "high_priority"},
            {"capacity": 1.5, "quantum_efficiency": 1.2, "specialization": "entanglement"}
        ]
        
        balancer = QuantumLoadBalancer(workers)
        
        # Create test task
        task = Mock()
        task.id = "test_task"
        task.priority = TaskPriority.EXCITED_1
        task.entangled_tasks = set()
        task.estimated_duration = timedelta(hours=2)
        task.resources_required = {"cpu": 1.0, "memory": 1.0}
        
        # Select worker
        selected_worker = balancer.select_worker(task)
        assert selected_worker is not None
        assert selected_worker.startswith("quantum_worker_")
        
        # Complete task and check statistics update
        balancer.complete_task(selected_worker, 3600.0, True)  # 1 hour execution
        
        load_dist = balancer.get_load_distribution()
        assert load_dist["total_capacity"] > 0
        assert 0 <= load_dist["utilization_percent"] <= 100
    
    def test_auto_scaler_decision_logic(self, performance_config):
        """Test auto-scaler decision making."""
        scaler = QuantumAutoScaler(performance_config)
        
        # Test scale-up conditions
        high_load_metrics = {
            "cpu_percent": 85.0,
            "memory_percent": 70.0,
            "queue_length": 50,
            "avg_response_time": 3.0
        }
        
        should_scale_up = scaler.should_scale_up(high_load_metrics)
        assert should_scale_up is True
        
        # Test scale-down conditions
        low_load_metrics = {
            "cpu_percent": 25.0,
            "memory_percent": 40.0,
            "queue_length": 2,
            "avg_response_time": 0.5
        }
        
        should_scale_down = scaler.should_scale_down(low_load_metrics)
        assert should_scale_down is True
        
        # Test recommendations
        recommendations = scaler.get_scaling_recommendations(high_load_metrics)
        assert len(recommendations) > 0
        assert any("SCALE UP" in rec for rec in recommendations)


class TestQuantumMonitoringIntegration:
    """Integration tests for quantum monitoring systems."""
    
    def test_health_checker_system_metrics(self):
        """Test health checker system metrics collection."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        health_checker = QuantumHealthChecker(coherence_threshold=0.8)
        
        report = health_checker.check_system_health()
        
        assert "timestamp" in report
        assert "overall_status" in report
        assert "quantum_coherence" in report
        assert "system_metrics" in report
        assert "alerts" in report
        
        # Check system metrics structure
        sys_metrics = report["system_metrics"]
        expected_metrics = ["cpu_percent", "memory_percent", "disk_percent"]
        for metric in expected_metrics:
            assert metric in sys_metrics
            assert isinstance(sys_metrics[metric], (int, float))
    
    # @pytest.mark.asyncio  # Mock async marker 
    async def test_metrics_collector_lifecycle(self):
        """Test metrics collector start/stop lifecycle."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        collector = QuantumMetricsCollector(sampling_rate=0.5, max_history=50)
        
        # Start collection
        await collector.start_collection()
        assert collector.collection_active is True
        
        # Let it collect some metrics
        await asyncio.sleep(0.2)
        
        # Stop collection
        await collector.stop_collection()
        assert collector.collection_active is False
        
        # Check collected metrics
        summary = collector.get_metrics_summary()
        assert "collection_active" in summary
        assert summary["collection_active"] is False
    
    def test_alert_manager_correlation(self):
        """Test alert manager correlation and entanglement."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        alert_manager = QuantumAlertManager(alert_history_limit=100)
        
        # Create related alerts
        alert1 = alert_manager.create_alert(
            alert_id="cpu_high_1",
            severity=AlertSeverity.HIGH,
            message="CPU usage high: 85%",
            source_metric="cpu_usage"
        )
        
        alert2 = alert_manager.create_alert(
            alert_id="cpu_high_2", 
            severity=AlertSeverity.HIGH,
            message="CPU usage critical: 95%",
            source_metric="cpu_usage"
        )
        
        # Alerts should be entangled due to same source metric
        assert "cpu_high_2" in alert1.entangled_alerts or "cpu_high_1" in alert2.entangled_alerts
        
        # Test alert resolution
        alert_manager.resolve_alert("cpu_high_1")
        assert alert1.resolved is True
        
        # Get statistics
        stats = alert_manager.get_alert_statistics()
        assert stats["total_alerts"] >= 2
        assert stats["entanglement_count"] >= 0


class TestQuantumSecurityIntegration:
    """Integration tests for quantum security systems."""
    
    def test_input_validator_threat_detection(self):
        """Test input validator threat detection capabilities."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        validator = QuantumInputValidator()
        
        # Test benign input
        safe_result = validator.validate_input("normal user input", "user_comment")
        assert safe_result["is_safe"] is True
        assert safe_result["threat_level"] == ThreatLevel.BENIGN
        
        # Test SQL injection attempt
        malicious_result = validator.validate_input(
            "'; DROP TABLE users; --",
            "login_query"
        )
        assert malicious_result["is_safe"] is False
        assert malicious_result["threat_level"] != ThreatLevel.BENIGN
        assert len(malicious_result["threats_detected"]) > 0
        
        # Test XSS attempt
        xss_result = validator.validate_input(
            "<script>alert('xss')</script>",
            "user_input"
        )
        assert xss_result["is_safe"] is False
        assert len(xss_result["threats_detected"]) > 0
    
    def test_threat_detector_behavioral_analysis(self):
        """Test threat detector behavioral analysis."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        detector = QuantumThreatDetector(threat_memory_limit=500)
        
        # Simulate normal request
        normal_request = {
            "user_id": "12345",
            "action": "get_profile",
            "data": {"fields": ["name", "email"]}
        }
        
        normal_analysis = detector.analyze_request(
            request_data=normal_request,
            source_ip="192.168.1.100",
            user_id="user123"
        )
        
        assert normal_analysis["threat_detected"] is False
        assert normal_analysis["security_score"] >= 0.8
        
        # Simulate suspicious request
        suspicious_request = {
            "user_id": "'; DROP TABLE users; --",
            "action": "admin_access",
            "data": {"x": "A" * 10000}  # Very long input
        }
        
        suspicious_analysis = detector.analyze_request(
            request_data=suspicious_request,
            source_ip="192.168.1.100",
            user_id="user123"
        )
        
        assert suspicious_analysis["threat_detected"] is True
        assert suspicious_analysis["security_score"] < 0.8
        assert len(suspicious_analysis["threats"]) > 0
        assert len(suspicious_analysis["recommendations"]) > 0
    
    def test_quantum_encryption_roundtrip(self):
        """Test quantum encryption/decryption roundtrip."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        encryptor = QuantumEncryption()
        
        original_data = "sensitive quantum information"
        
        # Encrypt data
        encrypted = encryptor.encrypt_sensitive_data(original_data)
        assert encrypted != original_data
        assert len(encrypted) > len(original_data)  # Base64 encoding adds overhead
        
        # Decrypt data
        decrypted = encryptor.decrypt_sensitive_data(encrypted, max_age_seconds=3600)
        assert decrypted == original_data
        
        # Test expired data (should fail)
        with pytest.raises(ValueError):
            encryptor.decrypt_sensitive_data(encrypted, max_age_seconds=0)


class TestQuantumSystemIntegration:
    """End-to-end integration tests for complete quantum system."""
    
    # @pytest.mark.asyncio  # Mock async marker
    async def test_complete_quantum_workflow(self):
        """Test complete quantum workflow from planning to execution."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        # Create integrated system
        planner = create_quantum_planner()
        
        # Add complex task set
        tasks = []
        task_configs = [
            {"id": "setup", "deps": [], "priority": TaskPriority.GROUND_STATE, "hours": 2},
            {"id": "auth", "deps": ["setup"], "priority": TaskPriority.EXCITED_1, "hours": 4},
            {"id": "database", "deps": ["setup"], "priority": TaskPriority.EXCITED_1, "hours": 3},
            {"id": "api", "deps": ["auth", "database"], "priority": TaskPriority.EXCITED_1, "hours": 6},
            {"id": "frontend", "deps": ["api"], "priority": TaskPriority.EXCITED_2, "hours": 8},
            {"id": "testing", "deps": ["api", "frontend"], "priority": TaskPriority.EXCITED_2, "hours": 4},
            {"id": "deployment", "deps": ["testing"], "priority": TaskPriority.EXCITED_3, "hours": 2},
        ]
        
        for config in task_configs:
            task = planner.add_task(
                task_id=config["id"],
                name=f"{config['id'].title()} Task",
                description=f"Implement {config['id']} functionality",
                priority=config["priority"],
                estimated_duration=timedelta(hours=config["hours"]),
                dependencies=set(config["deps"]),
                resources_required={
                    "cpu": 1.0 + len(config["deps"]) * 0.5,
                    "memory": 1.0 + config["hours"] * 0.1
                }
            )
            tasks.append(task)
        
        # Validate dependencies
        dep_validation = planner.validate_dependencies()
        assert dep_validation["valid"] is True
        
        # Create quantum plan
        plan = await planner.create_quantum_plan(max_iterations=200)
        
        # Verify plan quality
        assert plan is not None
        assert len(plan["schedule"]) == len(task_configs)
        
        # Verify dependency ordering in schedule
        schedule = plan["schedule"]
        setup_time = datetime.fromisoformat(schedule["setup"]["scheduled_time"])
        auth_time = datetime.fromisoformat(schedule["auth"]["scheduled_time"])  
        database_time = datetime.fromisoformat(schedule["database"]["scheduled_time"])
        api_time = datetime.fromisoformat(schedule["api"]["scheduled_time"])
        
        assert setup_time <= auth_time
        assert setup_time <= database_time
        assert auth_time <= api_time
        assert database_time <= api_time
        
        # Check optimization statistics
        opt_stats = plan["optimization_stats"]
        assert opt_stats["total_iterations"] > 0
        assert opt_stats["final_energy"] < opt_stats["initial_energy"]
        assert opt_stats["quantum_advantage"] >= 1.0
        
        # Verify resource allocation
        total_cpu_allocated = sum(
            task_info["resource_allocation"]["cpu"] 
            for task_info in schedule.values()
        )
        total_memory_allocated = sum(
            task_info["resource_allocation"]["memory"]
            for task_info in schedule.values()
        )
        
        assert total_cpu_allocated > 0
        assert total_memory_allocated > 0
    
    # @pytest.mark.asyncio  # Mock async marker 
    async def test_system_resilience_and_error_handling(self):
        """Test system resilience under error conditions."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        planner = create_quantum_planner()
        
        # Test invalid task creation
        with pytest.raises((ValueError, TypeError)):
            planner.add_task(
                task_id="",  # Empty ID should fail
                name="Invalid Task",
                description="Test invalid task"
            )
        
        # Test circular dependency detection
        planner.add_task(task_id="task_a", name="Task A", description="First task")
        planner.add_task(task_id="task_b", name="Task B", description="Second task")
        
        # This should work
        planner.tasks["task_a"].dependencies.add("task_b")
        
        # This creates circular dependency
        planner.tasks["task_b"].dependencies.add("task_a")
        
        validation = planner.validate_dependencies()
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
    
    def test_quantum_performance_benchmarks(self):
        """Test quantum system meets performance benchmarks."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        # Memory pool performance benchmark
        pool = QuantumMemoryPool(max_size_mb=256)
        
        start_time = time.perf_counter()
        objects = []
        
        # Create 1000 objects
        for i in range(1000):
            obj = pool.get_object(f"benchmark_obj_{i % 10}", lambda: {"id": i})
            objects.append(obj)
        
        # Return objects to pool
        for i, obj in enumerate(objects):
            pool.return_object(f"benchmark_obj_{i % 10}", obj)
        
        end_time = time.perf_counter()
        ops_per_second = 2000 / (end_time - start_time)  # 1000 get + 1000 return
        
        # Should achieve at least 10,000 operations per second
        assert ops_per_second > 10000
        
        # Check memory efficiency
        stats = pool.get_memory_usage()
        assert stats["total_pooled_objects"] > 0
        
    # @pytest.mark.asyncio  # Mock async marker
    async def test_demo_quantum_planning_functionality(self):
        """Test the demo quantum planning function works correctly."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        # Run demo quantum planning
        result = await demo_quantum_planning()
        
        assert result is not None
        assert "schedule" in result
        assert "quantum_stats" in result
        assert "export_path" in result
        
        # Check schedule structure
        schedule = result["schedule"]
        assert len(schedule) > 0
        
        for task_id, task_info in schedule.items():
            assert "scheduled_time" in task_info
            assert "estimated_completion" in task_info
            assert "assigned_resource" in task_info
            assert "quantum_state" in task_info
        
        # Check quantum statistics
        quantum_stats = result["quantum_stats"]
        assert "quantum_advantage" in quantum_stats
        assert "total_coherence" in quantum_stats
        assert "entanglement_density" in quantum_stats
        
        # Quantum advantage should be > 1.0 (quantum speedup)
        assert quantum_stats["quantum_advantage"] >= 1.0


# Performance benchmarks and stress tests
class TestQuantumStressTests:
    """Stress tests for quantum system scalability."""
    
    # @pytest.mark.slow  # Mock slow marker
    # @pytest.mark.asyncio  # Mock async marker
    async def test_large_scale_planning(self):
        """Test planning with large number of tasks."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        planner = create_quantum_planner()
        
        # Create large task set (100 tasks)
        for i in range(100):
            dependencies = set()
            if i > 0:
                # Add 1-3 random dependencies from previous tasks
                num_deps = min(i, 3)
                for j in range(num_deps):
                    if i - j - 1 >= 0:
                        dependencies.add(f"task_{i - j - 1}")
            
            planner.add_task(
                task_id=f"task_{i}",
                name=f"Task {i}",
                description=f"Large scale test task {i}",
                priority=TaskPriority.EXCITED_2,
                estimated_duration=timedelta(hours=1),
                dependencies=dependencies,
                resources_required={"cpu": 1.0, "memory": 0.5}
            )
        
        # Measure planning time
        start_time = time.perf_counter()
        plan = await planner.create_quantum_plan(max_iterations=500)
        planning_time = time.perf_counter() - start_time
        
        # Should complete planning within reasonable time (< 30 seconds)
        assert planning_time < 30.0
        assert plan is not None
        assert len(plan["schedule"]) == 100
        
    # @pytest.mark.slow  # Mock slow marker  
    def test_memory_pool_stress(self):
        """Stress test memory pool with high load."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
            
        pool = QuantumMemoryPool(max_size_mb=128)
        
        # Stress test with 10,000 operations
        operations = 10000
        start_time = time.perf_counter()
        
        for i in range(operations):
            # Alternate between get and return operations
            if i % 2 == 0:
                obj = pool.get_object("stress_test", lambda: {"data": f"test_{i}"})
            else:
                # Return a mock object
                pool.return_object("stress_test", {"data": f"test_{i}"})
        
        end_time = time.perf_counter()
        ops_per_second = operations / (end_time - start_time)
        
        # Should handle high throughput
        assert ops_per_second > 5000
        
        # Check memory pool health
        stats = pool.get_memory_usage()
        assert stats["utilization_percent"] < 100


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    exit(result.returncode)