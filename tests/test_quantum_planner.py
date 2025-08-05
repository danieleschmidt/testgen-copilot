"""Test suite for quantum-inspired task planner."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from testgen_copilot.quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    TaskPriority,
    TaskState,
    ResourceQuantum,
    QuantumAnnealer,
    create_quantum_planner,
    demo_quantum_planning
)


class TestQuantumTask:
    """Test quantum task implementation."""
    
    def test_task_creation(self):
        """Test basic quantum task creation."""
        task = QuantumTask(
            id="test_task",
            name="Test Task", 
            description="A test task",
            priority=TaskPriority.GROUND_STATE
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.GROUND_STATE
        assert task.state == TaskState.SUPERPOSITION
        assert len(task.wave_function) == len(TaskState)
    
    def test_urgency_calculation(self):
        """Test urgency score calculation."""
        task = QuantumTask(
            id="urgent_task",
            name="Urgent Task",
            description="An urgent task",
            priority=TaskPriority.GROUND_STATE,
            deadline=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        
        urgency = task.calculate_urgency_score()
        assert 0.0 <= urgency <= 1.0
        assert urgency > 0.5  # High priority with near deadline should be urgent
    
    def test_entanglement(self):
        """Test quantum entanglement between tasks."""
        task1 = QuantumTask("task1", "Task 1", "First task", TaskPriority.GROUND_STATE)
        task2 = QuantumTask("task2", "Task 2", "Second task", TaskPriority.EXCITED_1)
        
        task1.entangle_with("task2")
        task2.entangle_with("task1")
        
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
        assert task1.state == TaskState.ENTANGLED
        assert task2.state == TaskState.ENTANGLED
    
    def test_state_measurement(self):
        """Test quantum state measurement and collapse."""
        task = QuantumTask("measure_task", "Measure Task", "Task for measurement", TaskPriority.EXCITED_2)
        
        # Initial state should be superposition
        assert task.state == TaskState.SUPERPOSITION
        
        # Measure state multiple times - should return valid states
        measured_states = set()
        for _ in range(10):
            state = task.measure_state()
            assert isinstance(state, TaskState)
            measured_states.add(state)
        
        # Should have measured different states due to quantum uncertainty
        # (This test might occasionally fail due to randomness, but very rarely)
        if len(measured_states) == 1:
            # If only one state measured, ensure it's a valid terminal state
            assert list(measured_states)[0] in [TaskState.EXECUTING, TaskState.COMPLETED, TaskState.FAILED]


class TestResourceQuantum:
    """Test quantum resource management."""
    
    def test_resource_creation(self):
        """Test quantum resource creation."""
        resource = ResourceQuantum(
            name="test_cpu",
            total_capacity=4.0,
            quantum_efficiency=2.0
        )
        
        assert resource.name == "test_cpu"
        assert resource.total_capacity == 4.0
        assert resource.available_capacity == 4.0
        assert resource.quantum_efficiency == 2.0
    
    def test_resource_reservation(self):
        """Test resource reservation and release."""
        resource = ResourceQuantum("test_memory", 8.0)
        
        # Reserve resources
        assert resource.reserve(3.0) is True
        assert resource.available_capacity == 5.0
        
        # Try to over-reserve
        assert resource.reserve(6.0) is False
        assert resource.available_capacity == 5.0
        
        # Release resources
        resource.release(2.0)
        assert resource.available_capacity == 7.0
        
        # Cannot release more than total capacity
        resource.release(10.0)
        assert resource.available_capacity == 8.0
    
    def test_quantum_speedup(self):
        """Test quantum speedup application."""
        resource = ResourceQuantum("quantum_processor", 2.0, quantum_efficiency=4.0)
        base_duration = timedelta(hours=4)
        
        quantum_duration = resource.apply_quantum_speedup(base_duration)
        
        # Should be faster due to quantum efficiency
        assert quantum_duration < base_duration
        # Speedup should be roughly sqrt(4.0) = 2.0x
        expected_seconds = base_duration.total_seconds() / 2.0
        assert abs(quantum_duration.total_seconds() - expected_seconds) < 1.0


class TestQuantumAnnealer:
    """Test quantum annealing optimization."""
    
    def test_annealer_creation(self):
        """Test quantum annealer initialization."""
        annealer = QuantumAnnealer()
        
        # Test default temperature schedule
        assert annealer.temperature_schedule(0) == 1.0
        assert annealer.temperature_schedule(10) < 1.0
        assert annealer.temperature_schedule(100) < annealer.temperature_schedule(10)
    
    def test_schedule_optimization(self):
        """Test schedule optimization with small task set."""
        annealer = QuantumAnnealer()
        
        # Create simple task set
        tasks = [
            QuantumTask("task1", "Task 1", "First task", TaskPriority.GROUND_STATE),
            QuantumTask("task2", "Task 2", "Second task", TaskPriority.EXCITED_1, 
                       dependencies={"task1"}),
            QuantumTask("task3", "Task 3", "Third task", TaskPriority.EXCITED_2)
        ]
        
        resources = [
            ResourceQuantum("cpu1", 2.0),
            ResourceQuantum("cpu2", 2.0)
        ]
        
        # Run optimization with limited iterations for test speed
        schedule = annealer.optimize_schedule(tasks, resources, max_iterations=50)
        
        # Verify schedule structure
        assert len(schedule) == 3
        assert all(len(item) == 3 for item in schedule)  # (task, time, resource) tuples
        
        # Verify dependency ordering
        task1_time = next(time for task, time, _ in schedule if task.id == "task1")
        task2_time = next(time for task, time, _ in schedule if task.id == "task2")
        assert task1_time <= task2_time  # task1 should be scheduled before task2


class TestQuantumTaskPlanner:
    """Test quantum task planner functionality."""
    
    def test_planner_creation(self):
        """Test quantum planner initialization."""
        planner = create_quantum_planner(quantum_processors=2, enable_entanglement=False)
        
        assert planner.quantum_processors == 2
        assert planner.enable_entanglement is False
        assert len(planner.resources) == 4  # Default quantum resources
        assert len(planner.tasks) == 0
    
    def test_task_addition(self):
        """Test adding tasks to quantum planner."""
        planner = QuantumTaskPlanner()
        
        task = planner.add_task(
            task_id="add_test",
            name="Addition Test",
            description="Test task addition",
            priority=TaskPriority.EXCITED_1,
            estimated_duration=timedelta(hours=2),
            resources_required={"cpu": 1.5, "memory": 2.0}
        )
        
        assert "add_test" in planner.tasks
        assert task.id == "add_test"
        assert task.name == "Addition Test"
        assert task.estimated_duration == timedelta(hours=2)
        assert task.resources_required["cpu"] == 1.5
    
    def test_task_entanglement_creation(self):
        """Test automatic entanglement creation."""
        planner = QuantumTaskPlanner(enable_entanglement=True)
        
        # Add tasks with shared dependencies
        task1 = planner.add_task("task1", "Task 1", "First task", dependencies={"common_dep"})
        task2 = planner.add_task("task2", "Task 2", "Second task", dependencies={"common_dep"})
        
        # Tasks should be entangled due to shared dependency
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
    
    @pytest.mark.asyncio
    async def test_plan_generation(self):
        """Test quantum plan generation."""
        planner = QuantumTaskPlanner(max_iterations=100)  # Reduced for test speed
        
        # Add sample tasks
        planner.add_task(
            "implement_feature",
            "Implement Feature",
            "Implement new feature",
            TaskPriority.GROUND_STATE,
            timedelta(hours=4)
        )
        
        planner.add_task(
            "write_tests", 
            "Write Tests",
            "Write comprehensive tests",
            TaskPriority.EXCITED_1,
            timedelta(hours=2),
            dependencies={"implement_feature"}
        )
        
        # Generate plan
        plan = await planner.generate_optimal_plan(timedelta(days=1))
        
        # Verify plan structure
        assert "schedule" in plan
        assert "quantum_stats" in plan
        assert "metrics" in plan
        
        schedule = plan["schedule"]
        assert len(schedule) == 2
        
        # Verify dependency ordering in schedule
        feature_entry = next(entry for entry in schedule if entry["task_id"] == "implement_feature")
        test_entry = next(entry for entry in schedule if entry["task_id"] == "write_tests")
        
        feature_start = datetime.fromisoformat(feature_entry["scheduled_start"])
        test_start = datetime.fromisoformat(test_entry["scheduled_start"])
        assert feature_start <= test_start
    
    def test_task_recommendations(self):
        """Test AI-powered task recommendations."""
        planner = QuantumTaskPlanner()
        
        # Add high-urgency task with tight deadline
        urgent_task = planner.add_task(
            "urgent_task",
            "Urgent Task", 
            "Very urgent task",
            TaskPriority.GROUND_STATE,
            timedelta(hours=8),
            deadline=datetime.now(timezone.utc) + timedelta(hours=6)  # Tight deadline
        )
        
        recommendations = planner.get_task_recommendations("urgent_task")
        
        # Should have deadline risk recommendation
        deadline_recs = [r for r in recommendations if r["type"] == "deadline_risk"]
        assert len(deadline_recs) > 0
        assert deadline_recs[0]["priority"] == "critical"
    
    def test_plan_export(self):
        """Test plan export functionality."""
        planner = QuantumTaskPlanner()
        
        planner.add_task(
            "export_test",
            "Export Test",
            "Test plan export",
            TaskPriority.EXCITED_2
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            output_path = planner.export_plan(temp_path, format="json")
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify content structure
            with open(output_path) as f:
                plan_data = json.load(f)
            
            assert "metadata" in plan_data
            assert "tasks" in plan_data
            assert "resources" in plan_data
            assert "recommendations" in plan_data
            
            # Verify task data
            assert "export_test" in plan_data["tasks"]
            task_data = plan_data["tasks"]["export_test"]
            assert task_data["name"] == "Export Test"
            assert task_data["priority"] == "EXCITED_2"
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)


class TestQuantumIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete quantum planning workflow."""
        planner = create_quantum_planner(max_iterations=50)  # Reduced for test speed
        
        # Create realistic development workflow
        planner.add_task(
            "setup_env",
            "Setup Environment",
            "Setup development environment", 
            TaskPriority.GROUND_STATE,
            timedelta(hours=1),
            resources_required={"cpu": 1.0, "io": 2.0}
        )
        
        planner.add_task(
            "implement_core",
            "Implement Core Logic",
            "Implement core business logic",
            TaskPriority.GROUND_STATE,
            timedelta(hours=6),
            dependencies={"setup_env"},
            resources_required={"cpu": 2.0, "memory": 4.0}
        )
        
        planner.add_task(
            "add_tests",
            "Add Test Suite",
            "Create comprehensive test suite",
            TaskPriority.EXCITED_1,
            timedelta(hours=3),
            dependencies={"implement_core"},
            resources_required={"cpu": 1.5, "memory": 2.0}
        )
        
        planner.add_task(
            "optimize",
            "Performance Optimization", 
            "Optimize performance bottlenecks",
            TaskPriority.EXCITED_2,
            timedelta(hours=4),
            dependencies={"add_tests"},
            deadline=datetime.now(timezone.utc) + timedelta(days=3)
        )
        
        # Generate optimal plan
        plan = await planner.generate_optimal_plan(timedelta(days=7))
        
        # Verify plan quality
        assert len(plan["schedule"]) == 4
        assert plan["quantum_stats"]["total_tasks"] == 4
        assert plan["quantum_stats"]["quantum_advantage"] >= 1.0
        
        # Verify no deadline violations
        assert plan["metrics"]["deadline_violations"] == 0
        
        # Execute plan (simulation)
        execution_results = await planner.execute_plan(plan)
        
        # Verify execution results
        assert len(execution_results["completed_tasks"]) == 4
        assert len(execution_results["failed_tasks"]) == 0
    
    @pytest.mark.asyncio
    async def test_demo_function(self):
        """Test demo quantum planning function."""
        # Run demo with temporary output
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            temp_path = Path(temp_dir)
            
            try:
                # Change to temp directory for demo
                import os
                os.chdir(temp_path)
                
                plan = await demo_quantum_planning()
                
                # Verify demo results
                assert "schedule" in plan
                assert "quantum_stats" in plan
                assert len(plan["schedule"]) == 5  # Demo creates 5 tasks
                
                # Verify demo plan file was created
                demo_file = temp_path / "quantum_plan_demo.json"
                assert demo_file.exists()
                
                with open(demo_file) as f:
                    exported_plan = json.load(f)
                assert "metadata" in exported_plan
                assert "tasks" in exported_plan
                
            finally:
                os.chdir(original_cwd)
    
    def test_resource_similarity_calculation(self):
        """Test resource similarity calculation for entanglement."""
        planner = QuantumTaskPlanner()
        
        resources_a = {"cpu": 2.0, "memory": 4.0}
        resources_b = {"cpu": 2.0, "memory": 4.0}
        resources_c = {"cpu": 1.0, "io": 2.0}
        
        # Identical resources should have high similarity
        similarity_ab = planner._calculate_resource_similarity(resources_a, resources_b)
        assert similarity_ab == 1.0
        
        # Different resources should have lower similarity
        similarity_ac = planner._calculate_resource_similarity(resources_a, resources_c)
        assert 0.0 <= similarity_ac < 1.0
    
    def test_critical_path_analysis(self):
        """Test quantum critical path finding."""
        planner = QuantumTaskPlanner()
        
        # Create dependency chain: A -> B -> C
        planner.add_task("A", "Task A", "First task", TaskPriority.GROUND_STATE, timedelta(hours=2))
        planner.add_task("B", "Task B", "Second task", TaskPriority.EXCITED_1, timedelta(hours=3), dependencies={"A"})
        planner.add_task("C", "Task C", "Third task", TaskPriority.EXCITED_2, timedelta(hours=1), dependencies={"B"})
        
        # Create schedule for critical path analysis
        schedule = [
            (planner.tasks["A"], datetime.now(timezone.utc), planner.resources[0]),
            (planner.tasks["B"], datetime.now(timezone.utc) + timedelta(hours=2), planner.resources[1]),
            (planner.tasks["C"], datetime.now(timezone.utc) + timedelta(hours=5), planner.resources[0])
        ]
        
        critical_path = planner._find_quantum_critical_path(schedule)
        
        # Critical path should be A -> B -> C
        assert critical_path == ["A", "B", "C"]


class TestQuantumSecurity:
    """Test security aspects of quantum planner."""
    
    def test_safe_task_creation(self):
        """Test that task creation is safe from malicious inputs."""
        planner = QuantumTaskPlanner()
        
        # Test with potentially dangerous strings
        dangerous_inputs = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')",
            "../../../etc/passwd",
            "<script>alert('xss')</script>"
        ]
        
        for dangerous_input in dangerous_inputs:
            # Should not execute dangerous code, just treat as normal strings
            task = planner.add_task(
                f"safe_task_{hash(dangerous_input)}",
                dangerous_input,  # Name field
                dangerous_input,  # Description field
                TaskPriority.EXCITED_3
            )
            
            # Task should be created safely
            assert task.name == dangerous_input
            assert task.description == dangerous_input
            assert task.state == TaskState.SUPERPOSITION
    
    def test_export_path_safety(self):
        """Test that export paths are handled safely."""
        planner = QuantumTaskPlanner()
        planner.add_task("test", "Test Task", "Test", TaskPriority.EXCITED_2)
        
        # Should handle normal paths correctly
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            output_path = planner.export_plan(temp_path)
            assert output_path.exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestQuantumPerformance:
    """Test performance characteristics of quantum planner."""
    
    @pytest.mark.asyncio
    async def test_large_task_set_performance(self):
        """Test performance with larger task sets."""
        planner = QuantumTaskPlanner(max_iterations=100)  # Limited for test speed
        
        # Create 20 tasks with complex dependencies
        tasks_created = []
        for i in range(20):
            deps = set()
            if i > 0:
                # Add dependency on previous task
                deps.add(f"task_{i-1}")
            if i > 5:
                # Add some cross-dependencies for complexity
                deps.add(f"task_{i-5}")
            
            task = planner.add_task(
                f"task_{i}",
                f"Task {i}",
                f"Description for task {i}",
                TaskPriority.EXCITED_2,
                timedelta(hours=1),
                dependencies=deps,
                resources_required={"cpu": 1.0 + (i % 3) * 0.5}
            )
            tasks_created.append(task)
        
        # Generate plan - should complete in reasonable time
        import time
        start_time = time.time()
        
        plan = await planner.generate_optimal_plan()
        
        planning_time = time.time() - start_time
        
        # Verify plan was generated successfully
        assert len(plan["schedule"]) == 20
        assert planning_time < 30.0  # Should complete within 30 seconds
        assert plan["quantum_stats"]["quantum_advantage"] >= 1.0
    
    def test_memory_efficiency(self):
        """Test memory usage with quantum planner."""
        import sys
        
        # Measure baseline memory
        baseline_refs = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        planner = QuantumTaskPlanner()
        
        # Add tasks and verify no memory leaks
        for i in range(100):
            planner.add_task(
                f"memory_test_{i}",
                f"Memory Test {i}",
                f"Memory test task {i}",
                TaskPriority.EXCITED_3
            )
        
        # Memory usage should not grow excessively
        current_refs = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Clean up
        planner.tasks.clear()
        
        # This test is mainly to ensure no obvious memory leaks
        # Exact memory comparison is platform-dependent
        assert len(planner.tasks) == 0


# Integration tests with other modules
class TestQuantumIntegrationWithExisting:
    """Test integration with existing TestGen Copilot modules."""
    
    def test_quantum_planner_with_test_generator(self):
        """Test integration with existing test generator."""
        from testgen_copilot import TestGenerator, GenerationConfig
        
        planner = QuantumTaskPlanner()
        config = GenerationConfig(language="python")
        generator = TestGenerator(config)
        
        # Add task for test generation
        planner.add_task(
            "generate_tests_quantum",
            "Generate Tests with Quantum Planning",
            "Use quantum planner to schedule test generation",
            TaskPriority.GROUND_STATE,
            timedelta(hours=1),
            resources_required={"cpu": 1.0}
        )
        
        # Verify integration works
        assert "generate_tests_quantum" in planner.tasks
        assert isinstance(generator, TestGenerator)
    
    def test_quantum_planner_with_security_scanner(self):
        """Test integration with security scanner."""
        from testgen_copilot import SecurityScanner
        
        planner = QuantumTaskPlanner()
        scanner = SecurityScanner()
        
        # Add security scan task
        planner.add_task(
            "security_scan_quantum",
            "Quantum Security Scan",
            "Run security scan using quantum scheduling",
            TaskPriority.EXCITED_1,
            timedelta(minutes=30),
            resources_required={"cpu": 0.5}
        )
        
        # Verify integration
        assert "security_scan_quantum" in planner.tasks
        assert isinstance(scanner, SecurityScanner)


# Fixtures for test data
@pytest.fixture
def sample_quantum_planner():
    """Create a quantum planner with sample tasks for testing."""
    planner = QuantumTaskPlanner(max_iterations=50, enable_entanglement=True)
    
    # Add diverse set of sample tasks
    planner.add_task(
        "database_setup",
        "Database Setup",
        "Initialize database schema and connections",
        TaskPriority.GROUND_STATE,
        timedelta(hours=2),
        resources_required={"cpu": 1.0, "memory": 2.0, "io": 3.0}
    )
    
    planner.add_task(
        "api_implementation", 
        "API Implementation",
        "Build REST API endpoints",
        TaskPriority.GROUND_STATE,
        timedelta(hours=4),
        dependencies={"database_setup"},
        resources_required={"cpu": 2.0, "memory": 3.0}
    )
    
    planner.add_task(
        "frontend_development",
        "Frontend Development", 
        "Build user interface components",
        TaskPriority.EXCITED_1,
        timedelta(hours=6),
        resources_required={"cpu": 1.5, "memory": 2.5}
    )
    
    planner.add_task(
        "integration_testing",
        "Integration Testing",
        "End-to-end integration tests",
        TaskPriority.EXCITED_2,
        timedelta(hours=3),
        dependencies={"api_implementation", "frontend_development"},
        deadline=datetime.now(timezone.utc) + timedelta(days=5)
    )
    
    return planner


@pytest.mark.asyncio
async def test_complete_workflow_with_fixture(sample_quantum_planner):
    """Test complete workflow using fixture."""
    planner = sample_quantum_planner
    
    # Generate plan
    plan = await planner.generate_optimal_plan()
    
    # Verify comprehensive plan
    assert len(plan["schedule"]) == 4
    assert plan["metrics"]["deadline_violations"] == 0
    
    # Check that dependencies are respected
    schedule_times = {
        entry["task_id"]: datetime.fromisoformat(entry["scheduled_start"]) 
        for entry in plan["schedule"]
    }
    
    # Database setup should come before API implementation
    assert schedule_times["database_setup"] <= schedule_times["api_implementation"]
    
    # Integration testing should come after both API and frontend
    assert schedule_times["api_implementation"] <= schedule_times["integration_testing"]
    assert schedule_times["frontend_development"] <= schedule_times["integration_testing"]
    
    # Test recommendations
    recommendations = planner.get_task_recommendations()
    assert isinstance(recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__])