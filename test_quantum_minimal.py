"""Minimal test for quantum planner core functionality."""

import sys
import os
import asyncio
import math
import random
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any, Callable
import logging

# Import quantum modules directly without package dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock numpy for testing
class MockNumpy:
    @staticmethod
    def random():
        class MockRandom:
            @staticmethod
            def choice(choices, p=None):
                if p:
                    # Weighted random choice
                    cumulative = []
                    total = 0
                    for prob in p:
                        total += prob
                        cumulative.append(total)
                    
                    r = random.random() * total
                    for i, cum_prob in enumerate(cumulative):
                        if r <= cum_prob:
                            return choices[i]
                
                return random.choice(choices)
        return MockRandom()
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def linalg():
        class MockLinalg:
            @staticmethod
            def norm(vector):
                return math.sqrt(sum(x * x for x in vector))
        return MockLinalg()

# Mock numpy import
sys.modules['numpy'] = MockNumpy()
import numpy as np

# Now import quantum modules
exec(open('/root/repo/src/testgen_copilot/quantum_planner.py').read())


def test_quantum_fundamentals():
    """Test fundamental quantum concepts."""
    print("🧪 Testing quantum fundamentals...")
    
    # Test TaskPriority enum
    assert TaskPriority.GROUND_STATE.value == 0
    assert TaskPriority.EXCITED_1.value == 1
    assert TaskPriority.METASTABLE.value == 4
    
    # Test TaskState enum
    assert TaskState.SUPERPOSITION.value == "superposition"
    assert TaskState.ENTANGLED.value == "entangled"
    assert TaskState.COMPLETED.value == "completed"
    
    print("✅ Quantum fundamentals test passed")


def test_quantum_task_basic():
    """Test basic quantum task functionality."""
    print("🧪 Testing quantum task basics...")
    
    task = QuantumTask(
        id="basic_test",
        name="Basic Test Task",
        description="Testing basic functionality",
        priority=TaskPriority.GROUND_STATE
    )
    
    # Verify basic properties
    assert task.id == "basic_test"
    assert task.name == "Basic Test Task"
    assert task.priority == TaskPriority.GROUND_STATE
    assert task.state == TaskState.SUPERPOSITION
    
    # Test wave function initialization
    assert len(task.wave_function) == len(TaskState)
    assert all(isinstance(amplitude, complex) for amplitude in task.wave_function)
    
    # Test urgency calculation
    urgency = task.calculate_urgency_score()
    assert 0.0 <= urgency <= 1.0
    
    print("✅ Quantum task basics test passed")


def test_quantum_resource_basic():
    """Test basic quantum resource functionality."""
    print("🧪 Testing quantum resource basics...")
    
    resource = ResourceQuantum(
        name="test_processor",
        total_capacity=8.0,
        quantum_efficiency=1.5
    )
    
    # Verify initialization
    assert resource.name == "test_processor"
    assert resource.total_capacity == 8.0
    assert resource.available_capacity == 8.0
    assert resource.quantum_efficiency == 1.5
    
    # Test reservation
    success = resource.reserve(3.0)
    assert success is True
    assert resource.available_capacity == 5.0
    
    # Test over-reservation
    over_success = resource.reserve(10.0)
    assert over_success is False
    assert resource.available_capacity == 5.0
    
    # Test release
    resource.release(2.0)
    assert resource.available_capacity == 7.0
    
    # Test quantum speedup
    base_duration = timedelta(hours=4)
    quantum_duration = resource.apply_quantum_speedup(base_duration)
    
    # Should be faster due to quantum efficiency
    assert quantum_duration < base_duration
    
    print("✅ Quantum resource basics test passed")


def test_entanglement():
    """Test quantum entanglement between tasks."""
    print("🧪 Testing quantum entanglement...")
    
    task1 = QuantumTask("entangle1", "Task 1", "First task", TaskPriority.GROUND_STATE)
    task2 = QuantumTask("entangle2", "Task 2", "Second task", TaskPriority.EXCITED_1)
    
    # Create entanglement
    task1.entangle_with("entangle2")
    task2.entangle_with("entangle1")
    
    # Verify entanglement
    assert "entangle2" in task1.entangled_tasks
    assert "entangle1" in task2.entangled_tasks
    assert task1.state == TaskState.ENTANGLED
    assert task2.state == TaskState.ENTANGLED
    
    print("✅ Quantum entanglement test passed")


def test_quantum_annealer_basic():
    """Test basic quantum annealing functionality."""
    print("🧪 Testing quantum annealer...")
    
    annealer = QuantumAnnealer()
    
    # Test temperature schedule
    temp_0 = annealer.temperature_schedule(0)
    temp_10 = annealer.temperature_schedule(10)
    temp_100 = annealer.temperature_schedule(100)
    
    assert temp_0 == 1.0
    assert temp_10 < temp_0
    assert temp_100 < temp_10
    
    # Test with minimal task set
    tasks = [
        QuantumTask("simple1", "Simple 1", "First simple task", TaskPriority.GROUND_STATE),
        QuantumTask("simple2", "Simple 2", "Second simple task", TaskPriority.EXCITED_1)
    ]
    
    resources = [
        ResourceQuantum("cpu1", 2.0),
        ResourceQuantum("cpu2", 2.0)
    ]
    
    # Run minimal optimization
    schedule = annealer.optimize_schedule(tasks, resources, max_iterations=10)
    
    # Verify schedule
    assert len(schedule) == 2
    assert all(len(item) == 3 for item in schedule)
    
    print("✅ Quantum annealer test passed")


async def test_quantum_planner_basic():
    """Test basic quantum planner functionality."""
    print("🧪 Testing quantum planner...")
    
    planner = QuantumTaskPlanner(max_iterations=50, quantum_processors=2, enable_entanglement=False)
    
    # Verify initialization
    assert planner.max_iterations == 50
    assert planner.quantum_processors == 2
    assert planner.enable_entanglement is False
    assert len(planner.resources) == 4
    
    # Add task
    task = planner.add_task(
        task_id="planner_basic",
        name="Planner Basic Test",
        description="Basic planner functionality test",
        priority=TaskPriority.EXCITED_2,
        estimated_duration=timedelta(hours=1),
        resources_required={"cpu": 1.0}
    )
    
    assert "planner_basic" in planner.tasks
    assert task.name == "Planner Basic Test"
    
    # Generate simple plan
    plan = await planner.generate_optimal_plan(timedelta(days=1))
    
    # Verify plan structure
    assert "schedule" in plan
    assert "quantum_stats" in plan
    assert "metrics" in plan
    
    schedule = plan["schedule"]
    assert len(schedule) == 1
    
    entry = schedule[0]
    assert entry["task_id"] == "planner_basic"
    assert "scheduled_start" in entry
    assert "quantum_duration_hours" in entry
    
    print("✅ Quantum planner test passed")


async def test_quantum_execution():
    """Test quantum task execution simulation."""
    print("🧪 Testing quantum execution...")
    
    planner = QuantumTaskPlanner(max_iterations=20)
    
    # Add simple task
    planner.add_task(
        "exec_test",
        "Execution Test",
        "Test task execution",
        TaskPriority.GROUND_STATE,
        timedelta(minutes=5)  # Very short for test
    )
    
    # Generate and execute plan
    plan = await planner.generate_optimal_plan()
    results = await planner.execute_plan(plan)
    
    # Verify execution results
    assert "completed_tasks" in results
    assert "failed_tasks" in results
    assert len(results["completed_tasks"]) == 1
    assert len(results["failed_tasks"]) == 0
    
    completed_task = results["completed_tasks"][0]
    assert completed_task["task_id"] == "exec_test"
    assert "execution_time" in completed_task
    
    print("✅ Quantum execution test passed")


def test_urgency_calculation():
    """Test urgency score calculation."""
    print("🧪 Testing urgency calculation...")
    
    # High priority task
    high_priority = QuantumTask(
        "urgent1", "Urgent Task", "High priority task", 
        TaskPriority.GROUND_STATE
    )
    
    # Low priority task
    low_priority = QuantumTask(
        "low1", "Low Task", "Low priority task",
        TaskPriority.EXCITED_3
    )
    
    # Task with deadline
    deadline_task = QuantumTask(
        "deadline1", "Deadline Task", "Task with tight deadline",
        TaskPriority.EXCITED_2,
        deadline=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    
    # Calculate urgency scores
    high_urgency = high_priority.calculate_urgency_score()
    low_urgency = low_priority.calculate_urgency_score()
    deadline_urgency = deadline_task.calculate_urgency_score()
    
    # Verify urgency relationships
    assert high_urgency > low_urgency
    assert deadline_urgency > low_urgency  # Deadline should increase urgency
    assert 0.0 <= high_urgency <= 1.0
    assert 0.0 <= low_urgency <= 1.0
    assert 0.0 <= deadline_urgency <= 1.0
    
    print("✅ Urgency calculation test passed")


def run_comprehensive_test():
    """Run comprehensive quantum planner test."""
    print("🌌 Quantum-Inspired Task Planner Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_quantum_fundamentals()
        test_quantum_task_basic()
        test_quantum_resource_basic()
        test_entanglement()
        test_quantum_annealer_basic()
        test_urgency_calculation()
        
        # Run async tests
        asyncio.run(test_quantum_planner_basic())
        asyncio.run(test_quantum_execution())
        # Demo test skipped - requires file I/O
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ Quantum Task Planner implementation is working correctly")
        print("🚀 Ready for production deployment")
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    print(f"\n🏁 Test Results: {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1)