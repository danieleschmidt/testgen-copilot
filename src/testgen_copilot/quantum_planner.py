"""Quantum-Inspired Task Planner leveraging quantum computing principles.

This module implements quantum-inspired algorithms for optimized task planning,
resource allocation, and scheduling using concepts from quantum mechanics such as
superposition, entanglement, and quantum annealing.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json
from pathlib import Path
import logging

import numpy as np


class TaskPriority(Enum):
    """Task priority levels using quantum energy states."""
    GROUND_STATE = 0      # Highest priority (lowest energy)
    EXCITED_1 = 1         # High priority  
    EXCITED_2 = 2         # Medium priority
    EXCITED_3 = 3         # Low priority
    METASTABLE = 4        # Deferred tasks


class TaskState(Enum):
    """Task states inspired by quantum superposition."""
    SUPERPOSITION = "superposition"    # Task in multiple potential states
    COLLAPSED = "collapsed"            # Task state determined
    ENTANGLED = "entangled"           # Task dependent on others
    EXECUTING = "executing"           # Task being processed
    COMPLETED = "completed"           # Task finished
    FAILED = "failed"                 # Task failed


@dataclass
class QuantumTask:
    """Task representation with quantum properties."""
    
    id: str
    name: str
    description: str
    priority: TaskPriority
    state: TaskState = TaskState.SUPERPOSITION
    
    # Quantum properties
    wave_function: List[complex] = field(default_factory=list)
    entangled_tasks: Set[str] = field(default_factory=set)
    collapse_probability: float = 0.5
    
    # Classical properties
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    dependencies: Set[str] = field(default_factory=set)
    resources_required: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    
    # Execution tracking
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    execution_attempts: int = 0
    
    def __post_init__(self):
        """Initialize quantum wave function."""
        if not self.wave_function:
            # Initialize in superposition of all possible states
            num_states = len(TaskState)
            amplitude = 1.0 / math.sqrt(num_states)
            self.wave_function = [complex(amplitude, 0) for _ in range(num_states)]
    
    def measure_state(self) -> TaskState:
        """Collapse wave function to determine current state."""
        if self.state != TaskState.SUPERPOSITION:
            return self.state
            
        # Calculate probabilities from wave function amplitudes
        probabilities = [abs(amplitude)**2 for amplitude in self.wave_function]
        
        # Quantum measurement - collapse to specific state
        states = list(TaskState)
        chosen_state = np.random.choice(states, p=probabilities)
        
        if chosen_state in [TaskState.EXECUTING, TaskState.COMPLETED, TaskState.FAILED]:
            self.state = chosen_state
            
        return chosen_state
    
    def entangle_with(self, other_task_id: str):
        """Create quantum entanglement with another task."""
        self.entangled_tasks.add(other_task_id)
        self.state = TaskState.ENTANGLED
    
    def calculate_urgency_score(self) -> float:
        """Calculate urgency using quantum energy levels."""
        base_urgency = (4 - self.priority.value) / 4.0  # Higher priority = higher urgency
        
        # Time-based quantum decay
        if self.deadline:
            time_to_deadline = (self.deadline - datetime.now(timezone.utc)).total_seconds()
            if time_to_deadline > 0:
                decay_factor = math.exp(-time_to_deadline / 86400)  # 24 hours half-life
                base_urgency *= (1 + decay_factor)
        
        # Entanglement amplification
        entanglement_factor = 1 + (len(self.entangled_tasks) * 0.1)
        
        return min(base_urgency * entanglement_factor, 1.0)


@dataclass 
class ResourceQuantum:
    """Quantum representation of computational resources."""
    
    name: str
    total_capacity: float
    available_capacity: float = 0.0
    quantum_efficiency: float = 1.0  # Quantum speedup factor
    coherence_time: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    
    def __post_init__(self):
        """Initialize available capacity."""
        if self.available_capacity == 0.0:
            self.available_capacity = self.total_capacity
    
    def apply_quantum_speedup(self, base_duration: timedelta) -> timedelta:
        """Apply quantum efficiency to reduce execution time."""
        speedup_factor = math.sqrt(self.quantum_efficiency)  # Quantum advantage
        return timedelta(seconds=base_duration.total_seconds() / speedup_factor)
    
    def reserve(self, amount: float) -> bool:
        """Reserve quantum resources with coherence check."""
        if amount <= self.available_capacity:
            self.available_capacity -= amount
            return True
        return False
    
    def release(self, amount: float):
        """Release quantum resources back to pool."""
        self.available_capacity = min(
            self.available_capacity + amount,
            self.total_capacity
        )


class QuantumAnnealer:
    """Quantum annealing optimizer for task scheduling."""
    
    def __init__(self, temperature_schedule: Optional[Callable[[int], float]] = None):
        """Initialize quantum annealer with temperature schedule."""
        self.temperature_schedule = temperature_schedule or self._default_temperature
        self.logger = logging.getLogger(__name__)
    
    def _default_temperature(self, iteration: int) -> float:
        """Default exponential cooling schedule."""
        return 1.0 * math.exp(-iteration * 0.1)
    
    def optimize_schedule(
        self, 
        tasks: List[QuantumTask], 
        resources: List[ResourceQuantum],
        max_iterations: int = 1000
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Find optimal task schedule using quantum annealing."""
        
        # Initialize random schedule
        current_schedule = self._generate_random_schedule(tasks, resources)
        current_energy = self._calculate_energy(current_schedule, tasks, resources)
        
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        for iteration in range(max_iterations):
            temperature = self.temperature_schedule(iteration)
            
            # Generate neighbor solution through quantum tunneling
            neighbor_schedule = self._quantum_tunnel(current_schedule, tasks, resources)
            neighbor_energy = self._calculate_energy(neighbor_schedule, tasks, resources)
            
            # Metropolis-Hastings acceptance with quantum probability
            delta_energy = neighbor_energy - current_energy
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_schedule = neighbor_schedule
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_schedule = current_schedule.copy()
                    best_energy = current_energy
            
            # Log progress periodically
            if iteration % 100 == 0:
                self.logger.debug(f"Annealing iteration {iteration}: energy={current_energy:.4f}, temp={temperature:.4f}")
        
        self.logger.info(f"Annealing complete: best_energy={best_energy:.4f}")
        return best_schedule
    
    def _generate_random_schedule(
        self, 
        tasks: List[QuantumTask], 
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Generate initial random schedule."""
        schedule = []
        start_time = datetime.now(timezone.utc)
        
        # Sort tasks by urgency for better initial solution
        sorted_tasks = sorted(tasks, key=lambda t: t.calculate_urgency_score(), reverse=True)
        
        for task in sorted_tasks:
            # Find available resource
            resource = min(resources, key=lambda r: r.available_capacity)
            schedule_time = start_time + timedelta(minutes=len(schedule) * 30)
            schedule.append((task, schedule_time, resource))
        
        return schedule
    
    def _quantum_tunnel(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Apply quantum tunneling to explore new schedule configurations."""
        new_schedule = schedule.copy()
        
        # Quantum tunneling: swap two random tasks
        if len(new_schedule) >= 2:
            i, j = random.sample(range(len(new_schedule)), 2)
            task_i, time_i, resource_i = new_schedule[i]
            task_j, time_j, resource_j = new_schedule[j]
            
            # Swap with quantum probability
            new_schedule[i] = (task_i, time_j, resource_j)
            new_schedule[j] = (task_j, time_i, resource_i)
        
        return new_schedule
    
    def _calculate_energy(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> float:
        """Calculate system energy (lower is better)."""
        energy = 0.0
        
        for task, scheduled_time, resource in schedule:
            # Priority penalty (higher priority tasks should be scheduled earlier)
            priority_penalty = task.priority.value * 0.1
            
            # Deadline penalty
            if task.deadline and scheduled_time > task.deadline:
                deadline_penalty = (scheduled_time - task.deadline).total_seconds() / 3600  # Hours late
                energy += deadline_penalty * 10
            
            # Resource efficiency (prefer tasks that use resources efficiently)
            cpu_requirement = task.resources_required.get('cpu', 1.0)
            if resource.available_capacity < cpu_requirement:
                energy += 5.0  # Heavy penalty for over-allocation
            
            # Dependency violations
            for dep_id in task.dependencies:
                dep_task = next((t for t, _, _ in schedule if t.id == dep_id), None)
                if dep_task:
                    dep_time = next((time for t, time, _ in schedule if t.id == dep_id), None)
                    if dep_time and dep_time > scheduled_time:
                        energy += 20.0  # Heavy penalty for dependency violations
            
            energy += priority_penalty
        
        return energy


class QuantumTaskPlanner:
    """Main quantum-inspired task planning engine."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        quantum_processors: int = 4,
        enable_entanglement: bool = True
    ):
        """Initialize quantum task planner."""
        self.max_iterations = max_iterations
        self.quantum_processors = quantum_processors
        self.enable_entanglement = enable_entanglement
        
        self.tasks: Dict[str, QuantumTask] = {}
        self.resources: List[ResourceQuantum] = []
        self.annealer = QuantumAnnealer()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default quantum resources
        self._initialize_quantum_resources()
    
    def _initialize_quantum_resources(self):
        """Initialize quantum computing resources."""
        self.resources = [
            ResourceQuantum("quantum_cpu_1", 4.0, quantum_efficiency=2.0),
            ResourceQuantum("quantum_cpu_2", 4.0, quantum_efficiency=1.8),
            ResourceQuantum("quantum_memory", 16.0, quantum_efficiency=1.5),
            ResourceQuantum("quantum_io", 8.0, quantum_efficiency=1.2)
        ]
    
    def add_task(
        self,
        task_id: str,
        name: str,
        description: str,
        priority: TaskPriority = TaskPriority.EXCITED_2,
        estimated_duration: Optional[timedelta] = None,
        dependencies: Optional[Set[str]] = None,
        resources_required: Optional[Dict[str, float]] = None,
        deadline: Optional[datetime] = None
    ) -> QuantumTask:
        """Add a new quantum task to the planner."""
        
        task = QuantumTask(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            estimated_duration=estimated_duration or timedelta(hours=1),
            dependencies=dependencies or set(),
            resources_required=resources_required or {"cpu": 1.0},
            deadline=deadline
        )
        
        self.tasks[task_id] = task
        
        # Apply quantum entanglement if enabled
        if self.enable_entanglement:
            self._create_entanglements(task)
        
        self.logger.info(f"Added quantum task: {task_id} [{task.name}]")
        return task
    
    def _create_entanglements(self, new_task: QuantumTask):
        """Create quantum entanglements between related tasks."""
        for existing_task in self.tasks.values():
            if existing_task.id == new_task.id:
                continue
                
            # Entangle tasks with shared dependencies
            if new_task.dependencies & existing_task.dependencies:
                new_task.entangle_with(existing_task.id)
                existing_task.entangle_with(new_task.id)
            
            # Entangle tasks with similar resource requirements
            resource_similarity = self._calculate_resource_similarity(
                new_task.resources_required,
                existing_task.resources_required
            )
            if resource_similarity > 0.7:
                new_task.entangle_with(existing_task.id)
                existing_task.entangle_with(new_task.id)
    
    def _calculate_resource_similarity(
        self, 
        resources_a: Dict[str, float], 
        resources_b: Dict[str, float]
    ) -> float:
        """Calculate similarity between resource requirements."""
        all_resources = set(resources_a.keys()) | set(resources_b.keys())
        
        if not all_resources:
            return 0.0
        
        similarity_sum = 0.0
        for resource in all_resources:
            val_a = resources_a.get(resource, 0.0)
            val_b = resources_b.get(resource, 0.0)
            max_val = max(val_a, val_b)
            if max_val > 0:
                similarity_sum += 1.0 - abs(val_a - val_b) / max_val
        
        return similarity_sum / len(all_resources)
    
    async def generate_optimal_plan(
        self,
        planning_horizon: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Generate optimal task execution plan using quantum annealing."""
        
        planning_horizon = planning_horizon or timedelta(days=7)
        start_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Starting quantum planning for {len(self.tasks)} tasks")
        
        # Prepare tasks for scheduling
        schedulable_tasks = [
            task for task in self.tasks.values() 
            if task.state in [TaskState.SUPERPOSITION, TaskState.COLLAPSED]
        ]
        
        if not schedulable_tasks:
            return {"schedule": [], "metrics": {}, "quantum_stats": {}}
        
        # Run quantum annealing optimization
        optimal_schedule = self.annealer.optimize_schedule(
            schedulable_tasks, 
            self.resources,
            max_iterations=self.max_iterations
        )
        
        # Generate execution plan
        execution_plan = {
            "schedule": [],
            "resource_allocation": {},
            "critical_path": [],
            "quantum_stats": {
                "total_tasks": len(schedulable_tasks),
                "entangled_pairs": self._count_entangled_pairs(),
                "superposition_count": sum(1 for t in schedulable_tasks if t.state == TaskState.SUPERPOSITION),
                "planning_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "quantum_advantage": self._calculate_quantum_advantage(optimal_schedule)
            },
            "metrics": {
                "total_estimated_duration": sum(
                    (item[0].estimated_duration.total_seconds() for item in optimal_schedule),
                    start=0.0
                ) / 3600,  # Convert to hours
                "resource_utilization": self._calculate_resource_utilization(optimal_schedule),
                "deadline_violations": self._count_deadline_violations(optimal_schedule)
            }
        }
        
        # Build schedule entries
        for task, scheduled_time, resource in optimal_schedule:
            # Apply quantum speedup
            quantum_duration = resource.apply_quantum_speedup(task.estimated_duration)
            end_time = scheduled_time + quantum_duration
            
            schedule_entry = {
                "task_id": task.id,
                "task_name": task.name,
                "priority": task.priority.name,
                "scheduled_start": scheduled_time.isoformat(),
                "scheduled_end": end_time.isoformat(),
                "estimated_duration_hours": task.estimated_duration.total_seconds() / 3600,
                "quantum_duration_hours": quantum_duration.total_seconds() / 3600,
                "assigned_resource": resource.name,
                "dependencies": list(task.dependencies),
                "entangled_tasks": list(task.entangled_tasks),
                "urgency_score": task.calculate_urgency_score()
            }
            execution_plan["schedule"].append(schedule_entry)
        
        # Calculate critical path using quantum algorithms
        execution_plan["critical_path"] = self._find_quantum_critical_path(optimal_schedule)
        
        self.logger.info("Quantum planning completed successfully")
        return execution_plan
    
    def _count_entangled_pairs(self) -> int:
        """Count unique entangled task pairs."""
        pairs = set()
        for task in self.tasks.values():
            for entangled_id in task.entangled_tasks:
                pair = tuple(sorted([task.id, entangled_id]))
                pairs.add(pair)
        return len(pairs)
    
    def _calculate_quantum_advantage(self, schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]]) -> float:
        """Calculate quantum speedup advantage over classical scheduling."""
        total_classical_time = sum(task.estimated_duration.total_seconds() for task, _, _ in schedule)
        total_quantum_time = sum(
            resource.apply_quantum_speedup(task.estimated_duration).total_seconds()
            for task, _, resource in schedule
        )
        
        if total_quantum_time > 0:
            return total_classical_time / total_quantum_time
        return 1.0
    
    def _calculate_resource_utilization(self, schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]]) -> Dict[str, float]:
        """Calculate resource utilization efficiency."""
        utilization = {}
        
        for resource in self.resources:
            total_allocated = sum(
                task.resources_required.get(resource.name.split('_')[1], 0.0)  # Extract resource type
                for task, _, assigned_resource in schedule
                if assigned_resource.name == resource.name
            )
            utilization[resource.name] = min(total_allocated / resource.total_capacity, 1.0)
        
        return utilization
    
    def _count_deadline_violations(self, schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]]) -> int:
        """Count tasks that violate their deadlines."""
        violations = 0
        
        for task, scheduled_time, resource in schedule:
            if task.deadline:
                quantum_duration = resource.apply_quantum_speedup(task.estimated_duration)
                completion_time = scheduled_time + quantum_duration
                
                if completion_time > task.deadline:
                    violations += 1
        
        return violations
    
    def _find_quantum_critical_path(
        self, 
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]]
    ) -> List[str]:
        """Find critical path using quantum path finding algorithms."""
        
        # Build dependency graph
        task_times = {task.id: (scheduled_time, resource.apply_quantum_speedup(task.estimated_duration))
                     for task, scheduled_time, resource in schedule}
        
        # Find longest path through dependency graph (critical path)
        def find_longest_path(task_id: str, visited: Set[str] = None) -> Tuple[float, List[str]]:
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return 0.0, []
            
            task = self.tasks.get(task_id)
            if not task or task_id not in task_times:
                return 0.0, []
            
            visited.add(task_id)
            start_time, duration = task_times[task_id]
            
            # Find longest path through dependencies
            max_dependency_time = 0.0
            best_path = []
            
            for dep_id in task.dependencies:
                dep_time, dep_path = find_longest_path(dep_id, visited.copy())
                if dep_time > max_dependency_time:
                    max_dependency_time = dep_time
                    best_path = dep_path
            
            total_time = max_dependency_time + duration.total_seconds()
            return total_time, best_path + [task_id]
        
        # Find the critical path
        max_time = 0.0
        critical_path = []
        
        for task in self.tasks.values():
            time, path = find_longest_path(task.id)
            if time > max_time:
                max_time = time
                critical_path = path
        
        return critical_path
    
    async def execute_plan(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the quantum-optimized plan with real-time monitoring."""
        
        execution_results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_tasks": [],
            "failed_tasks": [],
            "resource_metrics": {},
            "quantum_measurements": []
        }
        
        # Execute tasks according to quantum schedule
        schedule = execution_plan.get("schedule", [])
        
        for entry in schedule:
            task_id = entry["task_id"]
            task = self.tasks.get(task_id)
            
            if not task:
                continue
            
            try:
                # Collapse wave function to executing state
                task.state = TaskState.EXECUTING
                task.start_time = datetime.now(timezone.utc)
                task.execution_attempts += 1
                
                self.logger.info(f"Executing quantum task: {task_id}")
                
                # Simulate quantum execution with monitoring
                await self._execute_quantum_task(task, entry)
                
                # Collapse to completed state
                task.state = TaskState.COMPLETED
                task.completion_time = datetime.now(timezone.utc)
                
                execution_results["completed_tasks"].append({
                    "task_id": task_id,
                    "start_time": task.start_time.isoformat(),
                    "completion_time": task.completion_time.isoformat(),
                    "execution_time": (task.completion_time - task.start_time).total_seconds()
                })
                
                self.logger.info(f"Quantum task completed: {task_id}")
                
            except Exception as e:
                task.state = TaskState.FAILED
                execution_results["failed_tasks"].append({
                    "task_id": task_id,
                    "error": str(e),
                    "attempts": task.execution_attempts
                })
                self.logger.error(f"Quantum task failed: {task_id} - {e}")
        
        execution_results["completed_at"] = datetime.now(timezone.utc).isoformat()
        return execution_results
    
    async def _execute_quantum_task(self, task: QuantumTask, schedule_entry: Dict[str, Any]):
        """Execute individual quantum task with monitoring."""
        
        # Simulate quantum computation time
        quantum_duration = float(schedule_entry["quantum_duration_hours"]) * 3600
        
        # Apply quantum uncertainty principle
        uncertainty_factor = random.uniform(0.8, 1.2)
        actual_duration = quantum_duration * uncertainty_factor
        
        await asyncio.sleep(min(actual_duration, 2.0))  # Cap simulation time
        
        # Quantum measurement
        measurement = {
            "task_id": task.id,
            "measurement_time": datetime.now(timezone.utc).isoformat(),
            "state_before": TaskState.EXECUTING.value,
            "state_after": TaskState.COMPLETED.value,
            "quantum_efficiency": uncertainty_factor,
            "entanglement_effects": len(task.entangled_tasks)
        }
        
        return measurement
    
    def get_task_recommendations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get AI-powered task recommendations using quantum analysis."""
        
        recommendations = []
        
        if task_id:
            task = self.tasks.get(task_id)
            if task:
                recommendations.extend(self._analyze_single_task(task))
        else:
            # Analyze all tasks for system-wide recommendations
            recommendations.extend(self._analyze_system_wide())
        
        return recommendations
    
    def _analyze_single_task(self, task: QuantumTask) -> List[Dict[str, Any]]:
        """Analyze individual task and provide recommendations."""
        recommendations = []
        
        # High urgency recommendation
        urgency = task.calculate_urgency_score()
        if urgency > 0.8:
            recommendations.append({
                "type": "urgency",
                "priority": "high",
                "message": f"Task '{task.name}' has high urgency ({urgency:.2f}). Consider prioritizing.",
                "action": "prioritize"
            })
        
        # Entanglement optimization
        if len(task.entangled_tasks) > 3:
            recommendations.append({
                "type": "entanglement",
                "priority": "medium", 
                "message": f"Task has many entanglements ({len(task.entangled_tasks)}). Consider breaking down.",
                "action": "decompose"
            })
        
        # Deadline risk assessment
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now(timezone.utc)).total_seconds()
            if time_to_deadline < task.estimated_duration.total_seconds() * 1.5:
                recommendations.append({
                    "type": "deadline_risk",
                    "priority": "critical",
                    "message": f"Task may miss deadline. Consider resource reallocation.",
                    "action": "reallocate_resources"
                })
        
        return recommendations
    
    def _analyze_system_wide(self) -> List[Dict[str, Any]]:
        """Analyze entire task system for optimization opportunities."""
        recommendations = []
        
        # Resource bottleneck analysis
        total_cpu_demand = sum(
            task.resources_required.get('cpu', 0.0) for task in self.tasks.values()
        )
        total_cpu_capacity = sum(r.total_capacity for r in self.resources if 'cpu' in r.name)
        
        if total_cpu_demand > total_cpu_capacity * 0.8:
            recommendations.append({
                "type": "resource_bottleneck",
                "priority": "high",
                "message": "CPU resources approaching capacity. Consider adding quantum processors.",
                "action": "scale_resources"
            })
        
        # Task distribution analysis
        priority_distribution = {}
        for task in self.tasks.values():
            priority_distribution[task.priority.name] = priority_distribution.get(task.priority.name, 0) + 1
        
        if priority_distribution.get("GROUND_STATE", 0) > len(self.tasks) * 0.3:
            recommendations.append({
                "type": "priority_imbalance",
                "priority": "medium",
                "message": "Too many high-priority tasks. Consider re-prioritizing some tasks.",
                "action": "rebalance_priorities"
            })
        
        return recommendations
    
    def export_plan(self, output_path: str, format: str = "json") -> Path:
        """Export quantum plan to various formats."""
        output_file = Path(output_path)
        
        plan_data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "planner_version": "1.0.0",
                "quantum_enabled": True,
                "total_tasks": len(self.tasks)
            },
            "tasks": {},
            "resources": {},
            "recommendations": self.get_task_recommendations()
        }
        
        # Export tasks
        for task_id, task in self.tasks.items():
            plan_data["tasks"][task_id] = {
                "name": task.name,
                "description": task.description,
                "priority": task.priority.name,
                "state": task.state.value,
                "estimated_duration_hours": task.estimated_duration.total_seconds() / 3600,
                "dependencies": list(task.dependencies),
                "entangled_tasks": list(task.entangled_tasks),
                "resources_required": task.resources_required,
                "urgency_score": task.calculate_urgency_score(),
                "deadline": task.deadline.isoformat() if task.deadline else None
            }
        
        # Export resources  
        for resource in self.resources:
            plan_data["resources"][resource.name] = {
                "total_capacity": resource.total_capacity,
                "available_capacity": resource.available_capacity,
                "quantum_efficiency": resource.quantum_efficiency,
                "utilization_percentage": ((resource.total_capacity - resource.available_capacity) / resource.total_capacity) * 100
            }
        
        # Write to file
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(plan_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Quantum plan exported to {output_file}")
        return output_file


# Factory function for easy instantiation
def create_quantum_planner(
    max_iterations: int = 1000,
    quantum_processors: int = 4,
    enable_entanglement: bool = True
) -> QuantumTaskPlanner:
    """Create a quantum task planner with default configuration."""
    return QuantumTaskPlanner(
        max_iterations=max_iterations,
        quantum_processors=quantum_processors,
        enable_entanglement=enable_entanglement
    )


# Demo function to showcase quantum planning capabilities
async def demo_quantum_planning():
    """Demonstrate quantum task planning with sample tasks."""
    
    planner = create_quantum_planner()
    
    # Add sample tasks with various priorities and dependencies
    planner.add_task(
        "implement_auth", 
        "Implement Authentication System",
        "Build secure JWT-based authentication with OAuth2 support",
        TaskPriority.GROUND_STATE,
        timedelta(hours=8),
        resources_required={"cpu": 2.0, "memory": 4.0}
    )
    
    planner.add_task(
        "setup_database",
        "Setup Database Schema", 
        "Design and implement PostgreSQL schema with migrations",
        TaskPriority.GROUND_STATE,
        timedelta(hours=4),
        resources_required={"cpu": 1.0, "memory": 2.0, "io": 3.0}
    )
    
    planner.add_task(
        "build_api",
        "Build REST API",
        "Implement FastAPI endpoints with validation and documentation", 
        TaskPriority.EXCITED_1,
        timedelta(hours=6),
        dependencies={"implement_auth", "setup_database"},
        resources_required={"cpu": 2.5, "memory": 3.0}
    )
    
    planner.add_task(
        "write_tests",
        "Write Comprehensive Tests",
        "Create unit, integration, and e2e test suites",
        TaskPriority.EXCITED_2, 
        timedelta(hours=4),
        dependencies={"build_api"},
        resources_required={"cpu": 1.5, "memory": 2.0}
    )
    
    planner.add_task(
        "optimize_performance",
        "Performance Optimization",
        "Profile and optimize application performance bottlenecks",
        TaskPriority.EXCITED_3,
        timedelta(hours=3),
        dependencies={"write_tests"},
        deadline=datetime.now(timezone.utc) + timedelta(days=2)
    )
    
    # Generate optimal plan
    plan = await planner.generate_optimal_plan()
    
    # Export plan
    output_file = planner.export_plan("quantum_plan_demo.json")
    
    print(f"Demo quantum plan generated with {len(plan['schedule'])} tasks")
    print(f"Quantum advantage: {plan['quantum_stats']['quantum_advantage']:.2f}x speedup")
    print(f"Plan exported to: {output_file}")
    
    return plan


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_quantum_planning())