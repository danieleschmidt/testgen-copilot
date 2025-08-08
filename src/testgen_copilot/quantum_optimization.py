"""Quantum optimization algorithms for large-scale task planning and resource allocation."""

from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .quantum_planner import QuantumTask, ResourceQuantum, TaskPriority


@dataclass
class OptimizationConfig:
    """Configuration for quantum optimization algorithms."""

    # Annealing parameters
    max_iterations: int = 10000
    initial_temperature: float = 100.0
    cooling_rate: float = 0.995
    min_temperature: float = 0.01

    # Parallel processing
    max_workers: int = field(default_factory=lambda: min(multiprocessing.cpu_count(), 8))
    use_process_pool: bool = True

    # Quantum-specific parameters
    quantum_tunneling_probability: float = 0.1
    entanglement_strength: float = 0.8
    coherence_preservation: float = 0.9

    # Optimization targets
    minimize_makespan: bool = True
    maximize_resource_utilization: bool = True
    minimize_deadline_violations: bool = True
    quantum_advantage_factor: float = 2.0


class QuantumGeneticAlgorithm:
    """Genetic algorithm with quantum-inspired operations."""

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1
    ):
        """Initialize quantum genetic algorithm."""
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.elite_count = int(population_size * elite_ratio)

        self.logger = logging.getLogger(__name__)

    def optimize_schedule(
        self,
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum],
        generations: int = 500
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Optimize task schedule using quantum genetic algorithm."""

        # Initialize population
        population = self._create_initial_population(tasks, resources)

        best_schedule = None
        best_fitness = float('inf')

        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = [self._evaluate_fitness(schedule, tasks, resources) for schedule in population]

            # Track best solution
            min_fitness_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_schedule = population[min_fitness_idx].copy()

            # Selection and reproduction
            new_population = self._reproduce_population(population, fitness_scores, tasks, resources)
            population = new_population

            # Log progress
            if generation % 50 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                self.logger.debug(f"Generation {generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")

        self.logger.info(f"Genetic optimization completed: fitness={best_fitness:.4f}")
        return best_schedule

    def _create_initial_population(
        self,
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[List[Tuple[QuantumTask, datetime, ResourceQuantum]]]:
        """Create initial population of random schedules."""

        population = []

        for _ in range(self.population_size):
            schedule = self._create_random_schedule(tasks, resources)
            population.append(schedule)

        return population

    def _create_random_schedule(
        self,
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Create a random but valid schedule."""

        # Shuffle tasks for randomness
        shuffled_tasks = tasks.copy()
        random.shuffle(shuffled_tasks)

        # Respect dependencies using topological sort
        sorted_tasks = self._topological_sort(shuffled_tasks)

        schedule = []
        start_time = datetime.now(timezone.utc)
        resource_end_times = {resource.name: start_time for resource in resources}

        for task in sorted_tasks:
            # Find best available resource
            available_resources = [r for r in resources if self._can_assign_resource(task, r)]
            if not available_resources:
                available_resources = resources  # Fallback to any resource

            # Choose resource with earliest availability
            chosen_resource = min(available_resources, key=lambda r: resource_end_times[r.name])

            # Schedule task after dependencies and resource availability
            earliest_start = max(
                resource_end_times[chosen_resource.name],
                self._calculate_dependency_completion_time(task, schedule)
            )

            # Apply quantum uncertainty
            quantum_jitter = timedelta(minutes=random.uniform(-15, 15))
            scheduled_time = earliest_start + quantum_jitter

            schedule.append((task, scheduled_time, chosen_resource))

            # Update resource end time
            task_duration = chosen_resource.apply_quantum_speedup(task.estimated_duration)
            resource_end_times[chosen_resource.name] = scheduled_time + task_duration

        return schedule

    def _topological_sort(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Sort tasks respecting dependencies."""

        # Build dependency graph
        in_degree = {task.id: 0 for task in tasks}
        adjacency = defaultdict(list)
        task_map = {task.id: task for task in tasks}

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    adjacency[dep_id].append(task.id)
                    in_degree[task.id] += 1

        # Topological sort using Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []

        while queue:
            # Quantum randomization - shuffle queue to introduce variety
            if len(queue) > 1:
                random.shuffle(queue)

            current_id = queue.pop(0)
            sorted_tasks.append(task_map[current_id])

            # Update neighbors
            for neighbor_id in adjacency[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Add any remaining tasks (circular dependencies - rare but possible)
        remaining_task_ids = set(task.id for task in tasks) - set(t.id for t in sorted_tasks)
        for task_id in remaining_task_ids:
            sorted_tasks.append(task_map[task_id])

        return sorted_tasks

    def _can_assign_resource(self, task: QuantumTask, resource: ResourceQuantum) -> bool:
        """Check if resource can handle task requirements."""

        required_capacity = sum(task.resources_required.values())
        return required_capacity <= resource.total_capacity

    def _calculate_dependency_completion_time(
        self,
        task: QuantumTask,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]]
    ) -> datetime:
        """Calculate when all dependencies will be completed."""

        if not task.dependencies:
            return datetime.now(timezone.utc)

        max_completion_time = datetime.now(timezone.utc)

        for scheduled_task, start_time, resource in schedule:
            if scheduled_task.id in task.dependencies:
                task_duration = resource.apply_quantum_speedup(scheduled_task.estimated_duration)
                completion_time = start_time + task_duration
                max_completion_time = max(max_completion_time, completion_time)

        return max_completion_time

    def _evaluate_fitness(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> float:
        """Evaluate fitness of schedule (lower is better)."""

        fitness = 0.0

        # Makespan penalty (total completion time)
        if schedule:
            makespan = max(
                start_time + resource.apply_quantum_speedup(task.estimated_duration)
                for task, start_time, resource in schedule
            )
            start_time = min(start_time for _, start_time, _ in schedule)
            total_time = (makespan - start_time).total_seconds() / 3600  # Hours
            fitness += total_time * 0.5

        # Resource utilization penalty
        resource_utilization = self._calculate_resource_utilization_fitness(schedule, resources)
        fitness += (1.0 - resource_utilization) * 10.0

        # Deadline violation penalties
        deadline_penalty = 0.0
        for task, start_time, resource in schedule:
            if task.deadline:
                completion_time = start_time + resource.apply_quantum_speedup(task.estimated_duration)
                if completion_time > task.deadline:
                    hours_late = (completion_time - task.deadline).total_seconds() / 3600
                    deadline_penalty += hours_late * 50.0  # Heavy penalty for late tasks

        fitness += deadline_penalty

        # Priority penalties (high priority tasks should be scheduled early)
        for i, (task, start_time, _) in enumerate(schedule):
            priority_penalty = task.priority.value * (i + 1) * 0.1
            fitness += priority_penalty

        # Quantum entanglement bonus (entangled tasks scheduled together get bonus)
        entanglement_bonus = self._calculate_entanglement_bonus(schedule)
        fitness -= entanglement_bonus * 2.0

        return fitness

    def _calculate_resource_utilization_fitness(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        resources: List[ResourceQuantum]
    ) -> float:
        """Calculate resource utilization efficiency."""

        if not schedule:
            return 0.0

        resource_usage = defaultdict(float)
        total_time = 0.0

        for task, start_time, resource in schedule:
            task_duration = resource.apply_quantum_speedup(task.estimated_duration).total_seconds()
            resource_usage[resource.name] += task_duration
            total_time = max(total_time, task_duration)

        if total_time == 0:
            return 0.0

        # Calculate average utilization
        total_capacity = sum(r.total_capacity for r in resources) * total_time / 3600
        total_usage = sum(resource_usage.values()) / 3600

        return min(total_usage / total_capacity, 1.0) if total_capacity > 0 else 0.0

    def _calculate_entanglement_bonus(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]]
    ) -> float:
        """Calculate bonus for scheduling entangled tasks together."""

        bonus = 0.0

        for i, (task1, time1, _) in enumerate(schedule):
            for j, (task2, time2, _) in enumerate(schedule):
                if i < j and task2.id in task1.entangled_tasks:
                    # Bonus based on temporal proximity
                    time_diff = abs((time2 - time1).total_seconds()) / 3600  # Hours
                    proximity_bonus = max(0.0, 1.0 - time_diff / 24.0)  # Bonus decreases over 24 hours
                    bonus += proximity_bonus

        return bonus

    def _reproduce_population(
        self,
        population: List[List[Tuple[QuantumTask, datetime, ResourceQuantum]]],
        fitness_scores: List[float],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[List[Tuple[QuantumTask, datetime, ResourceQuantum]]]:
        """Create next generation through selection, crossover, and mutation."""

        new_population = []

        # Elitism - keep best individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:self.elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # Quantum crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._quantum_crossover(parent1, parent2, tasks, resources)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Quantum mutation
            if random.random() < self.mutation_rate:
                child1 = self._quantum_mutation(child1, tasks, resources)
            if random.random() < self.mutation_rate:
                child2 = self._quantum_mutation(child2, tasks, resources)

            new_population.extend([child1, child2])

        # Trim to exact population size
        return new_population[:self.population_size]

    def _tournament_selection(
        self,
        population: List[List[Tuple[QuantumTask, datetime, ResourceQuantum]]],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Select individual using tournament selection."""

        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def _quantum_crossover(
        self,
        parent1: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        parent2: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> Tuple[List[Tuple[QuantumTask, datetime, ResourceQuantum]], List[Tuple[QuantumTask, datetime, ResourceQuantum]]]:
        """Perform quantum-inspired crossover operation."""

        if len(parent1) != len(parent2):
            return parent1.copy(), parent2.copy()

        # Quantum superposition crossover - blend schedules at quantum level
        child1, child2 = [], []

        for i, ((task1, time1, res1), (task2, time2, res2)) in enumerate(zip(parent1, parent2)):
            # Quantum probability determines which parent contributes to each child
            quantum_prob = random.random()

            if quantum_prob < 0.5:
                # Child 1 gets parent 1's assignment, child 2 gets parent 2's
                child1.append((task1, time1, res1))
                child2.append((task2, time2, res2))
            else:
                # Quantum entanglement - swap assignments
                child1.append((task1, time2, res2))
                child2.append((task2, time1, res1))

        # Ensure valid schedules
        child1 = self._repair_schedule(child1, tasks, resources)
        child2 = self._repair_schedule(child2, tasks, resources)

        return child1, child2

    def _quantum_mutation(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Apply quantum-inspired mutations to schedule."""

        if len(schedule) < 2:
            return schedule

        mutated_schedule = schedule.copy()

        # Quantum tunneling mutation - tasks can tunnel through time barriers
        if random.random() < 0.3:  # 30% chance
            # Time shift mutation
            idx = random.randint(0, len(mutated_schedule) - 1)
            task, time, resource = mutated_schedule[idx]

            # Apply quantum time shift
            time_shift = timedelta(hours=random.uniform(-2, 2))
            new_time = time + time_shift

            mutated_schedule[idx] = (task, new_time, resource)

        # Quantum swap mutation
        if random.random() < 0.4:  # 40% chance
            # Swap two random tasks
            i, j = random.sample(range(len(mutated_schedule)), 2)
            task_i, time_i, res_i = mutated_schedule[i]
            task_j, time_j, res_j = mutated_schedule[j]

            # Quantum swap with entanglement consideration
            if task_j.id in task_i.entangled_tasks or task_i.id in task_j.entangled_tasks:
                # Entangled tasks prefer to stay close - swap times only
                mutated_schedule[i] = (task_i, time_j, res_i)
                mutated_schedule[j] = (task_j, time_i, res_j)
            else:
                # Full swap
                mutated_schedule[i] = (task_i, time_j, res_j)
                mutated_schedule[j] = (task_j, time_i, res_i)

        # Resource reassignment mutation
        if random.random() < 0.2:  # 20% chance
            idx = random.randint(0, len(mutated_schedule) - 1)
            task, time, current_resource = mutated_schedule[idx]

            # Try different resource
            available_resources = [r for r in resources if r.name != current_resource.name]
            if available_resources:
                new_resource = random.choice(available_resources)
                mutated_schedule[idx] = (task, time, new_resource)

        # Repair schedule to ensure validity
        return self._repair_schedule(mutated_schedule, tasks, resources)

    def _repair_schedule(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Repair schedule to ensure dependency constraints are satisfied."""

        # Build task completion times map
        completion_times = {}

        for task, start_time, resource in schedule:
            task_duration = resource.apply_quantum_speedup(task.estimated_duration)
            completion_times[task.id] = start_time + task_duration

        # Fix dependency violations
        repaired_schedule = []

        for task, start_time, resource in schedule:
            # Find latest dependency completion
            latest_dep_completion = start_time

            for dep_id in task.dependencies:
                if dep_id in completion_times:
                    dep_completion = completion_times[dep_id]
                    latest_dep_completion = max(latest_dep_completion, dep_completion)

            # Adjust start time if necessary
            adjusted_start_time = max(start_time, latest_dep_completion)
            repaired_schedule.append((task, adjusted_start_time, resource))

            # Update completion time
            task_duration = resource.apply_quantum_speedup(task.estimated_duration)
            completion_times[task.id] = adjusted_start_time + task_duration

        return repaired_schedule


class QuantumParallelOptimizer:
    """Parallel optimization using quantum-inspired algorithms."""

    def __init__(self, config: OptimizationConfig):
        """Initialize quantum parallel optimizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def optimize_large_scale(
        self,
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Optimize large-scale task scheduling using parallel quantum algorithms."""

        self.logger.info(f"Starting large-scale optimization for {len(tasks)} tasks with {len(resources)} resources")

        # Divide tasks into quantum-coherent clusters
        task_clusters = self._create_quantum_clusters(tasks)

        # Optimize each cluster in parallel
        cluster_results = await self._optimize_clusters_parallel(task_clusters, resources)

        # Merge cluster results using quantum interference patterns
        merged_schedule = self._merge_cluster_schedules(cluster_results, resources)

        # Global optimization pass
        final_schedule = await self._global_optimization_pass(merged_schedule, tasks, resources)

        self.logger.info("Large-scale quantum optimization completed")
        return final_schedule

    def _create_quantum_clusters(self, tasks: List[QuantumTask]) -> List[List[QuantumTask]]:
        """Create clusters of tasks based on quantum entanglement and dependencies."""

        clusters = []
        unassigned_tasks = set(task.id for task in tasks)
        task_map = {task.id: task for task in tasks}

        while unassigned_tasks:
            # Start new cluster with random unassigned task
            seed_task_id = random.choice(list(unassigned_tasks))
            cluster_task_ids = {seed_task_id}
            unassigned_tasks.remove(seed_task_id)

            # Grow cluster using entanglement and dependencies
            growth_queue = [seed_task_id]

            while growth_queue and len(cluster_task_ids) < 20:  # Limit cluster size
                current_task_id = growth_queue.pop(0)
                current_task = task_map[current_task_id]

                # Add entangled tasks
                for entangled_id in current_task.entangled_tasks:
                    if entangled_id in unassigned_tasks:
                        cluster_task_ids.add(entangled_id)
                        unassigned_tasks.remove(entangled_id)
                        growth_queue.append(entangled_id)

                # Add dependent tasks
                for dep_id in current_task.dependencies:
                    if dep_id in unassigned_tasks:
                        cluster_task_ids.add(dep_id)
                        unassigned_tasks.remove(dep_id)
                        growth_queue.append(dep_id)

                # Add tasks that depend on current task
                for task in tasks:
                    if task.id in unassigned_tasks and current_task_id in task.dependencies:
                        cluster_task_ids.add(task.id)
                        unassigned_tasks.remove(task.id)
                        growth_queue.append(task.id)

            # Create cluster
            cluster_tasks = [task_map[task_id] for task_id in cluster_task_ids]
            clusters.append(cluster_tasks)

        self.logger.info(f"Created {len(clusters)} quantum task clusters")
        return clusters

    async def _optimize_clusters_parallel(
        self,
        task_clusters: List[List[QuantumTask]],
        resources: List[ResourceQuantum]
    ) -> List[List[Tuple[QuantumTask, datetime, ResourceQuantum]]]:
        """Optimize clusters in parallel using different algorithms."""

        # Prepare executor
        executor_class = ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor

        with executor_class(max_workers=self.config.max_workers) as executor:
            # Create optimization tasks
            optimization_tasks = []

            for i, cluster in enumerate(task_clusters):
                # Assign subset of resources to each cluster
                cluster_resources = self._distribute_resources(resources, len(task_clusters), i)

                # Choose optimization algorithm based on cluster size
                if len(cluster) <= 10:
                    # Small clusters - use genetic algorithm
                    task = executor.submit(self._optimize_small_cluster, cluster, cluster_resources)
                else:
                    # Large clusters - use quantum annealing
                    task = executor.submit(self._optimize_large_cluster, cluster, cluster_resources)

                optimization_tasks.append(task)

            # Wait for all optimizations to complete
            results = []
            for task in optimization_tasks:
                try:
                    result = task.result(timeout=300)  # 5 minute timeout per cluster
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Cluster optimization failed: {e}")
                    # Fallback to simple scheduling
                    fallback_result = self._simple_schedule_fallback(
                        task_clusters[len(results)],
                        resources
                    )
                    results.append(fallback_result)

        return results

    def _distribute_resources(
        self,
        resources: List[ResourceQuantum],
        num_clusters: int,
        cluster_index: int
    ) -> List[ResourceQuantum]:
        """Distribute resources among clusters with quantum load balancing."""

        # Simple round-robin distribution with quantum efficiency consideration
        cluster_resources = []

        for i, resource in enumerate(resources):
            if i % num_clusters == cluster_index:
                cluster_resources.append(resource)

        # Ensure each cluster gets at least one resource
        if not cluster_resources:
            cluster_resources = [resources[cluster_index % len(resources)]]

        return cluster_resources

    def _optimize_small_cluster(
        self,
        cluster: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Optimize small cluster using genetic algorithm."""

        genetic_optimizer = QuantumGeneticAlgorithm(
            population_size=50,  # Smaller population for small clusters
            mutation_rate=0.15,
            crossover_rate=0.8
        )

        return genetic_optimizer.optimize_schedule(cluster, resources, generations=200)

    def _optimize_large_cluster(
        self,
        cluster: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Optimize large cluster using quantum annealing."""

        from .quantum_planner import QuantumAnnealer

        annealer = QuantumAnnealer()
        return annealer.optimize_schedule(cluster, resources, max_iterations=1000)

    def _simple_schedule_fallback(
        self,
        cluster: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Simple fallback scheduling when optimization fails."""

        # Sort by priority and dependencies
        genetic_optimizer = QuantumGeneticAlgorithm()
        sorted_tasks = genetic_optimizer._topological_sort(cluster)

        schedule = []
        start_time = datetime.now(timezone.utc)

        for task in sorted_tasks:
            # Assign to first available resource
            resource = resources[0] if resources else ResourceQuantum("fallback", 1.0)
            schedule.append((task, start_time, resource))

            # Update start time for next task
            task_duration = resource.apply_quantum_speedup(task.estimated_duration)
            start_time += task_duration

        return schedule

    def _merge_cluster_schedules(
        self,
        cluster_results: List[List[Tuple[QuantumTask, datetime, ResourceQuantum]]],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Merge cluster schedules using quantum interference patterns."""

        merged_schedule = []

        # Simple merge - concatenate all cluster schedules
        for cluster_schedule in cluster_results:
            merged_schedule.extend(cluster_schedule)

        # Resolve resource conflicts using quantum priority
        return self._resolve_resource_conflicts(merged_schedule, resources)

    def _resolve_resource_conflicts(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Resolve resource conflicts using quantum resolution."""

        # Group by resource
        resource_schedules = defaultdict(list)
        for task, time, resource in schedule:
            resource_schedules[resource.name].append((task, time, resource))

        resolved_schedule = []

        # Resolve conflicts within each resource
        for resource_name, resource_tasks in resource_schedules.items():
            # Sort by start time
            resource_tasks.sort(key=lambda x: x[1])

            current_time = datetime.now(timezone.utc)

            for task, scheduled_time, resource in resource_tasks:
                # Ensure no overlap
                actual_start_time = max(scheduled_time, current_time)
                resolved_schedule.append((task, actual_start_time, resource))

                # Update current time
                task_duration = resource.apply_quantum_speedup(task.estimated_duration)
                current_time = actual_start_time + task_duration

        return resolved_schedule

    async def _global_optimization_pass(
        self,
        schedule: List[Tuple[QuantumTask, datetime, ResourceQuantum]],
        tasks: List[QuantumTask],
        resources: List[ResourceQuantum]
    ) -> List[Tuple[QuantumTask, datetime, ResourceQuantum]]:
        """Final global optimization pass using quantum algorithms."""

        # Apply quantum simulated annealing for final refinement
        from .quantum_planner import QuantumAnnealer

        annealer = QuantumAnnealer()

        # Quick annealing pass for final refinement
        optimized_schedule = annealer.optimize_schedule(
            tasks,
            resources,
            max_iterations=self.config.max_iterations // 10  # Quick pass
        )

        return optimized_schedule


class QuantumResourceOptimizer:
    """Optimize resource allocation using quantum algorithms."""

    def __init__(self):
        """Initialize quantum resource optimizer."""
        self.logger = logging.getLogger(__name__)

    def optimize_resource_allocation(
        self,
        tasks: List[QuantumTask],
        available_resources: Dict[str, float],
        optimization_objective: str = "minimize_cost"
    ) -> Dict[str, Any]:
        """Optimize resource allocation across tasks."""

        self.logger.info(f"Optimizing resource allocation for {len(tasks)} tasks")

        # Calculate resource demand
        total_demand = defaultdict(float)
        for task in tasks:
            for resource_type, amount in task.resources_required.items():
                total_demand[resource_type] += amount

        # Identify bottlenecks
        bottlenecks = {}
        for resource_type, demand in total_demand.items():
            available = available_resources.get(resource_type, 0)
            if demand > available:
                bottlenecks[resource_type] = {
                    "demand": demand,
                    "available": available,
                    "deficit": demand - available,
                    "utilization": demand / available if available > 0 else float('inf')
                }

        # Generate optimization recommendations
        recommendations = self._generate_resource_recommendations(
            bottlenecks, total_demand, available_resources
        )

        # Calculate optimal scaling
        scaling_suggestions = self._calculate_quantum_scaling(
            bottlenecks, tasks, optimization_objective
        )

        return {
            "resource_analysis": {
                "total_demand": dict(total_demand),
                "available_resources": available_resources,
                "bottlenecks": bottlenecks,
                "overall_utilization": sum(total_demand.values()) / sum(available_resources.values()) if available_resources else 0
            },
            "optimization_recommendations": recommendations,
            "scaling_suggestions": scaling_suggestions,
            "quantum_efficiency_score": self._calculate_quantum_efficiency_score(
                total_demand, available_resources, bottlenecks
            )
        }

    def _generate_resource_recommendations(
        self,
        bottlenecks: Dict[str, Dict[str, float]],
        total_demand: Dict[str, float],
        available_resources: Dict[str, float]
    ) -> List[str]:
        """Generate quantum-optimized resource recommendations."""

        recommendations = []

        if not bottlenecks:
            recommendations.append("✅ Resource allocation is optimal - no bottlenecks detected")
            return recommendations

        # Prioritize bottleneck resolution
        sorted_bottlenecks = sorted(
            bottlenecks.items(),
            key=lambda x: x[1]["deficit"],
            reverse=True
        )

        for resource_type, bottleneck in sorted_bottlenecks:
            deficit = bottleneck["deficit"]
            recommendations.append(
                f"⚠️ {resource_type.upper()} bottleneck: need {deficit:.1f} additional units"
            )

            # Quantum-specific recommendations
            if resource_type == "cpu":
                recommendations.append(
                    f"💡 Consider adding {math.ceil(deficit / 2)} quantum processors"
                )
            elif resource_type == "memory":
                recommendations.append(
                    f"💡 Scale memory by {math.ceil(deficit)} GB with quantum coherence support"
                )
            elif resource_type == "io":
                recommendations.append(
                    f"💡 Add {math.ceil(deficit)} high-speed quantum I/O channels"
                )

        # Load balancing recommendations
        utilization_variance = self._calculate_utilization_variance(total_demand, available_resources)
        if utilization_variance > 0.3:
            recommendations.append(
                "🔄 Consider rebalancing workload distribution across resources"
            )

        return recommendations

    def _calculate_quantum_scaling(
        self,
        bottlenecks: Dict[str, Dict[str, float]],
        tasks: List[QuantumTask],
        objective: str
    ) -> Dict[str, Any]:
        """Calculate optimal scaling using quantum algorithms."""

        scaling_suggestions = {
            "horizontal_scaling": {},
            "vertical_scaling": {},
            "quantum_enhancement": {},
            "cost_analysis": {}
        }

        for resource_type, bottleneck in bottlenecks.items():
            deficit = bottleneck["deficit"]

            # Horizontal scaling (add more resources)
            scaling_suggestions["horizontal_scaling"][resource_type] = {
                "additional_units": math.ceil(deficit),
                "scaling_factor": deficit / bottleneck["available"] if bottleneck["available"] > 0 else 1.0,
                "estimated_cost_multiplier": 1.0 + (deficit / bottleneck["available"]) if bottleneck["available"] > 0 else 2.0
            }

            # Vertical scaling (upgrade existing resources)
            scaling_suggestions["vertical_scaling"][resource_type] = {
                "upgrade_factor": 1.0 + (deficit / bottleneck["available"]) if bottleneck["available"] > 0 else 2.0,
                "quantum_efficiency_gain": self._calculate_quantum_efficiency_gain(resource_type, deficit)
            }

            # Quantum enhancement (apply quantum speedup)
            scaling_suggestions["quantum_enhancement"][resource_type] = {
                "quantum_speedup_required": math.sqrt(1.0 + deficit / bottleneck["available"]) if bottleneck["available"] > 0 else 1.5,
                "coherence_time_needed": 60.0,  # Minutes
                "entanglement_optimization": True
            }

        return scaling_suggestions

    def _calculate_utilization_variance(
        self,
        demand: Dict[str, float],
        available: Dict[str, float]
    ) -> float:
        """Calculate variance in resource utilization."""

        utilizations = []
        for resource_type in available:
            demand_val = demand.get(resource_type, 0)
            available_val = available[resource_type]
            if available_val > 0:
                utilizations.append(demand_val / available_val)

        if len(utilizations) < 2:
            return 0.0

        mean_util = sum(utilizations) / len(utilizations)
        variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)

        return math.sqrt(variance)  # Return standard deviation

    def _calculate_quantum_efficiency_gain(self, resource_type: str, deficit: float) -> float:
        """Calculate potential quantum efficiency gains."""

        # Different resource types have different quantum speedup potentials
        base_efficiency = {
            "cpu": 2.0,      # Quantum computing advantage
            "memory": 1.5,   # Quantum memory coherence
            "io": 1.2,       # Quantum I/O optimization
            "network": 1.3   # Quantum networking protocols
        }

        base = base_efficiency.get(resource_type, 1.1)

        # Efficiency improves with scale (quantum advantage scales)
        scale_factor = 1.0 + math.log10(1.0 + deficit)

        return base * scale_factor

    def _calculate_quantum_efficiency_score(
        self,
        demand: Dict[str, float],
        available: Dict[str, float],
        bottlenecks: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall quantum efficiency score."""

        if not available:
            return 0.0

        # Base efficiency from resource utilization
        total_demand = sum(demand.values())
        total_available = sum(available.values())
        base_efficiency = min(total_demand / total_available, 1.0) if total_available > 0 else 0.0

        # Penalty for bottlenecks
        bottleneck_penalty = len(bottlenecks) * 0.1

        # Quantum coherence bonus (simulated)
        coherence_bonus = 0.1 if base_efficiency > 0.8 else 0.0

        efficiency_score = max(0.0, base_efficiency - bottleneck_penalty + coherence_bonus)

        return min(efficiency_score, 1.0)


# Factory functions
def create_quantum_optimizer(config: Optional[OptimizationConfig] = None) -> QuantumParallelOptimizer:
    """Create quantum optimizer with default or custom configuration."""
    config = config or OptimizationConfig()
    return QuantumParallelOptimizer(config)


def create_resource_optimizer() -> QuantumResourceOptimizer:
    """Create quantum resource optimizer."""
    return QuantumResourceOptimizer()


# Demo function
async def demo_quantum_optimization():
    """Demonstrate quantum optimization capabilities."""

    from .quantum_planner import create_quantum_planner

    # Create planner with many tasks
    planner = create_quantum_planner()

    # Add complex task set
    task_dependencies = {
        "foundation": [],
        "auth": ["foundation"],
        "database": ["foundation"],
        "api": ["auth", "database"],
        "frontend": ["api"],
        "testing": ["api", "frontend"],
        "deployment": ["testing"],
        "monitoring": ["deployment"]
    }

    for task_id, deps in task_dependencies.items():
        planner.add_task(
            task_id=task_id,
            name=f"{task_id.title()} Task",
            description=f"Implement {task_id} functionality",
            priority=TaskPriority.GROUND_STATE if not deps else TaskPriority.EXCITED_1,
            estimated_duration=timedelta(hours=random.uniform(2, 8)),
            dependencies=set(deps),
            resources_required={
                "cpu": random.uniform(1.0, 3.0),
                "memory": random.uniform(1.0, 4.0),
                "io": random.uniform(0.5, 2.0)
            }
        )

    # Create optimizer
    config = OptimizationConfig(max_iterations=1000, max_workers=4)
    optimizer = create_quantum_optimizer(config)

    # Run optimization
    print("🚀 Running quantum optimization demo...")

    start_time = time.time()
    optimized_schedule = await optimizer.optimize_large_scale(
        list(planner.tasks.values()),
        planner.resources
    )
    optimization_time = time.time() - start_time

    # Analyze results
    makespan = max(
        start_time + resource.apply_quantum_speedup(task.estimated_duration)
        for task, start_time, resource in optimized_schedule
    ) - min(start_time for _, start_time, _ in optimized_schedule)

    print(f"✅ Optimization completed in {optimization_time:.2f}s")
    print(f"📊 Optimized schedule for {len(optimized_schedule)} tasks")
    print(f"⏱️ Total makespan: {makespan.total_seconds() / 3600:.1f} hours")

    # Resource optimization demo
    resource_optimizer = create_resource_optimizer()

    available_resources = {
        "cpu": 8.0,
        "memory": 16.0,
        "io": 4.0
    }

    resource_analysis = resource_optimizer.optimize_resource_allocation(
        list(planner.tasks.values()),
        available_resources
    )

    print(f"🔧 Resource efficiency score: {resource_analysis['quantum_efficiency_score']:.2f}")
    print(f"📈 Optimization recommendations: {len(resource_analysis['optimization_recommendations'])}")

    return {
        "optimization_time": optimization_time,
        "schedule_length": len(optimized_schedule),
        "makespan_hours": makespan.total_seconds() / 3600,
        "resource_efficiency": resource_analysis["quantum_efficiency_score"]
    }


if __name__ == "__main__":
    import time
    asyncio.run(demo_quantum_optimization())
