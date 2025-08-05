"""CLI interface for quantum-inspired task planning."""

import asyncio
import click
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .quantum_planner import (
    QuantumTaskPlanner, 
    TaskPriority, 
    TaskState,
    create_quantum_planner,
    demo_quantum_planning
)
from .logging_config import get_core_logger


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def quantum(ctx, verbose):
    """Quantum-inspired task planning commands."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)


@quantum.command()
@click.option('--max-iterations', '-i', default=1000, help='Maximum annealing iterations')
@click.option('--processors', '-p', default=4, help='Number of quantum processors')
@click.option('--no-entanglement', is_flag=True, help='Disable quantum entanglement')
@click.pass_context
def create(ctx, max_iterations, processors, no_entanglement):
    """Create a new quantum task planner instance."""
    
    planner = create_quantum_planner(
        max_iterations=max_iterations,
        quantum_processors=processors,
        enable_entanglement=not no_entanglement
    )
    
    # Save planner state
    config = {
        "max_iterations": max_iterations,
        "quantum_processors": processors,
        "enable_entanglement": not no_entanglement,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(".quantum_planner.json", "w") as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"🚀 Quantum planner created with {processors} processors")
    if not no_entanglement:
        click.echo("🔗 Quantum entanglement enabled")


@quantum.command()
@click.argument('task_id')
@click.argument('name')
@click.argument('description')
@click.option('--priority', '-p', 
              type=click.Choice(['ground', 'high', 'medium', 'low', 'deferred']),
              default='medium', help='Task priority level')
@click.option('--duration', '-d', default=1.0, help='Estimated duration in hours')
@click.option('--dependencies', help='Comma-separated list of dependency task IDs')
@click.option('--deadline', help='Task deadline (ISO format: YYYY-MM-DDTHH:MM:SS)')
@click.option('--cpu', default=1.0, help='CPU resource requirement')
@click.option('--memory', default=1.0, help='Memory resource requirement')
@click.option('--io', default=0.0, help='I/O resource requirement')
def add_task(task_id, name, description, priority, duration, dependencies, deadline, cpu, memory, io):
    """Add a new quantum task to the planner."""
    
    # Load existing planner or create new one
    planner = _load_or_create_planner()
    
    # Parse priority
    priority_map = {
        'ground': TaskPriority.GROUND_STATE,
        'high': TaskPriority.EXCITED_1, 
        'medium': TaskPriority.EXCITED_2,
        'low': TaskPriority.EXCITED_3,
        'deferred': TaskPriority.METASTABLE
    }
    task_priority = priority_map[priority]
    
    # Parse dependencies
    deps = set()
    if dependencies:
        deps = set(dep.strip() for dep in dependencies.split(','))
    
    # Parse deadline
    task_deadline = None
    if deadline:
        try:
            task_deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        except ValueError:
            click.echo(f"❌ Invalid deadline format: {deadline}")
            return
    
    # Create resources dict
    resources = {}
    if cpu > 0:
        resources['cpu'] = cpu
    if memory > 0:
        resources['memory'] = memory  
    if io > 0:
        resources['io'] = io
    
    # Add task to planner
    task = planner.add_task(
        task_id=task_id,
        name=name,
        description=description,
        priority=task_priority,
        estimated_duration=timedelta(hours=duration),
        dependencies=deps,
        resources_required=resources,
        deadline=task_deadline
    )
    
    # Save updated planner state
    _save_planner_state(planner)
    
    click.echo(f"✅ Added quantum task: {task_id}")
    click.echo(f"   Priority: {priority} ({task.priority.name})")
    click.echo(f"   Duration: {duration}h")
    if deps:
        click.echo(f"   Dependencies: {', '.join(deps)}")
    if task_deadline:
        click.echo(f"   Deadline: {task_deadline.isoformat()}")


@quantum.command()
@click.option('--horizon', '-h', default=7, help='Planning horizon in days')
@click.option('--output', '-o', help='Output file path for the plan')
@click.option('--format', '-f', type=click.Choice(['json']), default='json', help='Output format')
def plan(horizon, output, format):
    """Generate optimal quantum task execution plan."""
    
    planner = _load_or_create_planner()
    
    if not planner.tasks:
        click.echo("❌ No tasks found. Add tasks first using 'quantum add-task'")
        return
    
    click.echo("🧮 Generating optimal quantum plan...")
    
    # Run quantum planning
    planning_horizon = timedelta(days=horizon)
    
    async def run_planning():
        return await planner.generate_optimal_plan(planning_horizon)
    
    plan_result = asyncio.run(run_planning())
    
    # Display summary
    schedule = plan_result.get("schedule", [])
    quantum_stats = plan_result.get("quantum_stats", {})
    
    click.echo(f"✅ Quantum plan generated:")
    click.echo(f"   📋 Tasks scheduled: {len(schedule)}")
    click.echo(f"   🔗 Entangled pairs: {quantum_stats.get('entangled_pairs', 0)}")
    click.echo(f"   ⚡ Quantum advantage: {quantum_stats.get('quantum_advantage', 1.0):.2f}x")
    click.echo(f"   ⏱️  Planning time: {quantum_stats.get('planning_time', 0):.2f}s")
    
    # Save or display plan
    if output:
        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump(plan_result, f, indent=2, default=str)
        click.echo(f"📁 Plan saved to: {output_path}")
    else:
        # Display schedule summary
        click.echo("\n📅 Schedule Summary:")
        for entry in schedule[:5]:  # Show first 5 tasks
            start_time = datetime.fromisoformat(entry["scheduled_start"])
            click.echo(f"   • {entry['task_name']} - {start_time.strftime('%Y-%m-%d %H:%M')}")
        
        if len(schedule) > 5:
            click.echo(f"   ... and {len(schedule) - 5} more tasks")


@quantum.command()
@click.option('--plan-file', '-p', help='Path to quantum plan JSON file')
def execute(plan_file):
    """Execute quantum task plan with real-time monitoring."""
    
    if not plan_file:
        click.echo("❌ Plan file required. Generate plan first using 'quantum plan -o plan.json'")
        return
    
    plan_path = Path(plan_file)
    if not plan_path.exists():
        click.echo(f"❌ Plan file not found: {plan_path}")
        return
    
    # Load plan
    with open(plan_path) as f:
        execution_plan = json.load(f)
    
    planner = _load_or_create_planner()
    
    click.echo("🚀 Executing quantum plan...")
    
    async def run_execution():
        return await planner.execute_plan(execution_plan)
    
    results = asyncio.run(run_execution())
    
    # Display execution results
    completed = len(results.get("completed_tasks", []))
    failed = len(results.get("failed_tasks", []))
    
    click.echo(f"✅ Execution completed:")
    click.echo(f"   ✅ Tasks completed: {completed}")
    click.echo(f"   ❌ Tasks failed: {failed}")
    
    if failed > 0:
        click.echo("\n❌ Failed tasks:")
        for failure in results.get("failed_tasks", []):
            click.echo(f"   • {failure['task_id']}: {failure['error']}")


@quantum.command()
@click.option('--task-id', help='Get recommendations for specific task')
def recommend(task_id):
    """Get AI-powered task recommendations using quantum analysis."""
    
    planner = _load_or_create_planner()
    recommendations = planner.get_task_recommendations(task_id)
    
    if not recommendations:
        click.echo("✅ No recommendations at this time")
        return
    
    click.echo("🧠 Quantum Task Recommendations:")
    
    for rec in recommendations:
        priority_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '💡',
            'low': 'ℹ️'
        }.get(rec['priority'], '📝')
        
        click.echo(f"\n{priority_emoji} {rec['type'].upper()}")
        click.echo(f"   {rec['message']}")
        click.echo(f"   🎯 Suggested action: {rec['action']}")


@quantum.command()
def status():
    """Show quantum planner status and task overview."""
    
    planner = _load_or_create_planner()
    
    if not planner.tasks:
        click.echo("📋 No tasks in quantum planner")
        return
    
    # Count tasks by state
    state_counts = {}
    for task in planner.tasks.values():
        state_counts[task.state.value] = state_counts.get(task.state.value, 0) + 1
    
    # Count tasks by priority
    priority_counts = {}
    for task in planner.tasks.values():
        priority_counts[task.priority.name] = priority_counts.get(task.priority.name, 0) + 1
    
    click.echo("🌌 Quantum Planner Status:")
    click.echo(f"   📋 Total tasks: {len(planner.tasks)}")
    click.echo(f"   ⚡ Quantum processors: {planner.quantum_processors}")
    click.echo(f"   🔗 Entanglement: {'enabled' if planner.enable_entanglement else 'disabled'}")
    
    click.echo("\n📊 Task States:")
    for state, count in state_counts.items():
        click.echo(f"   • {state}: {count}")
    
    click.echo("\n🎯 Priority Distribution:")
    for priority, count in priority_counts.items():
        click.echo(f"   • {priority}: {count}")
    
    # Show resource utilization
    click.echo("\n🔧 Resource Status:")
    for resource in planner.resources:
        utilization = ((resource.total_capacity - resource.available_capacity) / resource.total_capacity) * 100
        click.echo(f"   • {resource.name}: {utilization:.1f}% utilized")


@quantum.command()
def demo():
    """Run quantum planning demonstration with sample tasks."""
    
    click.echo("🌌 Running quantum planning demo...")
    
    async def run_demo():
        return await demo_quantum_planning()
    
    plan = asyncio.run(run_demo())
    
    click.echo("✅ Demo completed successfully!")
    click.echo("📁 Check 'quantum_plan_demo.json' for detailed results")


@quantum.command()
@click.argument('task_id')
def remove_task(task_id):
    """Remove a task from the quantum planner."""
    
    planner = _load_or_create_planner()
    
    if task_id not in planner.tasks:
        click.echo(f"❌ Task not found: {task_id}")
        return
    
    # Remove entanglements first
    task = planner.tasks[task_id]
    for entangled_id in task.entangled_tasks:
        if entangled_id in planner.tasks:
            planner.tasks[entangled_id].entangled_tasks.discard(task_id)
    
    # Remove task
    del planner.tasks[task_id]
    
    # Save updated state
    _save_planner_state(planner)
    
    click.echo(f"✅ Removed quantum task: {task_id}")


@quantum.command()
@click.option('--format', '-f', type=click.Choice(['json', 'table']), default='table')
def list_tasks(format):
    """List all quantum tasks with their current states."""
    
    planner = _load_or_create_planner()
    
    if not planner.tasks:
        click.echo("📋 No tasks found")
        return
    
    if format == 'json':
        tasks_data = {}
        for task_id, task in planner.tasks.items():
            tasks_data[task_id] = {
                "name": task.name,
                "description": task.description,
                "priority": task.priority.name,
                "state": task.state.value,
                "urgency_score": task.calculate_urgency_score(),
                "entangled_tasks": list(task.entangled_tasks),
                "dependencies": list(task.dependencies)
            }
        
        click.echo(json.dumps(tasks_data, indent=2))
    else:
        # Table format
        click.echo("📋 Quantum Tasks:")
        click.echo("-" * 80)
        click.echo(f"{'ID':<15} {'Name':<25} {'Priority':<12} {'State':<15} {'Urgency':<8}")
        click.echo("-" * 80)
        
        for task_id, task in planner.tasks.items():
            urgency = task.calculate_urgency_score()
            urgency_str = f"{urgency:.2f}"
            
            click.echo(f"{task_id:<15} {task.name[:24]:<25} {task.priority.name:<12} {task.state.value:<15} {urgency_str:<8}")


def _load_or_create_planner() -> QuantumTaskPlanner:
    """Load existing planner state or create new one."""
    config_path = Path(".quantum_planner.json")
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        planner = create_quantum_planner(
            max_iterations=config.get("max_iterations", 1000),
            quantum_processors=config.get("quantum_processors", 4),
            enable_entanglement=config.get("enable_entanglement", True)
        )
        
        # Load tasks if they exist
        tasks_path = Path(".quantum_tasks.json")
        if tasks_path.exists():
            _load_tasks(planner, tasks_path)
    else:
        planner = create_quantum_planner()
    
    return planner


def _save_planner_state(planner: QuantumTaskPlanner):
    """Save planner state to disk."""
    
    # Save tasks
    tasks_data = {}
    for task_id, task in planner.tasks.items():
        tasks_data[task_id] = {
            "name": task.name,
            "description": task.description, 
            "priority": task.priority.value,
            "state": task.state.value,
            "estimated_duration_seconds": task.estimated_duration.total_seconds(),
            "dependencies": list(task.dependencies),
            "resources_required": task.resources_required,
            "entangled_tasks": list(task.entangled_tasks),
            "created_at": task.created_at.isoformat(),
            "deadline": task.deadline.isoformat() if task.deadline else None,
            "execution_attempts": task.execution_attempts
        }
    
    with open(".quantum_tasks.json", "w") as f:
        json.dump(tasks_data, f, indent=2)


def _load_tasks(planner: QuantumTaskPlanner, tasks_path: Path):
    """Load tasks from disk into planner."""
    
    with open(tasks_path) as f:
        tasks_data = json.load(f)
    
    for task_id, data in tasks_data.items():
        # Reconstruct task
        priority = TaskPriority(data["priority"])
        state = TaskState(data["state"])
        
        duration = timedelta(seconds=data["estimated_duration_seconds"])
        dependencies = set(data.get("dependencies", []))
        resources = data.get("resources_required", {})
        entangled = set(data.get("entangled_tasks", []))
        
        created_at = datetime.fromisoformat(data["created_at"])
        deadline = None
        if data.get("deadline"):
            deadline = datetime.fromisoformat(data["deadline"])
        
        task = planner.add_task(
            task_id=task_id,
            name=data["name"],
            description=data["description"],
            priority=priority,
            estimated_duration=duration,
            dependencies=dependencies,
            resources_required=resources,
            deadline=deadline
        )
        
        # Restore quantum state
        task.state = state
        task.entangled_tasks = entangled
        task.created_at = created_at
        task.execution_attempts = data.get("execution_attempts", 0)


if __name__ == "__main__":
    quantum()