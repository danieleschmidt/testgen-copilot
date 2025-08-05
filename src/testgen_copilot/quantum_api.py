"""FastAPI endpoints for quantum task planning with real-time WebSocket support."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .quantum_planner import (
    QuantumTaskPlanner, 
    QuantumTask, 
    TaskPriority, 
    TaskState,
    create_quantum_planner
)
from .quantum_monitoring import (
    QuantumMonitoringDashboard,
    AlertSeverity,
    create_quantum_monitoring_dashboard
)


# Pydantic models for API
class TaskCreateRequest(BaseModel):
    """Request model for creating quantum tasks."""
    
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    priority: str = Field(default="medium", regex="^(ground|high|medium|low|deferred)$")
    estimated_duration_hours: float = Field(default=1.0, gt=0, le=100)
    dependencies: List[str] = Field(default_factory=list)
    resources_cpu: float = Field(default=1.0, ge=0, le=10)
    resources_memory: float = Field(default=1.0, ge=0, le=32)
    resources_io: float = Field(default=0.0, ge=0, le=10)
    deadline: Optional[str] = Field(None, description="ISO format deadline")
    
    @validator('deadline')
    def validate_deadline(cls, v):
        """Validate deadline format."""
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Deadline must be in ISO format")
        return v


class TaskResponse(BaseModel):
    """Response model for quantum tasks."""
    
    id: str
    name: str
    description: str
    priority: str
    state: str
    estimated_duration_hours: float
    dependencies: List[str]
    entangled_tasks: List[str]
    resources_required: Dict[str, float]
    urgency_score: float
    created_at: str
    deadline: Optional[str]


class PlanRequest(BaseModel):
    """Request model for plan generation."""
    
    horizon_days: int = Field(default=7, gt=0, le=365)
    max_iterations: int = Field(default=1000, gt=0, le=10000)


class PlanResponse(BaseModel):
    """Response model for generated plans."""
    
    schedule: List[Dict[str, Any]]
    quantum_stats: Dict[str, Any]
    metrics: Dict[str, Any]
    recommendations: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str
    timestamp: str
    quantum_coherence: float
    system_metrics: Dict[str, float]
    alerts_count: int


# Global planner instance
_planner: Optional[QuantumTaskPlanner] = None
_monitoring_dashboard: Optional[QuantumMonitoringDashboard] = None


def get_planner() -> QuantumTaskPlanner:
    """Get or create quantum planner instance."""
    global _planner
    if _planner is None:
        _planner = create_quantum_planner()
    return _planner


def get_monitoring_dashboard() -> QuantumMonitoringDashboard:
    """Get or create monitoring dashboard instance."""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = create_quantum_monitoring_dashboard()
    return _monitoring_dashboard


# FastAPI app configuration
app = FastAPI(
    title="Quantum Task Planner API",
    description="REST API for quantum-inspired task planning and execution",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logger = logging.getLogger(__name__)


# WebSocket connection manager
class QuantumWebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                self.logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(connection)
        
        # Remove failed connections
        for connection in disconnected:
            self.disconnect(connection)


# Global WebSocket manager
websocket_manager = QuantumWebSocketManager()


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint with quantum information."""
    return {
        "message": "Quantum Task Planner API",
        "version": "1.0.0",
        "quantum_enabled": True,
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Quantum-aware health check endpoint."""
    
    dashboard = get_monitoring_dashboard()
    health_data = dashboard.health_checker.check_system_health()
    
    return HealthResponse(
        status=health_data["overall_status"],
        timestamp=health_data["timestamp"],
        quantum_coherence=health_data["quantum_coherence"],
        system_metrics=health_data["system_metrics"],
        alerts_count=len(health_data["alerts"])
    )


@app.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskCreateRequest, background_tasks: BackgroundTasks):
    """Create a new quantum task."""
    
    planner = get_planner()
    
    # Parse priority
    priority_map = {
        'ground': TaskPriority.GROUND_STATE,
        'high': TaskPriority.EXCITED_1,
        'medium': TaskPriority.EXCITED_2,
        'low': TaskPriority.EXCITED_3,
        'deferred': TaskPriority.METASTABLE
    }
    
    # Parse deadline
    deadline = None
    if task_request.deadline:
        try:
            deadline = datetime.fromisoformat(task_request.deadline.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid deadline format")
    
    # Create resources dict
    resources = {}
    if task_request.resources_cpu > 0:
        resources['cpu'] = task_request.resources_cpu
    if task_request.resources_memory > 0:
        resources['memory'] = task_request.resources_memory
    if task_request.resources_io > 0:
        resources['io'] = task_request.resources_io
    
    # Generate unique task ID
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    # Add task to planner
    task = planner.add_task(
        task_id=task_id,
        name=task_request.name,
        description=task_request.description,
        priority=priority_map[task_request.priority],
        estimated_duration=timedelta(hours=task_request.estimated_duration_hours),
        dependencies=set(task_request.dependencies),
        resources_required=resources,
        deadline=deadline
    )
    
    # Broadcast task creation via WebSocket
    background_tasks.add_task(
        websocket_manager.broadcast,
        {
            "type": "task_created",
            "task": {
                "id": task.id,
                "name": task.name,
                "priority": task.priority.name,
                "state": task.state.value
            }
        }
    )
    
    return TaskResponse(
        id=task.id,
        name=task.name,
        description=task.description,
        priority=task.priority.name,
        state=task.state.value,
        estimated_duration_hours=task.estimated_duration.total_seconds() / 3600,
        dependencies=list(task.dependencies),
        entangled_tasks=list(task.entangled_tasks),
        resources_required=task.resources_required,
        urgency_score=task.calculate_urgency_score(),
        created_at=task.created_at.isoformat(),
        deadline=task.deadline.isoformat() if task.deadline else None
    )


@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks():
    """List all quantum tasks."""
    
    planner = get_planner()
    
    tasks = []
    for task in planner.tasks.values():
        tasks.append(TaskResponse(
            id=task.id,
            name=task.name,
            description=task.description,
            priority=task.priority.name,
            state=task.state.value,
            estimated_duration_hours=task.estimated_duration.total_seconds() / 3600,
            dependencies=list(task.dependencies),
            entangled_tasks=list(task.entangled_tasks),
            resources_required=task.resources_required,
            urgency_score=task.calculate_urgency_score(),
            created_at=task.created_at.isoformat(),
            deadline=task.deadline.isoformat() if task.deadline else None
        ))
    
    return sorted(tasks, key=lambda t: t.urgency_score, reverse=True)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get specific quantum task by ID."""
    
    planner = get_planner()
    
    if task_id not in planner.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = planner.tasks[task_id]
    
    return TaskResponse(
        id=task.id,
        name=task.name,
        description=task.description,
        priority=task.priority.name,
        state=task.state.value,
        estimated_duration_hours=task.estimated_duration.total_seconds() / 3600,
        dependencies=list(task.dependencies),
        entangled_tasks=list(task.entangled_tasks),
        resources_required=task.resources_required,
        urgency_score=task.calculate_urgency_score(),
        created_at=task.created_at.isoformat(),
        deadline=task.deadline.isoformat() if task.deadline else None
    )


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str, background_tasks: BackgroundTasks):
    """Delete quantum task and handle entanglements."""
    
    planner = get_planner()
    
    if task_id not in planner.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = planner.tasks[task_id]
    
    # Remove entanglements
    for entangled_id in task.entangled_tasks:
        if entangled_id in planner.tasks:
            planner.tasks[entangled_id].entangled_tasks.discard(task_id)
    
    # Remove task
    del planner.tasks[task_id]
    
    # Broadcast task deletion
    background_tasks.add_task(
        websocket_manager.broadcast,
        {
            "type": "task_deleted",
            "task_id": task_id
        }
    )
    
    return {"message": f"Task {task_id} deleted successfully"}


@app.post("/plan", response_model=PlanResponse)
async def generate_plan(plan_request: PlanRequest, background_tasks: BackgroundTasks):
    """Generate optimal quantum execution plan."""
    
    planner = get_planner()
    
    if not planner.tasks:
        raise HTTPException(status_code=400, detail="No tasks available for planning")
    
    # Set annealing iterations
    planner.max_iterations = plan_request.max_iterations
    
    # Generate plan
    planning_horizon = timedelta(days=plan_request.horizon_days)
    plan = await planner.generate_optimal_plan(planning_horizon)
    
    # Get recommendations
    recommendations = planner.get_task_recommendations()
    
    # Broadcast plan generation
    background_tasks.add_task(
        websocket_manager.broadcast,
        {
            "type": "plan_generated",
            "stats": plan["quantum_stats"],
            "task_count": len(plan["schedule"])
        }
    )
    
    return PlanResponse(
        schedule=plan["schedule"],
        quantum_stats=plan["quantum_stats"],
        metrics=plan["metrics"],
        recommendations=recommendations
    )


@app.post("/plan/execute")
async def execute_plan(background_tasks: BackgroundTasks):
    """Execute the current quantum plan."""
    
    planner = get_planner()
    
    # Check if we have a plan to execute
    if not planner.tasks:
        raise HTTPException(status_code=400, detail="No tasks available for execution")
    
    # Generate fresh plan for execution
    plan = await planner.generate_optimal_plan()
    
    # Execute plan
    background_tasks.add_task(_execute_plan_background, planner, plan)
    
    return {
        "message": "Plan execution started",
        "task_count": len(plan["schedule"]),
        "estimated_completion": (
            datetime.now(timezone.utc) + 
            timedelta(hours=plan["metrics"]["total_estimated_duration"])
        ).isoformat()
    }


async def _execute_plan_background(planner: QuantumTaskPlanner, plan: Dict[str, Any]):
    """Execute plan in background with WebSocket updates."""
    
    try:
        # Notify execution start
        await websocket_manager.broadcast({
            "type": "execution_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tasks": len(plan["schedule"])
        })
        
        # Execute plan with progress updates
        results = await planner.execute_plan(plan)
        
        # Notify execution completion
        await websocket_manager.broadcast({
            "type": "execution_completed", 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_tasks": len(results["completed_tasks"]),
            "failed_tasks": len(results["failed_tasks"])
        })
        
    except Exception as e:
        logger.error(f"Background plan execution failed: {e}")
        await websocket_manager.broadcast({
            "type": "execution_failed",
            "timestamp": datetime.now(timezone.utc).isoformat(), 
            "error": str(e)
        })


@app.get("/tasks/{task_id}/recommendations")
async def get_task_recommendations(task_id: str):
    """Get AI-powered recommendations for specific task."""
    
    planner = get_planner()
    
    if task_id not in planner.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    recommendations = planner.get_task_recommendations(task_id)
    
    return {"task_id": task_id, "recommendations": recommendations}


@app.get("/recommendations")
async def get_system_recommendations():
    """Get system-wide AI recommendations."""
    
    planner = get_planner()
    recommendations = planner.get_task_recommendations()
    
    return {"recommendations": recommendations}


@app.get("/resources")
async def get_resources():
    """Get quantum resource status."""
    
    planner = get_planner()
    
    resources = []
    for resource in planner.resources:
        utilization = ((resource.total_capacity - resource.available_capacity) / resource.total_capacity) * 100
        
        resources.append({
            "name": resource.name,
            "total_capacity": resource.total_capacity,
            "available_capacity": resource.available_capacity,
            "utilization_percent": utilization,
            "quantum_efficiency": resource.quantum_efficiency,
            "coherence_time_minutes": resource.coherence_time.total_seconds() / 60
        })
    
    return {"resources": resources}


@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics."""
    
    dashboard = get_monitoring_dashboard()
    return dashboard.get_dashboard_data()


@app.get("/export/plan")
async def export_plan(format: str = "json"):
    """Export current quantum plan."""
    
    if format != "json":
        raise HTTPException(status_code=400, detail="Only JSON format supported")
    
    planner = get_planner()
    
    if not planner.tasks:
        raise HTTPException(status_code=400, detail="No tasks to export")
    
    # Export to temporary file and return content
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        export_path = planner.export_plan(temp_path, format="json")
        
        with open(export_path) as f:
            plan_data = json.load(f)
        
        return plan_data
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, background_tasks: BackgroundTasks):
    """Resolve quantum alert."""
    
    dashboard = get_monitoring_dashboard()
    
    if alert_id not in dashboard.alert_manager.alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    dashboard.alert_manager.resolve_alert(alert_id)
    
    # Broadcast alert resolution
    background_tasks.add_task(
        websocket_manager.broadcast,
        {
            "type": "alert_resolved",
            "alert_id": alert_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
    
    return {"message": f"Alert {alert_id} resolved"}


@app.get("/alerts")
async def get_alerts(severity: Optional[str] = None):
    """Get active quantum alerts."""
    
    dashboard = get_monitoring_dashboard()
    
    severity_filter = None
    if severity:
        try:
            severity_filter = AlertSeverity(severity.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid severity level")
    
    active_alerts = dashboard.alert_manager.get_active_alerts(severity_filter)
    
    return {
        "alerts": [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source_metric": alert.source_metric,
                "entangled_alerts": alert.entangled_alerts
            }
            for alert in active_alerts
        ]
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time quantum updates."""
    
    await websocket_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to Quantum Task Planner"
        }))
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages with timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Parse and handle client message
                try:
                    data = json.loads(message)
                    await _handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }))
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


async def _handle_websocket_message(websocket: WebSocket, data: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    
    message_type = data.get("type")
    
    if message_type == "subscribe_metrics":
        # Send current metrics
        dashboard = get_monitoring_dashboard()
        metrics_data = dashboard.get_dashboard_data()
        
        await websocket.send_text(json.dumps({
            "type": "metrics_update",
            "data": metrics_data
        }, default=str))
    
    elif message_type == "subscribe_tasks":
        # Send current tasks
        planner = get_planner()
        tasks_data = []
        
        for task in planner.tasks.values():
            tasks_data.append({
                "id": task.id,
                "name": task.name,
                "priority": task.priority.name,
                "state": task.state.value,
                "urgency_score": task.calculate_urgency_score()
            })
        
        await websocket.send_text(json.dumps({
            "type": "tasks_update",
            "data": tasks_data
        }))
    
    elif message_type == "ping":
        # Respond to ping
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))


# Background task for monitoring updates
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on app startup."""
    
    logger.info("Starting Quantum Task Planner API")
    
    # Start monitoring dashboard
    dashboard = get_monitoring_dashboard()
    await dashboard.start_monitoring()
    
    # Start periodic WebSocket updates
    asyncio.create_task(_periodic_websocket_updates())


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on app shutdown."""
    
    logger.info("Shutting down Quantum Task Planner API")
    
    # Stop monitoring
    dashboard = get_monitoring_dashboard()
    await dashboard.stop_monitoring()


async def _periodic_websocket_updates():
    """Send periodic updates via WebSocket."""
    
    try:
        while True:
            # Wait 10 seconds between updates
            await asyncio.sleep(10)
            
            # Send metrics update
            dashboard = get_monitoring_dashboard()
            metrics_data = dashboard.get_dashboard_data()
            
            await websocket_manager.broadcast({
                "type": "periodic_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "quantum_coherence": metrics_data["quantum_state"]["coherence"],
                    "active_alerts": len(metrics_data["active_alerts"]),
                    "health_status": metrics_data["health_status"]["overall_status"]
                }
            })
            
    except asyncio.CancelledError:
        logger.info("Periodic WebSocket updates cancelled")
    except Exception as e:
        logger.error(f"Periodic WebSocket updates failed: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with quantum context."""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "quantum_timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with quantum error tracking."""
    
    logger.error(f"Unhandled exception in quantum API: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal quantum computation error",
            "quantum_timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
            "error_type": type(exc).__name__
        }
    )


# Development server function
def run_quantum_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    reload: bool = False
):
    """Run quantum task planner API server."""
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info(f"Starting Quantum Task Planner API on {host}:{port}")
    
    uvicorn.run(
        "testgen_copilot.quantum_api:app",
        host=host,
        port=port,
        debug=debug,
        reload=reload,
        log_level="debug" if debug else "info"
    )


if __name__ == "__main__":
    run_quantum_api(debug=True, reload=True)