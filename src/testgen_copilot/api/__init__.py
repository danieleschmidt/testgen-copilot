"""REST API layer for TestGen Copilot."""

from .server import create_app, TestGenAPI
from .routes import (
    analysis_bp,
    sessions_bp,
    security_bp, 
    metrics_bp,
    health_bp
)
from .middleware import (
    error_handler,
    cors_handler,
    auth_middleware,
    request_logging_middleware
)
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    SecurityScanRequest,
    SecurityScanResponse,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    "create_app",
    "TestGenAPI",
    "analysis_bp",
    "sessions_bp", 
    "security_bp",
    "metrics_bp",
    "health_bp",
    "error_handler",
    "cors_handler",
    "auth_middleware",
    "request_logging_middleware",
    "AnalysisRequest",
    "AnalysisResponse",
    "SecurityScanRequest",
    "SecurityScanResponse",
    "HealthResponse",
    "ErrorResponse"
]