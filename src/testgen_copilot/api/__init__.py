"""REST API layer for TestGen Copilot."""

from .middleware import auth_middleware, cors_handler, error_handler, request_logging_middleware
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    SecurityScanRequest,
    SecurityScanResponse,
)
from .routes import analysis_bp, health_bp, metrics_bp, security_bp, sessions_bp
from .server import TestGenAPI, create_app

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
