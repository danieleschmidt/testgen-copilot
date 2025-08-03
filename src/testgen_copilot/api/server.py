"""FastAPI server implementation for TestGen Copilot API."""

from __future__ import annotations

import os
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

from .routes import analysis_bp, sessions_bp, security_bp, metrics_bp, health_bp
from .middleware import error_handler, request_logging_middleware, auth_middleware
from ..database import run_migrations, get_database, close_database
from ..logging_config import get_logger, configure_logging
from ..version import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    logger = get_logger("testgen_copilot.api.server")
    
    # Startup
    logger.info("Starting TestGen Copilot API server", {
        "version": __version__,
        "environment": os.getenv("TESTGEN_ENV", "development")
    })
    
    try:
        # Initialize database
        logger.info("Initializing database")
        migrations_applied = run_migrations()
        logger.info("Database initialized", {
            "migrations_applied": migrations_applied
        })
        
        # Additional startup tasks
        logger.info("API server startup complete")
        
    except Exception as e:
        logger.error("Failed to start API server", {
            "error": str(e)
        })
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down TestGen Copilot API server")
    try:
        close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning("Error during shutdown", {
            "error": str(e)
        })
    
    logger.info("API server shutdown complete")


class TestGenAPI:
    """Main API application class."""
    
    def __init__(
        self,
        title: str = "TestGen Copilot API",
        description: str = "AI-powered test generation and security analysis API",
        version: str = __version__,
        debug: bool = False
    ):
        self.app = FastAPI(
            title=title,
            description=description,
            version=version,
            debug=debug,
            lifespan=lifespan,
            docs_url="/docs" if debug else None,
            redoc_url="/redoc" if debug else None
        )
        
        self.logger = get_logger("testgen_copilot.api")
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
    
    def _setup_middleware(self) -> None:
        """Configure middleware for the application."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Custom middleware
        self.app.middleware("http")(request_logging_middleware)
        
        # Authentication middleware (if API keys are enabled)
        if os.getenv("ENABLE_API_AUTH", "false").lower() == "true":
            self.app.middleware("http")(auth_middleware)
    
    def _setup_routes(self) -> None:
        """Register API route blueprints."""
        # Health check (always available)
        self.app.include_router(health_bp, prefix="/health", tags=["health"])
        
        # Main API routes
        self.app.include_router(analysis_bp, prefix="/api/v1/analysis", tags=["analysis"])
        self.app.include_router(sessions_bp, prefix="/api/v1/sessions", tags=["sessions"])
        self.app.include_router(security_bp, prefix="/api/v1/security", tags=["security"])
        self.app.include_router(metrics_bp, prefix="/api/v1/metrics", tags=["metrics"])
        
        # Root endpoint
        @self.app.get("/", response_model=Dict[str, Any])
        async def root():
            """API root endpoint with basic information."""
            return {
                "name": "TestGen Copilot API",
                "version": __version__,
                "status": "running",
                "docs_url": "/docs" if self.app.debug else None,
                "health_url": "/health"
            }
    
    def _setup_error_handlers(self) -> None:
        """Configure global error handlers."""
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Handle unexpected exceptions."""
            return await error_handler(request, exc)
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions."""
            self.logger.warning("HTTP exception occurred", {
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
                "method": request.method
            })
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "message": exc.detail,
                    "status_code": exc.status_code,
                    "path": request.url.path
                }
            )
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info"
    ) -> None:
        """Run the API server."""
        self.logger.info("Starting API server", {
            "host": host,
            "port": port,
            "workers": workers,
            "log_level": log_level
        })
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True
        )


def create_app(
    debug: bool = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> FastAPI:
    """Factory function to create FastAPI application."""
    if debug is None:
        debug = os.getenv("TESTGEN_ENV", "development") == "development"
    
    # Configure logging
    log_level = os.getenv("TESTGEN_LOG_LEVEL", "INFO")
    configure_logging(
        level=log_level,
        format_type="structured" if not debug else "simple",
        enable_console=True
    )
    
    # Create API instance
    api = TestGenAPI(debug=debug)
    
    # Apply configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(api.app, key, value)
    
    return api.app


if __name__ == "__main__":
    # Development server
    app = create_app(debug=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info"
    )