"""Middleware components for TestGen Copilot API."""

from __future__ import annotations

import os
import time
import uuid
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging_config import get_logger
from .models import ErrorResponse


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware for logging HTTP requests and responses."""
    logger = get_logger("testgen_copilot.api.middleware")
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log incoming request
    logger.info("Incoming request", {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "user_agent": request.headers.get("user-agent"),
        "client_ip": request.client.host if request.client else None,
        "content_length": request.headers.get("content-length")
    })
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info("Request completed", {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time, 2),
            "response_size": response.headers.get("content-length")
        })
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        
        logger.error("Request failed", {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time_ms": round(processing_time, 2)
        })
        
        # Return error response
        error_response = ErrorResponse(
            message="Internal server error",
            error_type=type(e).__name__,
            request_id=request_id,
            details={"processing_time_ms": round(processing_time, 2)}
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )


async def auth_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware for API authentication using API keys."""
    logger = get_logger("testgen_copilot.api.auth")
    
    # Skip authentication for health and docs endpoints
    skip_auth_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/"]
    if any(request.url.path.startswith(path) for path in skip_auth_paths):
        return await call_next(request)
    
    # Check for API key
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    
    # Remove "Bearer " prefix if present
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:]
    
    if not api_key:
        logger.warning("Missing API key", {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        })
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": True,
                "message": "API key required",
                "details": {
                    "required_header": "X-API-Key or Authorization",
                    "format": "X-API-Key: your-api-key or Authorization: Bearer your-api-key"
                }
            }
        )
    
    # Validate API key (implement your validation logic here)
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if valid_api_keys and api_key not in valid_api_keys:
        logger.warning("Invalid API key", {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None,
            "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else api_key
        })
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": True,
                "message": "Invalid API key"
            }
        )
    
    # API key is valid, continue with request
    logger.debug("API key validated", {
        "path": request.url.path,
        "method": request.method,
        "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else api_key
    })
    
    # Add user context to request
    request.state.authenticated = True
    request.state.api_key = api_key
    
    return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger("testgen_copilot.api.ratelimit")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting based on client IP."""
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip]["requests"] = [
                req_time for req_time in self.request_counts[ip]["requests"]
                if req_time > cutoff_time
            ]
            
            if not self.request_counts[ip]["requests"]:
                del self.request_counts[ip]
        
        # Initialize or update request count for client
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {"requests": []}
        
        # Check rate limit
        request_times = self.request_counts[client_ip]["requests"]
        if len(request_times) >= self.calls_per_minute:
            self.logger.warning("Rate limit exceeded", {
                "client_ip": client_ip,
                "requests_in_window": len(request_times),
                "limit": self.calls_per_minute,
                "path": request.url.path
            })
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": True,
                    "message": "Rate limit exceeded",
                    "details": {
                        "limit": self.calls_per_minute,
                        "window": "1 minute",
                        "retry_after": 60 - (current_time - min(request_times))
                    }
                },
                headers={"Retry-After": "60"}
            )
        
        # Record this request
        self.request_counts[client_ip]["requests"].append(current_time)
        
        return await call_next(request)


async def cors_handler(request: Request, call_next: Callable) -> Response:
    """Custom CORS handler for additional security."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add API version header
    response.headers["X-API-Version"] = "v1"
    
    return response


async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler for unhandled exceptions."""
    logger = get_logger("testgen_copilot.api.error")
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Log the error
    logger.error("Unhandled exception", {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "client_ip": request.client.host if request.client else None
    }, exc_info=True)
    
    # Determine error details based on environment
    debug_mode = os.getenv("TESTGEN_ENV", "production").lower() == "development"
    
    error_response = ErrorResponse(
        message="An unexpected error occurred",
        error_type=type(exc).__name__,
        request_id=request_id,
        details={
            "error_detail": str(exc) if debug_mode else None,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict(),
        headers={"X-Request-ID": request_id}
    )