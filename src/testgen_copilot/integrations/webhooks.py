"""Webhook management for TestGen Copilot."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from ..logging_config import get_logger


class WebhookEvent(str, Enum):
    """Webhook event types."""
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    SECURITY_ISSUES_FOUND = "security.issues_found"
    TESTS_GENERATED = "tests.generated"
    COVERAGE_UPDATED = "coverage.updated"


@dataclass
class WebhookPayload:
    """Webhook payload data."""
    event: WebhookEvent
    timestamp: datetime
    data: Dict[str, Any]
    webhook_id: str = None

    def __post_init__(self):
        if self.webhook_id is None:
            self.webhook_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "webhook_id": self.webhook_id,
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    url: str
    secret: Optional[str] = None
    events: List[WebhookEvent] = None
    active: bool = True
    retry_count: int = 3
    timeout: int = 30

    def __post_init__(self):
        if self.events is None:
            self.events = list(WebhookEvent)


class WebhookProcessor:
    """Process and deliver webhooks."""

    def __init__(self):
        self.logger = get_logger("testgen_copilot.integrations.webhooks.processor")
        self.client = httpx.AsyncClient()

    async def send_webhook(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> bool:
        """Send webhook to endpoint."""
        if not endpoint.active:
            self.logger.debug("Webhook endpoint inactive", {"url": endpoint.url})
            return False

        if payload.event not in endpoint.events:
            self.logger.debug("Event not subscribed", {
                "event": payload.event.value,
                "url": endpoint.url
            })
            return False

        # Prepare payload
        payload_dict = payload.to_dict()
        payload_json = json.dumps(payload_dict, sort_keys=True)

        # Create headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TestGen-Copilot-Webhook/1.0",
            "X-Webhook-ID": payload.webhook_id,
            "X-Webhook-Event": payload.event.value,
            "X-Webhook-Timestamp": payload.timestamp.isoformat()
        }

        # Add signature if secret is provided
        if endpoint.secret:
            signature = self._create_signature(payload_json, endpoint.secret)
            headers["X-Webhook-Signature"] = signature

        # Send webhook with retries
        for attempt in range(endpoint.retry_count + 1):
            try:
                response = await self.client.post(
                    endpoint.url,
                    content=payload_json,
                    headers=headers,
                    timeout=endpoint.timeout
                )

                if response.status_code == 200:
                    self.logger.info("Webhook delivered successfully", {
                        "url": endpoint.url,
                        "event": payload.event.value,
                        "webhook_id": payload.webhook_id,
                        "attempt": attempt + 1
                    })
                    return True
                else:
                    self.logger.warning("Webhook delivery failed", {
                        "url": endpoint.url,
                        "status_code": response.status_code,
                        "webhook_id": payload.webhook_id,
                        "attempt": attempt + 1
                    })

            except Exception as e:
                self.logger.error("Webhook delivery error", {
                    "url": endpoint.url,
                    "error": str(e),
                    "webhook_id": payload.webhook_id,
                    "attempt": attempt + 1
                })

            # Don't retry if this was the last attempt
            if attempt < endpoint.retry_count:
                # Exponential backoff
                import asyncio
                await asyncio.sleep(2 ** attempt)

        self.logger.error("Webhook delivery failed after all retries", {
            "url": endpoint.url,
            "webhook_id": payload.webhook_id,
            "max_attempts": endpoint.retry_count + 1
        })

        return False

    def _create_signature(self, payload: str, secret: str) -> str:
        """Create HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"sha256={signature}"

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class WebhookManager:
    """Manage webhook endpoints and delivery."""

    def __init__(self):
        self.logger = get_logger("testgen_copilot.integrations.webhooks")
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.processor = WebhookProcessor()

        # Event handlers
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = {}

    def add_endpoint(self, endpoint_id: str, endpoint: WebhookEndpoint):
        """Add webhook endpoint."""
        self.endpoints[endpoint_id] = endpoint

        self.logger.info("Added webhook endpoint", {
            "endpoint_id": endpoint_id,
            "url": endpoint.url,
            "events": [e.value for e in endpoint.events],
            "active": endpoint.active
        })

    def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove webhook endpoint."""
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints.pop(endpoint_id)

            self.logger.info("Removed webhook endpoint", {
                "endpoint_id": endpoint_id,
                "url": endpoint.url
            })

            return True

        return False

    def update_endpoint(self, endpoint_id: str, **kwargs) -> bool:
        """Update webhook endpoint configuration."""
        if endpoint_id not in self.endpoints:
            return False

        endpoint = self.endpoints[endpoint_id]

        for key, value in kwargs.items():
            if hasattr(endpoint, key):
                setattr(endpoint, key, value)

        self.logger.info("Updated webhook endpoint", {
            "endpoint_id": endpoint_id,
            "updates": list(kwargs.keys())
        })

        return True

    def list_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """List all webhook endpoints."""
        return {
            endpoint_id: {
                "url": endpoint.url,
                "events": [e.value for e in endpoint.events],
                "active": endpoint.active,
                "retry_count": endpoint.retry_count,
                "timeout": endpoint.timeout
            }
            for endpoint_id, endpoint in self.endpoints.items()
        }

    async def trigger_event(self, event: WebhookEvent, data: Dict[str, Any]):
        """Trigger webhook event to all subscribed endpoints."""
        if not self.endpoints:
            self.logger.debug("No webhook endpoints configured")
            return

        payload = WebhookPayload(
            event=event,
            timestamp=datetime.now(timezone.utc),
            data=data
        )

        self.logger.info("Triggering webhook event", {
            "event": event.value,
            "webhook_id": payload.webhook_id,
            "endpoint_count": len(self.endpoints)
        })

        # Send to all subscribed endpoints
        for endpoint_id, endpoint in self.endpoints.items():
            if event in endpoint.events and endpoint.active:
                # Send webhook asynchronously (fire and forget)
                import asyncio
                asyncio.create_task(self._send_webhook_safe(endpoint_id, endpoint, payload))

    async def _send_webhook_safe(
        self,
        endpoint_id: str,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ):
        """Send webhook with error handling."""
        try:
            await self.processor.send_webhook(endpoint, payload)
        except Exception as e:
            self.logger.error("Webhook sending failed", {
                "endpoint_id": endpoint_id,
                "url": endpoint.url,
                "error": str(e),
                "webhook_id": payload.webhook_id
            })

    def add_event_handler(
        self,
        event: WebhookEvent,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """Add event handler for webhook events."""
        if event not in self.event_handlers:
            self.event_handlers[event] = []

        self.event_handlers[event].append(handler)

        self.logger.info("Added event handler", {
            "event": event.value,
            "handler_count": len(self.event_handlers[event])
        })

    async def handle_event(self, event: WebhookEvent, data: Dict[str, Any]):
        """Handle event with registered handlers."""
        # Trigger webhooks
        await self.trigger_event(event, data)

        # Run event handlers
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error("Event handler failed", {
                        "event": event.value,
                        "error": str(e)
                    })

    # Convenience methods for common events
    async def notify_analysis_started(
        self,
        session_id: str,
        project_name: str,
        file_count: int
    ):
        """Notify that analysis has started."""
        await self.handle_event(WebhookEvent.ANALYSIS_STARTED, {
            "session_id": session_id,
            "project_name": project_name,
            "file_count": file_count
        })

    async def notify_analysis_completed(
        self,
        session_id: str,
        project_name: str,
        metrics: Dict[str, Any]
    ):
        """Notify that analysis has completed."""
        await self.handle_event(WebhookEvent.ANALYSIS_COMPLETED, {
            "session_id": session_id,
            "project_name": project_name,
            "metrics": metrics
        })

    async def notify_security_issues_found(
        self,
        session_id: str,
        project_name: str,
        issue_count: int,
        severity_breakdown: Dict[str, int]
    ):
        """Notify that security issues were found."""
        await self.handle_event(WebhookEvent.SECURITY_ISSUES_FOUND, {
            "session_id": session_id,
            "project_name": project_name,
            "issue_count": issue_count,
            "severity_breakdown": severity_breakdown
        })

    async def notify_tests_generated(
        self,
        session_id: str,
        project_name: str,
        test_count: int,
        files_with_tests: List[str]
    ):
        """Notify that tests were generated."""
        await self.handle_event(WebhookEvent.TESTS_GENERATED, {
            "session_id": session_id,
            "project_name": project_name,
            "test_count": test_count,
            "files_with_tests": files_with_tests
        })

    async def close(self):
        """Close webhook manager and cleanup resources."""
        await self.processor.close()
