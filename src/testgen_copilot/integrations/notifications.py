"""Notification service integrations for TestGen Copilot."""

from __future__ import annotations

import json
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, EmailStr

from ..logging_config import get_logger


class NotificationChannel(str, Enum):
    """Notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    DISCORD = "discord"


class NotificationSeverity(str, Enum):
    """Notification severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NotificationMessage:
    """Notification message data."""
    title: str
    content: str
    severity: NotificationSeverity = NotificationSeverity.INFO
    recipient: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NotificationTemplate(BaseModel):
    """Template for notification messages."""
    subject_template: str
    content_template: str
    channel: NotificationChannel
    severity: NotificationSeverity = NotificationSeverity.INFO
    
    def render(self, context: Dict[str, Any]) -> NotificationMessage:
        """Render template with context data."""
        try:
            subject = self.subject_template.format(**context)
            content = self.content_template.format(**context)
            
            return NotificationMessage(
                title=subject,
                content=content,
                severity=self.severity,
                metadata=context
            )
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")


class BaseNotifier(ABC):
    """Base class for notification providers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"testgen_copilot.integrations.notifications.{name}")
    
    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """Send notification message."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if notifier is properly configured."""
        pass


class EmailNotifier(BaseNotifier):
    """Email notification provider."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        from_email: Optional[str] = None
    ):
        super().__init__("email")
        
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.use_tls = use_tls
        self.from_email = from_email or os.getenv("SMTP_FROM_EMAIL") or self.username
    
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return all([
            self.smtp_host,
            self.smtp_port,
            self.username,
            self.password,
            self.from_email
        ])
    
    async def send(self, message: NotificationMessage) -> bool:
        """Send email notification."""
        if not self.is_configured():
            self.logger.error("Email notifier not properly configured")
            return False
        
        if not message.recipient:
            self.logger.error("No recipient specified for email")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = message.recipient
            msg['Subject'] = message.title
            
            # Add severity indicator to subject
            severity_indicators = {
                NotificationSeverity.INFO: "ℹ️",
                NotificationSeverity.WARNING: "⚠️",
                NotificationSeverity.ERROR: "❌",
                NotificationSeverity.CRITICAL: "🚨"
            }
            indicator = severity_indicators.get(message.severity, "")
            if indicator:
                msg['Subject'] = f"{indicator} {message.title}"
            
            # Create HTML content
            html_content = self._create_html_content(message)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info("Email sent successfully", {
                "recipient": message.recipient,
                "subject": message.title,
                "severity": message.severity
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to send email", {
                "recipient": message.recipient,
                "subject": message.title,
                "error": str(e)
            })
            return False
    
    def _create_html_content(self, message: NotificationMessage) -> str:
        """Create HTML email content."""
        severity_colors = {
            NotificationSeverity.INFO: "#007bff",
            NotificationSeverity.WARNING: "#ffc107", 
            NotificationSeverity.ERROR: "#dc3545",
            NotificationSeverity.CRITICAL: "#ff1744"
        }
        
        color = severity_colors.get(message.severity, "#007bff")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">{message.title}</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Severity: {message.severity.value.title()}</p>
                </div>
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 0 0 5px 5px; border: 1px solid #dee2e6;">
                    <div style="background-color: white; padding: 20px; border-radius: 5px;">
                        {message.content.replace('\n', '<br>')}
                    </div>
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d;">
                        <p>Sent by TestGen Copilot at {datetime.now(timezone.utc).isoformat()}</p>
                        {self._render_metadata(message.metadata)}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _render_metadata(self, metadata: Dict[str, Any]) -> str:
        """Render metadata as HTML."""
        if not metadata:
            return ""
        
        items = []
        for key, value in metadata.items():
            if key not in ['recipient', 'content', 'title']:  # Skip duplicated fields
                items.append(f"<strong>{key}:</strong> {value}")
        
        if items:
            return "<p>Additional Information:</p><ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
        return ""


class SlackNotifier(BaseNotifier):
    """Slack notification provider."""
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        token: Optional[str] = None,
        default_channel: Optional[str] = None
    ):
        super().__init__("slack")
        
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.token = token or os.getenv("SLACK_TOKEN")
        self.default_channel = default_channel or os.getenv("SLACK_DEFAULT_CHANNEL")
        
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def is_configured(self) -> bool:
        """Check if Slack is properly configured."""
        return bool(self.webhook_url or (self.token and self.default_channel))
    
    async def send(self, message: NotificationMessage) -> bool:
        """Send Slack notification."""
        if not self.is_configured():
            self.logger.error("Slack notifier not properly configured")
            return False
        
        try:
            if self.webhook_url:
                return await self._send_webhook(message)
            else:
                return await self._send_api(message)
        except Exception as e:
            self.logger.error("Failed to send Slack notification", {
                "error": str(e),
                "title": message.title
            })
            return False
    
    async def _send_webhook(self, message: NotificationMessage) -> bool:
        """Send notification via webhook."""
        # Create Slack message payload
        payload = self._create_slack_payload(message)
        
        response = await self.client.post(self.webhook_url, json=payload)
        response.raise_for_status()
        
        self.logger.info("Slack webhook sent successfully", {
            "title": message.title,
            "severity": message.severity
        })
        
        return True
    
    async def _send_api(self, message: NotificationMessage) -> bool:
        """Send notification via Slack API."""
        channel = message.recipient or self.default_channel
        if not channel:
            self.logger.error("No Slack channel specified")
            return False
        
        payload = self._create_slack_payload(message)
        payload["channel"] = channel
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = await self.client.post(
            "https://slack.com/api/chat.postMessage",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        if not result.get("ok"):
            raise Exception(f"Slack API error: {result.get('error')}")
        
        self.logger.info("Slack API message sent successfully", {
            "channel": channel,
            "title": message.title
        })
        
        return True
    
    def _create_slack_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Create Slack message payload."""
        # Map severity to Slack colors
        severity_colors = {
            NotificationSeverity.INFO: "#36a64f",      # Green
            NotificationSeverity.WARNING: "#ff9500",   # Orange
            NotificationSeverity.ERROR: "#ff0000",     # Red
            NotificationSeverity.CRITICAL: "#8B0000"   # Dark Red
        }
        
        # Map severity to emojis
        severity_emojis = {
            NotificationSeverity.INFO: ":information_source:",
            NotificationSeverity.WARNING: ":warning:",
            NotificationSeverity.ERROR: ":x:",
            NotificationSeverity.CRITICAL: ":rotating_light:"
        }
        
        color = severity_colors.get(message.severity, "#36a64f")
        emoji = severity_emojis.get(message.severity, ":information_source:")
        
        # Create attachment
        attachment = {
            "color": color,
            "title": f"{emoji} {message.title}",
            "text": message.content,
            "footer": "TestGen Copilot",
            "ts": int(datetime.now(timezone.utc).timestamp()),
            "fields": []
        }
        
        # Add metadata fields
        if message.metadata:
            for key, value in message.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    attachment["fields"].append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True
                    })
        
        return {
            "attachments": [attachment],
            "username": "TestGen Copilot",
            "icon_emoji": ":robot_face:"
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class WebhookNotifier(BaseNotifier):
    """Generic webhook notification provider."""
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        super().__init__("webhook")
        
        self.webhook_url = webhook_url or os.getenv("WEBHOOK_URL")
        self.headers = headers or {}
        self.timeout = timeout
        
        # Add authentication header if token is available
        webhook_token = os.getenv("WEBHOOK_TOKEN")
        if webhook_token:
            self.headers["Authorization"] = f"Bearer {webhook_token}"
        
        self.client = httpx.AsyncClient(timeout=timeout)
    
    def is_configured(self) -> bool:
        """Check if webhook is properly configured."""
        return bool(self.webhook_url)
    
    async def send(self, message: NotificationMessage) -> bool:
        """Send webhook notification."""
        if not self.is_configured():
            self.logger.error("Webhook notifier not properly configured")
            return False
        
        try:
            payload = {
                "title": message.title,
                "content": message.content,
                "severity": message.severity.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": message.metadata or {}
            }
            
            if message.recipient:
                payload["recipient"] = message.recipient
            
            response = await self.client.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            self.logger.info("Webhook sent successfully", {
                "url": self.webhook_url,
                "title": message.title,
                "status_code": response.status_code
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to send webhook", {
                "url": self.webhook_url,
                "error": str(e),
                "title": message.title
            })
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class NotificationService:
    """Central notification service managing multiple providers."""
    
    def __init__(self):
        self.logger = get_logger("testgen_copilot.integrations.notifications")
        self.notifiers: Dict[NotificationChannel, BaseNotifier] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # Initialize default notifiers
        self._initialize_notifiers()
        self._load_default_templates()
    
    def _initialize_notifiers(self):
        """Initialize notification providers."""
        # Email notifier
        email_notifier = EmailNotifier()
        if email_notifier.is_configured():
            self.notifiers[NotificationChannel.EMAIL] = email_notifier
            self.logger.info("Email notifier configured")
        
        # Slack notifier
        slack_notifier = SlackNotifier()
        if slack_notifier.is_configured():
            self.notifiers[NotificationChannel.SLACK] = slack_notifier
            self.logger.info("Slack notifier configured")
        
        # Webhook notifier
        webhook_notifier = WebhookNotifier()
        if webhook_notifier.is_configured():
            self.notifiers[NotificationChannel.WEBHOOK] = webhook_notifier
            self.logger.info("Webhook notifier configured")
    
    def _load_default_templates(self):
        """Load default notification templates."""
        self.templates.update({
            "analysis_complete": NotificationTemplate(
                subject_template="TestGen Analysis Complete - {project_name}",
                content_template="""
Analysis completed for project: {project_name}

Results:
- Files analyzed: {files_analyzed}
- Tests generated: {tests_generated}
- Coverage: {coverage_percentage:.1f}%
- Quality Score: {quality_score:.1f}%
- Security Issues: {security_issues}

Session ID: {session_id}
Duration: {processing_time:.1f} seconds
""",
                channel=NotificationChannel.EMAIL,
                severity=NotificationSeverity.INFO
            ),
            
            "security_issues_found": NotificationTemplate(
                subject_template="Security Issues Found - {project_name}",
                content_template="""
Security analysis found {security_issues} issues in {project_name}:

Critical: {critical_issues}
High: {high_issues}
Medium: {medium_issues}
Low: {low_issues}

Please review and address these security concerns.

Session ID: {session_id}
""",
                channel=NotificationChannel.SLACK,
                severity=NotificationSeverity.WARNING
            ),
            
            "analysis_failed": NotificationTemplate(
                subject_template="TestGen Analysis Failed - {project_name}",
                content_template="""
Analysis failed for project: {project_name}

Error: {error_message}
Session ID: {session_id}
Failed Files: {failed_files}

Please check the logs for more details.
""",
                channel=NotificationChannel.EMAIL,
                severity=NotificationSeverity.ERROR
            )
        })
    
    def add_notifier(self, channel: NotificationChannel, notifier: BaseNotifier):
        """Add a custom notifier."""
        self.notifiers[channel] = notifier
        self.logger.info("Added custom notifier", {"channel": channel.value})
    
    def add_template(self, name: str, template: NotificationTemplate):
        """Add a custom template."""
        self.templates[name] = template
        self.logger.info("Added custom template", {"name": name})
    
    async def send(
        self,
        message: NotificationMessage,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[NotificationChannel, bool]:
        """Send notification to specified channels."""
        if channels is None:
            channels = list(self.notifiers.keys())
        
        results = {}
        for channel in channels:
            if channel in self.notifiers:
                try:
                    success = await self.notifiers[channel].send(message)
                    results[channel] = success
                except Exception as e:
                    self.logger.error("Notification failed", {
                        "channel": channel.value,
                        "error": str(e)
                    })
                    results[channel] = False
            else:
                self.logger.warning("Notifier not configured", {
                    "channel": channel.value
                })
                results[channel] = False
        
        return results
    
    async def send_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        recipient: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[NotificationChannel, bool]:
        """Send notification using a template."""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        message = template.render(context)
        
        if recipient:
            message.recipient = recipient
        
        # Use template's preferred channel if not specified
        if channels is None:
            channels = [template.channel]
        
        return await self.send(message, channels)
    
    async def notify_analysis_complete(
        self,
        project_name: str,
        session_id: str,
        metrics: Dict[str, Any],
        recipient: Optional[str] = None
    ):
        """Send analysis completion notification."""
        context = {
            "project_name": project_name,
            "session_id": session_id,
            "files_analyzed": metrics.get("files_analyzed", 0),
            "tests_generated": metrics.get("tests_generated", 0),
            "coverage_percentage": metrics.get("coverage_percentage", 0),
            "quality_score": metrics.get("quality_score", 0),
            "security_issues": metrics.get("security_issues_found", 0),
            "processing_time": metrics.get("processing_time_seconds", 0)
        }
        
        await self.send_template(
            "analysis_complete",
            context,
            recipient=recipient
        )
    
    async def notify_security_issues(
        self,
        project_name: str,
        session_id: str,
        security_summary: Dict[str, int],
        recipient: Optional[str] = None
    ):
        """Send security issues notification."""
        context = {
            "project_name": project_name,
            "session_id": session_id,
            "security_issues": security_summary.get("total_issues", 0),
            "critical_issues": security_summary.get("critical_issues", 0),
            "high_issues": security_summary.get("high_issues", 0),
            "medium_issues": security_summary.get("medium_issues", 0),
            "low_issues": security_summary.get("low_issues", 0)
        }
        
        await self.send_template(
            "security_issues_found",
            context,
            recipient=recipient,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
        )
    
    async def notify_analysis_failed(
        self,
        project_name: str,
        session_id: str,
        error_message: str,
        failed_files: int = 0,
        recipient: Optional[str] = None
    ):
        """Send analysis failure notification."""
        context = {
            "project_name": project_name,
            "session_id": session_id,
            "error_message": error_message,
            "failed_files": failed_files
        }
        
        await self.send_template(
            "analysis_failed",
            context,
            recipient=recipient,
            channels=[NotificationChannel.EMAIL]
        )
    
    async def close(self):
        """Close all notifiers."""
        for notifier in self.notifiers.values():
            if hasattr(notifier, 'close'):
                await notifier.close()