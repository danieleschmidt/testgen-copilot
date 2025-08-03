"""External service integrations for TestGen Copilot."""

from .github import GitHubClient, GitHubIntegration, GitHubWebhookHandler
from .notifications import (
    NotificationService, EmailNotifier, SlackNotifier, 
    WebhookNotifier, NotificationChannel
)
from .ci_cd import CICDIntegration, GitHubActionsClient, GitLabCIClient
from .auth import AuthProvider, OAuthHandler, JWTManager
from .cloud import CloudStorageClient, S3Client, GCSClient
from .webhooks import WebhookManager, WebhookEvent, WebhookProcessor

__all__ = [
    "GitHubClient",
    "GitHubIntegration", 
    "GitHubWebhookHandler",
    "NotificationService",
    "EmailNotifier",
    "SlackNotifier",
    "WebhookNotifier",
    "NotificationChannel",
    "CICDIntegration",
    "GitHubActionsClient",
    "GitLabCIClient",
    "AuthProvider",
    "OAuthHandler", 
    "JWTManager",
    "CloudStorageClient",
    "S3Client",
    "GCSClient",
    "WebhookManager",
    "WebhookEvent",
    "WebhookProcessor"
]