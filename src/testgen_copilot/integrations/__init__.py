"""External service integrations for TestGen Copilot."""

from .auth import AuthProvider, JWTManager, OAuthHandler
from .ci_cd import CICDIntegration, GitHubActionsClient, GitLabCIClient
from .cloud import CloudStorageClient, GCSClient, S3Client
from .github import GitHubClient, GitHubIntegration, GitHubWebhookHandler
from .notifications import (
    EmailNotifier,
    NotificationChannel,
    NotificationService,
    SlackNotifier,
    WebhookNotifier,
)
from .webhooks import WebhookEvent, WebhookManager, WebhookProcessor

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
