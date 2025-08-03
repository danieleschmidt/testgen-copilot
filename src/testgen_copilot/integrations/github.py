"""GitHub integration for TestGen Copilot."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import httpx
from pydantic import BaseModel

from ..logging_config import get_logger
from ..database import get_database, SessionRepository, AnalysisRepository


class GitHubEventType(str, Enum):
    """GitHub webhook event types."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    ISSUES = "issues"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"
    REPOSITORY = "repository"
    RELEASE = "release"


class PullRequestAction(str, Enum):
    """Pull request actions."""
    OPENED = "opened"
    CLOSED = "closed"
    SYNCHRONIZE = "synchronize"
    REOPENED = "reopened"
    READY_FOR_REVIEW = "ready_for_review"


@dataclass
class GitHubRepository:
    """GitHub repository information."""
    id: int
    name: str
    full_name: str
    owner: str
    private: bool
    clone_url: str
    ssh_url: str
    default_branch: str
    language: Optional[str] = None
    description: Optional[str] = None


@dataclass
class GitHubPullRequest:
    """GitHub pull request information."""
    number: int
    title: str
    body: Optional[str]
    state: str
    head_ref: str
    base_ref: str
    head_sha: str
    base_sha: str
    author: str
    created_at: datetime
    updated_at: datetime
    mergeable: Optional[bool] = None
    draft: bool = False


class GitHubClient:
    """Client for GitHub API interactions."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com"
    ):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url
        self.logger = get_logger("testgen_copilot.integrations.github")
        
        if not self.token:
            self.logger.warning("No GitHub token provided - some features may be limited")
        
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "TestGen-Copilot/1.0",
                **({} if not self.token else {"Authorization": f"Bearer {self.token}"})
            }
        )
    
    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information."""
        try:
            response = await self.client.get(f"/repos/{owner}/{repo}")
            response.raise_for_status()
            
            data = response.json()
            return GitHubRepository(
                id=data["id"],
                name=data["name"], 
                full_name=data["full_name"],
                owner=data["owner"]["login"],
                private=data["private"],
                clone_url=data["clone_url"],
                ssh_url=data["ssh_url"],
                default_branch=data["default_branch"],
                language=data.get("language"),
                description=data.get("description")
            )
            
        except httpx.HTTPStatusError as e:
            self.logger.error("Failed to get repository", {
                "owner": owner,
                "repo": repo,
                "status_code": e.response.status_code,
                "error": str(e)
            })
            raise
    
    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Get pull request information."""
        try:
            response = await self.client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
            response.raise_for_status()
            
            data = response.json()
            return GitHubPullRequest(
                number=data["number"],
                title=data["title"],
                body=data.get("body"),
                state=data["state"],
                head_ref=data["head"]["ref"],
                base_ref=data["base"]["ref"],
                head_sha=data["head"]["sha"],
                base_sha=data["base"]["sha"],
                author=data["user"]["login"],
                created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
                mergeable=data.get("mergeable"),
                draft=data.get("draft", False)
            )
            
        except httpx.HTTPStatusError as e:
            self.logger.error("Failed to get pull request", {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "status_code": e.response.status_code,
                "error": str(e)
            })
            raise
    
    async def create_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str
    ) -> Dict[str, Any]:
        """Create a comment on a pull request."""
        try:
            response = await self.client.post(
                f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
                json={"body": body}
            )
            response.raise_for_status()
            
            self.logger.info("Created PR comment", {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "comment_length": len(body)
            })
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            self.logger.error("Failed to create comment", {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "status_code": e.response.status_code,
                "error": str(e)
            })
            raise
    
    async def get_changed_files(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> List[Dict[str, Any]]:
        """Get list of changed files in a pull request."""
        try:
            response = await self.client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
            response.raise_for_status()
            
            files = response.json()
            self.logger.debug("Retrieved changed files", {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "file_count": len(files)
            })
            
            return files
            
        except httpx.HTTPStatusError as e:
            self.logger.error("Failed to get changed files", {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "status_code": e.response.status_code,
                "error": str(e)
            })
            raise
    
    async def create_check_run(
        self,
        owner: str,
        repo: str,
        name: str,
        head_sha: str,
        status: str = "in_progress",
        conclusion: Optional[str] = None,
        output: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a GitHub check run."""
        try:
            data = {
                "name": name,
                "head_sha": head_sha,
                "status": status
            }
            
            if conclusion:
                data["conclusion"] = conclusion
            
            if output:
                data["output"] = output
            
            response = await self.client.post(
                f"/repos/{owner}/{repo}/check-runs",
                json=data
            )
            response.raise_for_status()
            
            self.logger.info("Created check run", {
                "owner": owner,
                "repo": repo,
                "name": name,
                "head_sha": head_sha[:8],
                "status": status
            })
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            self.logger.error("Failed to create check run", {
                "owner": owner,
                "repo": repo,
                "name": name,
                "status_code": e.response.status_code,
                "error": str(e)
            })
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class GitHubWebhookHandler:
    """Handler for GitHub webhook events."""
    
    def __init__(self, secret: Optional[str] = None):
        self.secret = secret or os.getenv("GITHUB_WEBHOOK_SECRET")
        self.logger = get_logger("testgen_copilot.integrations.github.webhook")
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature."""
        if not self.secret:
            self.logger.warning("No webhook secret configured - skipping signature verification")
            return True
        
        expected_signature = "sha256=" + hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    def parse_event(self, payload: Dict[str, Any], event_type: str) -> Dict[str, Any]:
        """Parse GitHub webhook event payload."""
        self.logger.info("Processing GitHub webhook", {
            "event_type": event_type,
            "repository": payload.get("repository", {}).get("full_name")
        })
        
        if event_type == GitHubEventType.PULL_REQUEST:
            return self._parse_pull_request_event(payload)
        elif event_type == GitHubEventType.PUSH:
            return self._parse_push_event(payload)
        else:
            self.logger.debug("Unhandled event type", {"event_type": event_type})
            return {"event_type": event_type, "data": payload}
    
    def _parse_pull_request_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pull request webhook event."""
        pr_data = payload["pull_request"]
        action = payload["action"]
        
        return {
            "event_type": GitHubEventType.PULL_REQUEST,
            "action": action,
            "repository": {
                "owner": payload["repository"]["owner"]["login"],
                "name": payload["repository"]["name"],
                "full_name": payload["repository"]["full_name"]
            },
            "pull_request": {
                "number": pr_data["number"],
                "title": pr_data["title"],
                "body": pr_data.get("body"),
                "state": pr_data["state"],
                "head_ref": pr_data["head"]["ref"],
                "base_ref": pr_data["base"]["ref"],
                "head_sha": pr_data["head"]["sha"],
                "base_sha": pr_data["base"]["sha"],
                "author": pr_data["user"]["login"],
                "draft": pr_data.get("draft", False)
            }
        }
    
    def _parse_push_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse push webhook event."""
        return {
            "event_type": GitHubEventType.PUSH,
            "repository": {
                "owner": payload["repository"]["owner"]["login"],
                "name": payload["repository"]["name"],
                "full_name": payload["repository"]["full_name"]
            },
            "ref": payload["ref"],
            "before": payload["before"],
            "after": payload["after"], 
            "commits": payload["commits"],
            "pusher": payload["pusher"]["name"]
        }


class GitHubIntegration:
    """Main GitHub integration orchestrator."""
    
    def __init__(
        self,
        github_client: Optional[GitHubClient] = None,
        webhook_handler: Optional[GitHubWebhookHandler] = None
    ):
        self.github = github_client or GitHubClient()
        self.webhook_handler = webhook_handler or GitHubWebhookHandler()
        self.logger = get_logger("testgen_copilot.integrations.github")
        
        # Database repositories
        self.session_repo = SessionRepository(get_database())
        self.analysis_repo = AnalysisRepository(get_database())
    
    async def handle_webhook(
        self,
        payload: bytes,
        signature: str,
        event_type: str
    ) -> Dict[str, Any]:
        """Handle incoming GitHub webhook."""
        # Verify signature
        if not self.webhook_handler.verify_signature(payload, signature):
            raise ValueError("Invalid webhook signature")
        
        # Parse payload
        try:
            payload_dict = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON payload: {e}")
        
        # Parse event
        event = self.webhook_handler.parse_event(payload_dict, event_type)
        
        # Handle specific events
        if event["event_type"] == GitHubEventType.PULL_REQUEST:
            return await self._handle_pull_request_event(event)
        elif event["event_type"] == GitHubEventType.PUSH:
            return await self._handle_push_event(event)
        else:
            self.logger.info("Unhandled webhook event", {
                "event_type": event["event_type"]
            })
            return {"status": "ignored", "event_type": event["event_type"]}
    
    async def _handle_pull_request_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request webhook events."""
        action = event["action"]
        pr = event["pull_request"]
        repo = event["repository"]
        
        self.logger.info("Processing PR event", {
            "action": action,
            "pr_number": pr["number"],
            "repository": repo["full_name"]
        })
        
        if action in [PullRequestAction.OPENED, PullRequestAction.SYNCHRONIZE]:
            # Trigger analysis for changed files
            return await self._analyze_pull_request_changes(repo, pr)
        
        return {"status": "processed", "action": action}
    
    async def _handle_push_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle push webhook events."""
        repo = event["repository"]
        ref = event["ref"]
        commits = event["commits"]
        
        self.logger.info("Processing push event", {
            "repository": repo["full_name"],
            "ref": ref,
            "commit_count": len(commits)
        })
        
        # Only process pushes to main/master branches
        if ref in ["refs/heads/main", "refs/heads/master"]:
            return await self._analyze_push_changes(repo, commits)
        
        return {"status": "ignored", "reason": "non-main branch push"}
    
    async def _analyze_pull_request_changes(
        self,
        repo: Dict[str, Any],
        pr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze changes in a pull request."""
        try:
            # Get changed files
            changed_files = await self.github.get_changed_files(
                repo["owner"],
                repo["name"],
                pr["number"]
            )
            
            # Filter for source code files
            source_files = [
                f for f in changed_files
                if self._is_source_file(f["filename"]) and f["status"] != "removed"
            ]
            
            if not source_files:
                self.logger.info("No source files changed", {
                    "pr_number": pr["number"],
                    "repository": repo["full_name"]
                })
                return {"status": "no_source_changes"}
            
            # Create check run
            check_run = await self.github.create_check_run(
                repo["owner"],
                repo["name"],
                "TestGen Copilot Analysis",
                pr["head_sha"],
                status="in_progress"
            )
            
            self.logger.info("Created check run for PR analysis", {
                "pr_number": pr["number"],
                "check_run_id": check_run["id"],
                "source_files": len(source_files)
            })
            
            # TODO: Implement actual analysis of changed files
            # This would involve:
            # 1. Cloning the repository at the specific commit
            # 2. Running TestGen analysis on changed files
            # 3. Updating the check run with results
            # 4. Creating PR comments with analysis results
            
            return {
                "status": "analysis_queued",
                "check_run_id": check_run["id"],
                "source_files_count": len(source_files)
            }
            
        except Exception as e:
            self.logger.error("Failed to analyze PR changes", {
                "pr_number": pr["number"],
                "repository": repo["full_name"],
                "error": str(e)
            })
            raise
    
    async def _analyze_push_changes(
        self,
        repo: Dict[str, Any], 
        commits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze changes from push commits."""
        # Extract modified files from commits
        modified_files = set()
        for commit in commits:
            modified_files.update(commit.get("modified", []))
            modified_files.update(commit.get("added", []))
        
        # Filter source files
        source_files = [f for f in modified_files if self._is_source_file(f)]
        
        if not source_files:
            return {"status": "no_source_changes"}
        
        self.logger.info("Processing push with source changes", {
            "repository": repo["full_name"],
            "source_files": len(source_files)
        })
        
        # TODO: Implement push analysis
        return {
            "status": "analysis_queued",
            "source_files_count": len(source_files)
        }
    
    def _is_source_file(self, filename: str) -> bool:
        """Check if a file is a source code file we should analyze."""
        source_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cs", ".go", ".rs", ".cpp", ".c", ".h"
        }
        
        path = Path(filename)
        return (
            path.suffix.lower() in source_extensions and
            not any(exclude in str(path).lower() for exclude in [
                "test", "spec", "__pycache__", "node_modules", 
                ".git", "vendor", "dist", "build"
            ])
        )
    
    async def create_analysis_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a comment on PR with analysis results."""
        # Generate markdown comment from analysis results
        comment = self._generate_analysis_comment(analysis_results)
        
        return await self.github.create_comment(owner, repo, pr_number, comment)
    
    def _generate_analysis_comment(self, results: Dict[str, Any]) -> str:
        """Generate markdown comment from analysis results."""
        lines = [
            "## 🤖 TestGen Copilot Analysis Results",
            "",
            f"**Analysis Status:** {results.get('status', 'Unknown')}",
            f"**Files Analyzed:** {results.get('files_analyzed', 0)}",
            f"**Tests Generated:** {results.get('tests_generated', 0)}",
            ""
        ]
        
        # Add security issues if any
        security_issues = results.get('security_issues', 0)
        if security_issues > 0:
            lines.extend([
                f"⚠️ **Security Issues Found:** {security_issues}",
                ""
            ])
        
        # Add coverage information
        coverage = results.get('coverage_percentage')
        if coverage is not None:
            emoji = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
            lines.extend([
                f"{emoji} **Test Coverage:** {coverage:.1f}%",
                ""
            ])
        
        # Add quality score
        quality = results.get('quality_score')
        if quality is not None:
            emoji = "✅" if quality >= 75 else "⚠️" if quality >= 50 else "❌"
            lines.extend([
                f"{emoji} **Test Quality Score:** {quality:.1f}%",
                ""
            ])
        
        lines.extend([
            "---",
            "*Generated by [TestGen Copilot](https://github.com/testgen/copilot-assistant)*"
        ])
        
        return "\n".join(lines)
    
    async def close(self):
        """Close the integration and cleanup resources."""
        await self.github.close()