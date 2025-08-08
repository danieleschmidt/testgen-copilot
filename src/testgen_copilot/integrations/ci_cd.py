"""CI/CD integration services for TestGen Copilot."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from ..logging_config import get_logger


class CIPlatform(str, Enum):
    """CI/CD platform types."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"


@dataclass
class CIJobResult:
    """CI job execution result."""
    job_id: str
    status: str
    success: bool
    duration_seconds: float
    logs: Optional[str] = None
    artifacts: List[str] = None


class BaseCIClient(ABC):
    """Base class for CI/CD integrations."""

    def __init__(self, platform: CIPlatform):
        self.platform = platform
        self.logger = get_logger(f"testgen_copilot.integrations.ci_cd.{platform.value}")

    @abstractmethod
    async def trigger_job(self, repo: str, branch: str, parameters: Dict[str, Any]) -> str:
        """Trigger a CI job."""
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> CIJobResult:
        """Get job status and results."""
        pass


class GitHubActionsClient(BaseCIClient):
    """GitHub Actions integration client."""

    def __init__(self, token: Optional[str] = None):
        super().__init__(CIPlatform.GITHUB_ACTIONS)
        self.token = token or os.getenv("GITHUB_TOKEN")

        if not self.token:
            self.logger.warning("No GitHub token provided")

        self.client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"Bearer {self.token}" if self.token else ""
            }
        )

    async def trigger_job(self, repo: str, branch: str, parameters: Dict[str, Any]) -> str:
        """Trigger GitHub Actions workflow."""
        workflow_id = parameters.get("workflow_id", "testgen-analysis.yml")

        payload = {
            "ref": branch,
            "inputs": {k: str(v) for k, v in parameters.items() if k != "workflow_id"}
        }

        response = await self.client.post(
            f"/repos/{repo}/actions/workflows/{workflow_id}/dispatches",
            json=payload
        )
        response.raise_for_status()

        self.logger.info("Triggered GitHub Actions workflow", {
            "repo": repo,
            "workflow": workflow_id,
            "branch": branch
        })

        return f"{repo}/{workflow_id}/{branch}"

    async def get_job_status(self, job_id: str) -> CIJobResult:
        """Get GitHub Actions job status."""
        # Parse job_id format: repo/workflow/branch
        parts = job_id.split("/")
        if len(parts) < 3:
            raise ValueError("Invalid job ID format")

        repo = "/".join(parts[:-2])
        workflow = parts[-2]
        branch = parts[-1]

        # Get workflow runs
        response = await self.client.get(
            f"/repos/{repo}/actions/workflows/{workflow}/runs",
            params={"branch": branch, "per_page": 1}
        )
        response.raise_for_status()

        runs = response.json()["workflow_runs"]
        if not runs:
            raise ValueError("No workflow runs found")

        run = runs[0]

        return CIJobResult(
            job_id=job_id,
            status=run["status"],
            success=run["conclusion"] == "success",
            duration_seconds=(
                # Calculate duration if completed
                0 if run["status"] == "in_progress" else 60  # Placeholder
            )
        )


class GitLabCIClient(BaseCIClient):
    """GitLab CI integration client."""

    def __init__(self, token: Optional[str] = None, base_url: str = "https://gitlab.com/api/v4"):
        super().__init__(CIPlatform.GITLAB_CI)
        self.token = token or os.getenv("GITLAB_TOKEN")
        self.base_url = base_url

        if not self.token:
            self.logger.warning("No GitLab token provided")

        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Private-Token": self.token if self.token else ""
            }
        )

    async def trigger_job(self, repo: str, branch: str, parameters: Dict[str, Any]) -> str:
        """Trigger GitLab CI pipeline."""
        # Get project ID
        project_response = await self.client.get(f"/projects/{repo.replace('/', '%2F')}")
        project_response.raise_for_status()
        project_id = project_response.json()["id"]

        # Trigger pipeline
        payload = {
            "ref": branch,
            "variables": [
                {"key": k, "value": str(v)} for k, v in parameters.items()
            ]
        }

        response = await self.client.post(
            f"/projects/{project_id}/pipeline",
            json=payload
        )
        response.raise_for_status()

        pipeline_id = response.json()["id"]

        self.logger.info("Triggered GitLab CI pipeline", {
            "project_id": project_id,
            "pipeline_id": pipeline_id,
            "branch": branch
        })

        return f"{project_id}/{pipeline_id}"

    async def get_job_status(self, job_id: str) -> CIJobResult:
        """Get GitLab CI pipeline status."""
        project_id, pipeline_id = job_id.split("/")

        response = await self.client.get(f"/projects/{project_id}/pipelines/{pipeline_id}")
        response.raise_for_status()

        pipeline = response.json()

        return CIJobResult(
            job_id=job_id,
            status=pipeline["status"],
            success=pipeline["status"] == "success",
            duration_seconds=pipeline.get("duration", 0) or 0
        )


class CICDIntegration:
    """Main CI/CD integration service."""

    def __init__(self):
        self.logger = get_logger("testgen_copilot.integrations.ci_cd")
        self.clients: Dict[CIPlatform, BaseCIClient] = {}

        # Initialize available clients
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize CI/CD clients."""
        # GitHub Actions
        if os.getenv("GITHUB_TOKEN"):
            self.clients[CIPlatform.GITHUB_ACTIONS] = GitHubActionsClient()
            self.logger.info("GitHub Actions client initialized")

        # GitLab CI
        if os.getenv("GITLAB_TOKEN"):
            gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com/api/v4")
            self.clients[CIPlatform.GITLAB_CI] = GitLabCIClient(base_url=gitlab_url)
            self.logger.info("GitLab CI client initialized")

    def get_client(self, platform: CIPlatform) -> Optional[BaseCIClient]:
        """Get CI/CD client for platform."""
        return self.clients.get(platform)

    async def trigger_testgen_job(
        self,
        platform: CIPlatform,
        repo: str,
        branch: str = "main",
        file_paths: Optional[List[str]] = None,
        coverage_target: float = 85.0,
        enable_security: bool = True
    ) -> Optional[str]:
        """Trigger TestGen analysis job on CI/CD platform."""
        client = self.get_client(platform)
        if not client:
            self.logger.error("CI/CD client not available", {"platform": platform.value})
            return None

        parameters = {
            "coverage_target": coverage_target,
            "enable_security": enable_security,
            "workflow_id": "testgen-analysis.yml"
        }

        if file_paths:
            parameters["file_paths"] = ",".join(file_paths)

        try:
            job_id = await client.trigger_job(repo, branch, parameters)

            self.logger.info("Triggered TestGen CI job", {
                "platform": platform.value,
                "repo": repo,
                "branch": branch,
                "job_id": job_id
            })

            return job_id

        except Exception as e:
            self.logger.error("Failed to trigger CI job", {
                "platform": platform.value,
                "repo": repo,
                "error": str(e)
            })
            return None

    async def get_job_result(self, platform: CIPlatform, job_id: str) -> Optional[CIJobResult]:
        """Get job result from CI/CD platform."""
        client = self.get_client(platform)
        if not client:
            return None

        try:
            return await client.get_job_status(job_id)
        except Exception as e:
            self.logger.error("Failed to get job status", {
                "platform": platform.value,
                "job_id": job_id,
                "error": str(e)
            })
            return None

    def is_platform_available(self, platform: CIPlatform) -> bool:
        """Check if CI/CD platform is available."""
        return platform in self.clients
