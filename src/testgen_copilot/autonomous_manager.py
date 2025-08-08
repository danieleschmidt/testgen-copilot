#!/usr/bin/env python3
"""
Autonomous Backlog Management - Main Execution Loop

Implements the complete autonomous backlog management system with WSJF prioritization,
TDD execution, security checks, and continuous delivery.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .autonomous_backlog import BacklogItem, BacklogManager, TaskStatus
from .autonomous_execution import ExecutionResult, TDDExecutor
from .logging_config import get_core_logger
from .metrics_collector import MetricsCollector


class AutonomousManager:
    """Main autonomous backlog management orchestrator."""

    def __init__(self, repo_path: Path, config_path: Path, dry_run: bool = False):
        self.repo_path = repo_path
        self.config_path = config_path
        self.dry_run = dry_run
        self.logger = get_core_logger()

        # Initialize components
        self.backlog_manager = BacklogManager(repo_path, config_path)
        self.executor = TDDExecutor(repo_path)
        self.metrics_collector = MetricsCollector(repo_path)

        # Execution state
        self.completed_items: List[str] = []
        self.current_iteration = 0
        self.max_iterations = 10  # Safety limit
        self.max_prs_per_day = 5
        self.current_pr_count = 0

    def sync_repo_and_ci(self) -> bool:
        """Sync repository state and check CI health."""
        try:
            self.logger.info("Syncing repository and checking CI state...")

            # Fetch latest changes
            result = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Git fetch failed: {result.stderr}")
                return False

            # Check if we're on main branch
            branch_result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            current_branch = branch_result.stdout.strip()
            if current_branch != 'main':
                self.logger.info(f"Currently on branch: {current_branch}")

                # Switch to main if needed
                if not self.dry_run:
                    checkout_result = subprocess.run(
                        ['git', 'checkout', 'main'],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )

                    if checkout_result.returncode != 0:
                        self.logger.error(f"Could not checkout main: {checkout_result.stderr}")
                        return False

            # Pull latest changes
            if not self.dry_run:
                pull_result = subprocess.run(
                    ['git', 'pull', 'origin', 'main'],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )

                if pull_result.returncode != 0:
                    self.logger.error(f"Git pull failed: {pull_result.stderr}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Repository sync failed: {e}")
            return False

    def create_feature_branch(self, task: BacklogItem) -> str:
        """Create a feature branch for the task."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_title = "".join(c for c in task.title.lower() if c.isalnum() or c in '-_')[:30]
        branch_name = f"autonomous/{safe_title}-{timestamp}"

        if not self.dry_run:
            try:
                result = subprocess.run(
                    ['git', 'checkout', '-b', branch_name],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    self.logger.error(f"Branch creation failed: {result.stderr}")
                    return ""

            except Exception as e:
                self.logger.error(f"Error creating branch: {e}")
                return ""

        self.logger.info(f"Created feature branch: {branch_name}")
        return branch_name

    def commit_and_push_changes(self, task: BacklogItem, branch_name: str, execution_result: ExecutionResult) -> bool:
        """Commit changes and push to remote."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would commit and push changes for {task.title}")
            return True

        try:
            # Stage all changes
            add_result = subprocess.run(
                ['git', 'add', '.'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if add_result.returncode != 0:
                self.logger.error(f"Git add failed: {add_result.stderr}")
                return False

            # Check if there are changes to commit
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if not status_result.stdout.strip():
                self.logger.info("No changes to commit")
                return True

            # Create commit message
            commit_message = f"""
{task.title}

{task.description}

Acceptance Criteria:
{chr(10).join(f'- {criteria}' for criteria in task.acceptance_criteria)}

Execution Summary:
{chr(10).join(f'- {change}' for change in execution_result.changes_made)}

Security Status: {'✅ Passed' if execution_result.security_results.passed else '❌ Failed'}
Test Status: {execution_result.test_results.get('status', 'Unknown')}

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
""".strip()

            # Commit changes
            commit_result = subprocess.run(
                ['git', 'commit', '-m', commit_message],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if commit_result.returncode != 0:
                self.logger.error(f"Git commit failed: {commit_result.stderr}")
                return False

            # Push to remote
            push_result = subprocess.run(
                ['git', 'push', '-u', 'origin', branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if push_result.returncode != 0:
                self.logger.error(f"Git push failed: {push_result.stderr}")
                return False

            self.logger.info(f"Successfully committed and pushed changes for {task.title}")
            return True

        except Exception as e:
            self.logger.error(f"Error committing changes: {e}")
            return False

    def create_pull_request(self, task: BacklogItem, branch_name: str, execution_result: ExecutionResult) -> bool:
        """Create a pull request for the completed task."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create PR for {task.title}")
            return True

        try:
            # Create PR body
            pr_body = f"""
## Summary
{task.description}

## Acceptance Criteria
{chr(10).join(f'- [x] {criteria}' for criteria in task.acceptance_criteria)}

## Changes Made
{chr(10).join(f'- {change}' for change in execution_result.changes_made)}

## Security Analysis
- **Vulnerabilities**: {len(execution_result.security_results.vulnerabilities)}
- **Warnings**: {len(execution_result.security_results.warnings)}
- **Status**: {'✅ Passed' if execution_result.security_results.passed else '❌ Failed'}

## Test Results
- **Status**: {execution_result.test_results.get('status', 'Unknown')}
- **Coverage**: {execution_result.test_results.get('coverage', 'N/A')}

## Risk Assessment
- **Risk Tier**: {task.risk_tier.value}
- **WSJF Score**: {task.wsjf_score:.2f}

## Rollback Plan
If issues arise, revert this PR and return task to backlog with updated priority.

🤖 Generated with [Claude Code](https://claude.ai/code)
""".strip()

            # Create PR using GitHub CLI
            pr_result = subprocess.run([
                'gh', 'pr', 'create',
                '--title', task.title,
                '--body', pr_body,
                '--base', 'main',
                '--head', branch_name
            ], cwd=self.repo_path, capture_output=True, text=True)

            if pr_result.returncode != 0:
                self.logger.error(f"PR creation failed: {pr_result.stderr}")
                return False

            pr_url = pr_result.stdout.strip()
            self.logger.info(f"Created PR: {pr_url}")
            self.current_pr_count += 1

            # Update task status
            self.backlog_manager.update_item_status(task.id, TaskStatus.PR)

            return True

        except Exception as e:
            self.logger.error(f"Error creating PR: {e}")
            return False

    def execute_micro_cycle(self, task: BacklogItem) -> bool:
        """Execute the complete micro-cycle for a single task."""
        self.logger.info(f"Starting micro-cycle for task: {task.title}")

        # Update task status to DOING
        self.backlog_manager.update_item_status(task.id, TaskStatus.DOING)

        # Create feature branch
        branch_name = self.create_feature_branch(task)
        if not branch_name and not self.dry_run:
            self.logger.error(f"Failed to create branch for task: {task.title}")
            return False

        try:
            # Execute TDD cycle
            execution_result = self.executor.execute_task(task, branch_name)

            if not execution_result.success:
                self.logger.error(f"Task execution failed: {execution_result.error_message}")
                # Return to main branch
                if not self.dry_run:
                    subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path)
                return False

            # Commit and push changes
            if not self.commit_and_push_changes(task, branch_name, execution_result):
                self.logger.error(f"Failed to commit changes for task: {task.title}")
                return False

            # Create pull request
            if not self.create_pull_request(task, branch_name, execution_result):
                self.logger.error(f"Failed to create PR for task: {task.title}")
                return False

            # Mark task as completed
            self.backlog_manager.update_item_status(task.id, TaskStatus.DONE)
            self.completed_items.append(task.id)

            # Return to main branch
            if not self.dry_run:
                subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path)

            self.logger.info(f"Successfully completed micro-cycle for task: {task.title}")
            return True

        except Exception as e:
            self.logger.error(f"Micro-cycle execution failed: {e}")
            # Return to main branch
            if not self.dry_run:
                subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path)
            return False

    def should_continue_execution(self) -> bool:
        """Determine if execution should continue based on limits and conditions."""
        # Check iteration limit
        if self.current_iteration >= self.max_iterations:
            self.logger.info(f"Reached maximum iterations: {self.max_iterations}")
            return False

        # Check PR limit
        if self.current_pr_count >= self.max_prs_per_day:
            self.logger.info(f"Reached daily PR limit: {self.max_prs_per_day}")
            return False

        # Check CI health
        ci_metrics = self.metrics_collector.collect_ci_metrics()
        if ci_metrics.failure_rate > 30:
            self.logger.warning(f"CI failure rate too high: {ci_metrics.failure_rate:.1f}%")
            return False

        return True

    async def run_autonomous_cycle(self) -> Dict[str, Any]:
        """Run the complete autonomous backlog management cycle."""
        self.logger.info("🤖 Starting Autonomous Backlog Management Cycle")

        start_time = time.time()
        execution_summary = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "completed_tasks": [],
            "failed_tasks": [],
            "total_iterations": 0,
            "execution_time_seconds": 0,
            "final_metrics": {}
        }

        try:
            # Initial setup
            if not self.sync_repo_and_ci():
                raise Exception("Repository sync failed")

            # Load and discover backlog
            self.backlog_manager.load_backlog()
            self.backlog_manager.discover_and_sync()
            self.backlog_manager.save_backlog()

            # Main execution loop
            while self.should_continue_execution():
                self.current_iteration += 1
                self.logger.info(f"🔄 Starting iteration {self.current_iteration}")

                # Sync repository state
                if not self.sync_repo_and_ci():
                    self.logger.error("Repository sync failed, stopping execution")
                    break

                # Discover and update backlog
                self.backlog_manager.discover_and_sync()

                # Get next ready task
                next_task = self.backlog_manager.get_next_ready_task()
                if not next_task:
                    # Try to promote a NEW task to READY
                    actionable_items = self.backlog_manager.get_actionable_items()
                    new_tasks = [item for item in actionable_items if item.status == TaskStatus.NEW]

                    if new_tasks:
                        # Promote highest priority NEW task to READY
                        next_task = new_tasks[0]
                        self.backlog_manager.update_item_status(next_task.id, TaskStatus.READY)
                        self.logger.info(f"Promoted task to READY: {next_task.title}")
                    else:
                        self.logger.info("No actionable tasks available, stopping execution")
                        break

                # Execute micro-cycle
                if self.execute_micro_cycle(next_task):
                    execution_summary["completed_tasks"].append({
                        "id": next_task.id,
                        "title": next_task.title,
                        "wsjf_score": next_task.wsjf_score
                    })
                else:
                    execution_summary["failed_tasks"].append({
                        "id": next_task.id,
                        "title": next_task.title,
                        "error": "Execution failed"
                    })

                # Save updated backlog
                self.backlog_manager.save_backlog()

                # Brief pause between iterations
                await asyncio.sleep(1)

            # Final metrics collection
            execution_summary["total_iterations"] = self.current_iteration
            execution_summary["execution_time_seconds"] = time.time() - start_time
            execution_summary["end_time"] = datetime.now(timezone.utc).isoformat()

            # Collect and save metrics
            final_metrics = self.metrics_collector.collect_comprehensive_metrics()
            self.metrics_collector.save_metrics_report(final_metrics, self.completed_items)
            execution_summary["final_metrics"] = final_metrics.__dict__

            self.logger.info(f"✅ Autonomous cycle completed. Processed {len(execution_summary['completed_tasks'])} tasks.")

            return execution_summary

        except Exception as e:
            self.logger.error(f"Autonomous cycle failed: {e}")
            execution_summary["error"] = str(e)
            execution_summary["execution_time_seconds"] = time.time() - start_time
            return execution_summary


def main():
    """CLI entry point for autonomous backlog management."""
    parser = argparse.ArgumentParser(description="Autonomous Backlog Management System")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run in dry-run mode without making changes")
    parser.add_argument("--config", type=Path,
                       default=Path.cwd() / ".automation-scope.yaml",
                       help="Path to automation configuration file")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum number of iterations to run")
    parser.add_argument("--max-prs", type=int, default=5,
                       help="Maximum PRs to create per day")

    args = parser.parse_args()

    repo_path = Path.cwd()

    # Initialize manager
    manager = AutonomousManager(repo_path, args.config, args.dry_run)
    manager.max_iterations = args.max_iterations
    manager.max_prs_per_day = args.max_prs

    # Run autonomous cycle
    try:
        result = asyncio.run(manager.run_autonomous_cycle())
        print(json.dumps(result, indent=2, default=str))

        # Exit with appropriate code
        if result.get("error"):
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n🛑 Autonomous execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
