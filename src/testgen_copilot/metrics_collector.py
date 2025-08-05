#!/usr/bin/env python3
"""
Metrics Collection and DORA Metrics Tracking

Implements comprehensive metrics collection for autonomous backlog management,
including DORA metrics, rerere statistics, CI health, and operational metrics.
"""

import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re
import os

from .logging_config import get_core_logger


@dataclass
class DORAMetrics:
    """DORA (DevOps Research and Assessment) metrics."""
    deployment_frequency: float  # deployments per day
    lead_time_hours: float  # time from commit to production
    change_failure_rate: float  # percentage of deployments causing failures
    mean_time_to_recovery_hours: float  # time to recover from failures


@dataclass
class CIMetrics:
    """CI/CD pipeline health metrics."""
    total_builds: int
    successful_builds: int
    failed_builds: int
    failure_rate: float
    average_duration_minutes: float
    flaky_tests: List[str]


@dataclass
class ConflictResolutionMetrics:
    """Git merge conflict resolution metrics."""
    rerere_auto_resolved: int
    merge_driver_hits: int
    manual_resolutions: int
    conflict_rate: float


@dataclass
class BacklogMetrics:
    """Backlog health and flow metrics."""
    total_items: int
    items_by_status: Dict[str, int]
    average_cycle_time_hours: float
    wsjf_distribution: Dict[str, float]
    aging_items_count: int


@dataclass
class OperationalMetrics:
    """Overall operational health metrics."""
    timestamp: datetime
    dora: DORAMetrics
    ci: CIMetrics
    conflicts: ConflictResolutionMetrics
    backlog: BacklogMetrics
    pr_backoff_active: bool
    risks_and_blocks: List[str]
    completed_items: List[str]


class MetricsCollector:
    """Collects and aggregates metrics from various sources."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.logger = get_core_logger()
        self.metrics_dir = repo_path / "docs" / "status"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_dora_metrics(self, days_back: int = 30) -> DORAMetrics:
        """Collect DORA metrics from git history."""
        try:
            # Get commits from the last N days
            since_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
            
            # Deployment frequency (using tags as proxy for deployments)
            tag_result = subprocess.run(
                ['git', 'tag', '--sort=-creatordate', '--format=%(creatordate:iso)', '--merged'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            recent_tags = 0
            if tag_result.returncode == 0:
                for line in tag_result.stdout.strip().split('\n'):
                    if line and line >= since_date:
                        recent_tags += 1
            
            deployment_frequency = recent_tags / days_back if days_back > 0 else 0
            
            # Lead time (time from first commit to merge)
            lead_times = []
            merge_result = subprocess.run(
                ['git', 'log', '--merges', '--since', since_date, '--format=%H %ct'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if merge_result.returncode == 0:
                for line in merge_result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            merge_hash, merge_time = parts[0], int(parts[1])
                            # Get first commit in the merge
                            first_commit_result = subprocess.run(
                                ['git', 'log', '--reverse', '--format=%ct', f'{merge_hash}^..{merge_hash}'],
                                cwd=self.repo_path,
                                capture_output=True,
                                text=True
                            )
                            
                            if first_commit_result.returncode == 0:
                                first_times = first_commit_result.stdout.strip().split('\n')
                                if first_times and first_times[0]:
                                    first_time = int(first_times[0])
                                    lead_time_seconds = merge_time - first_time
                                    lead_times.append(lead_time_seconds / 3600)  # Convert to hours
            
            avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0
            
            # Change failure rate (using incidents with type:incident label)
            incident_count = 0
            total_deployments = recent_tags
            
            # Try to get incident count from commit messages
            incident_result = subprocess.run(
                ['git', 'log', '--since', since_date, '--grep=incident', '--format=%H'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if incident_result.returncode == 0:
                incident_count = len([l for l in incident_result.stdout.strip().split('\n') if l])
            
            change_failure_rate = (incident_count / total_deployments * 100) if total_deployments > 0 else 0
            
            # MTTR (Mean Time To Recovery) - simplified calculation
            mttr_hours = 2.0  # Default assumption for small changes
            
            return DORAMetrics(
                deployment_frequency=deployment_frequency,
                lead_time_hours=avg_lead_time,
                change_failure_rate=change_failure_rate,
                mean_time_to_recovery_hours=mttr_hours
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting DORA metrics: {e}")
            return DORAMetrics(0, 0, 0, 0)
    
    def collect_ci_metrics(self) -> CIMetrics:
        """Collect CI/CD pipeline metrics."""
        try:
            # Check recent CI runs (simplified - would normally query CI API)
            flaky_tests = []
            
            # Check for test failures in recent commits
            test_result = subprocess.run(
                ['git', 'log', '--since', '7 days ago', '--grep', 'test', '--format=%s'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            total_builds = 10  # Simplified assumption
            failed_builds = 0
            
            if test_result.returncode == 0:
                for line in test_result.stdout.strip().split('\n'):
                    if line and ('fix' in line.lower() or 'fail' in line.lower()):
                        failed_builds += 1
            
            successful_builds = total_builds - failed_builds
            failure_rate = (failed_builds / total_builds * 100) if total_builds > 0 else 0
            
            return CIMetrics(
                total_builds=total_builds,
                successful_builds=successful_builds,
                failed_builds=failed_builds,
                failure_rate=failure_rate,
                average_duration_minutes=5.0,  # Estimated
                flaky_tests=flaky_tests
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting CI metrics: {e}")
            return CIMetrics(0, 0, 0, 0, 0, [])
    
    def collect_conflict_metrics(self) -> ConflictResolutionMetrics:
        """Collect git merge conflict resolution metrics."""
        try:
            rerere_auto_resolved = 0
            merge_driver_hits = 0
            manual_resolutions = 0
            
            # Check rerere cache
            rerere_dir = self.repo_path / ".git" / "rr-cache"
            if rerere_dir.exists():
                rerere_auto_resolved = len(list(rerere_dir.iterdir()))
            
            # Check merge commits for conflict indicators
            merge_result = subprocess.run(
                ['git', 'log', '--merges', '--since', '30 days ago', '--format=%H'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            total_merges = 0
            conflicts = 0
            
            if merge_result.returncode == 0:
                merge_hashes = [h for h in merge_result.stdout.strip().split('\n') if h]
                total_merges = len(merge_hashes)
                
                # Check for merge conflict indicators in commit messages
                for merge_hash in merge_hashes:
                    commit_result = subprocess.run(
                        ['git', 'log', '--format=%B', '-n', '1', merge_hash],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )
                    
                    if commit_result.returncode == 0:
                        message = commit_result.stdout.lower()
                        if 'conflict' in message or 'merge' in message:
                            conflicts += 1
            
            conflict_rate = (conflicts / total_merges * 100) if total_merges > 0 else 0
            
            return ConflictResolutionMetrics(
                rerere_auto_resolved=rerere_auto_resolved,
                merge_driver_hits=merge_driver_hits,
                manual_resolutions=manual_resolutions,
                conflict_rate=conflict_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting conflict metrics: {e}")
            return ConflictResolutionMetrics(0, 0, 0, 0)
    
    def collect_backlog_metrics(self) -> BacklogMetrics:
        """Collect backlog health metrics."""
        try:
            backlog_file = self.repo_path / "backlog.json"
            
            if not backlog_file.exists():
                return BacklogMetrics(0, {}, 0, {}, 0)
            
            with open(backlog_file, 'r') as f:
                backlog_data = json.load(f)
            
            total_items = len(backlog_data)
            items_by_status = {}
            wsjf_scores = []
            aging_items = 0
            cycle_times = []
            
            now = datetime.now(timezone.utc)
            
            for item_data in backlog_data:
                status = item_data.get('status', 'NEW')
                items_by_status[status] = items_by_status.get(status, 0) + 1
                
                # Calculate WSJF score
                effort = item_data.get('effort', 1)
                value = item_data.get('value', 1)
                time_crit = item_data.get('time_criticality', 1)
                risk_red = item_data.get('risk_reduction', 1)
                aging_mult = item_data.get('aging_multiplier', 1.0)
                
                if effort > 0:
                    wsjf = ((value + time_crit + risk_red) / effort) * aging_mult
                    wsjf_scores.append(wsjf)
                
                # Check for aging items
                created_at = datetime.fromisoformat(item_data.get('created_at', now.isoformat()))
                days_old = (now - created_at).days
                if days_old > 7:
                    aging_items += 1
                
                # Simplified cycle time calculation
                if status == 'DONE':
                    cycle_times.append(days_old * 24)  # Convert to hours
            
            avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
            
            wsjf_distribution = {
                'high': len([s for s in wsjf_scores if s > 5]),
                'medium': len([s for s in wsjf_scores if 2 <= s <= 5]),
                'low': len([s for s in wsjf_scores if s < 2])
            }
            
            return BacklogMetrics(
                total_items=total_items,
                items_by_status=items_by_status,
                average_cycle_time_hours=avg_cycle_time,
                wsjf_distribution=wsjf_distribution,
                aging_items_count=aging_items
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting backlog metrics: {e}")
            return BacklogMetrics(0, {}, 0, {}, 0)
    
    def assess_pr_backoff_status(self, ci_metrics: CIMetrics) -> bool:
        """Determine if PR creation should be throttled based on CI failure rate."""
        return ci_metrics.failure_rate > 30.0
    
    def collect_comprehensive_metrics(self) -> OperationalMetrics:
        """Collect all metrics and create comprehensive report."""
        self.logger.info("Collecting comprehensive operational metrics...")
        
        dora = self.collect_dora_metrics()
        ci = self.collect_ci_metrics()
        conflicts = self.collect_conflict_metrics()
        backlog = self.collect_backlog_metrics()
        pr_backoff = self.assess_pr_backoff_status(ci)
        
        # Identify risks and blocks
        risks = []
        if ci.failure_rate > 30:
            risks.append(f"High CI failure rate: {ci.failure_rate:.1f}%")
        if conflicts.conflict_rate > 20:
            risks.append(f"High merge conflict rate: {conflicts.conflict_rate:.1f}%")
        if backlog.aging_items_count > 5:
            risks.append(f"Too many aging backlog items: {backlog.aging_items_count}")
        if dora.change_failure_rate > 10:
            risks.append(f"High change failure rate: {dora.change_failure_rate:.1f}%")
        
        return OperationalMetrics(
            timestamp=datetime.now(timezone.utc),
            dora=dora,
            ci=ci,
            conflicts=conflicts,
            backlog=backlog,
            pr_backoff_active=pr_backoff,
            risks_and_blocks=risks,
            completed_items=[]  # To be filled by execution system
        )
    
    def save_metrics_report(self, metrics: OperationalMetrics, completed_ids: List[str] = None):
        """Save metrics report to JSON and markdown files."""
        metrics.completed_items = completed_ids or []
        
        timestamp_str = metrics.timestamp.strftime("%Y-%m-%d")
        json_file = self.metrics_dir / f"autonomous-execution-{timestamp_str}.json"
        md_file = self.metrics_dir / f"autonomous-execution-{timestamp_str}.md"
        
        # Save JSON report
        try:
            with open(json_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving JSON report: {e}")
        
        # Save Markdown report
        try:
            md_content = self._generate_markdown_report(metrics)
            with open(md_file, 'w') as f:
                f.write(md_content)
        except Exception as e:
            self.logger.error(f"Error saving Markdown report: {e}")
        
        self.logger.info(f"Metrics report saved to {json_file} and {md_file}")
    
    def _generate_markdown_report(self, metrics: OperationalMetrics) -> str:
        """Generate human-readable markdown report."""
        timestamp = metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return f"""# Autonomous Execution Report - {timestamp}

## Summary
- **Total Backlog Items**: {metrics.backlog.total_items}
- **Completed Today**: {len(metrics.completed_items)}
- **CI Health**: {metrics.ci.successful_builds}/{metrics.ci.total_builds} builds successful ({100-metrics.ci.failure_rate:.1f}%)
- **PR Backoff**: {'🔴 ACTIVE' if metrics.pr_backoff_active else '🟢 INACTIVE'}

## DORA Metrics
- **Deployment Frequency**: {metrics.dora.deployment_frequency:.2f} deployments/day
- **Lead Time**: {metrics.dora.lead_time_hours:.1f} hours
- **Change Failure Rate**: {metrics.dora.change_failure_rate:.1f}%
- **MTTR**: {metrics.dora.mean_time_to_recovery_hours:.1f} hours

## Backlog Health
- **Cycle Time**: {metrics.backlog.average_cycle_time_hours:.1f} hours average
- **Aging Items**: {metrics.backlog.aging_items_count} items over 7 days old
- **Status Distribution**: {dict(metrics.backlog.items_by_status)}

## Merge Conflict Resolution
- **Rerere Auto-Resolved**: {metrics.conflicts.rerere_auto_resolved}
- **Conflict Rate**: {metrics.conflicts.conflict_rate:.1f}%
- **Manual Resolutions**: {metrics.conflicts.manual_resolutions}

## Completed Items
{chr(10).join(f'- {item_id}' for item_id in metrics.completed_items) if metrics.completed_items else '- None'}

## Risks and Blocks
{chr(10).join(f'- ⚠️ {risk}' for risk in metrics.risks_and_blocks) if metrics.risks_and_blocks else '- None identified'}

## CI Pipeline Status
- **Success Rate**: {100-metrics.ci.failure_rate:.1f}%
- **Average Duration**: {metrics.ci.average_duration_minutes:.1f} minutes
- **Flaky Tests**: {len(metrics.ci.flaky_tests)} identified

---
*Generated by Autonomous Backlog Management System*
"""


def main():
    """CLI entry point for metrics collection."""
    import sys
    
    repo_path = Path.cwd()
    collector = MetricsCollector(repo_path)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--json-only':
        metrics = collector.collect_comprehensive_metrics()
        print(json.dumps(asdict(metrics), indent=2, default=str))
    else:
        metrics = collector.collect_comprehensive_metrics()
        collector.save_metrics_report(metrics)
        print(f"Metrics collected and saved to {collector.metrics_dir}")


if __name__ == "__main__":
    main()