#!/usr/bin/env python3
"""
Autonomous Backlog Management System

Implements WSJF-based backlog discovery, prioritization, and execution for continuous delivery.
Follows trunk-based development with TDD and security-first principles.
"""

import json
import yaml
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import subprocess
import logging

from .logging_config import get_core_logger


class TaskType(Enum):
    """Task types for categorization."""
    FEATURE = "feature"
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    TECH_DEBT = "tech_debt"


class TaskStatus(Enum):
    """Task workflow states."""
    NEW = "NEW"
    REFINED = "REFINED"
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class RiskTier(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BacklogItem:
    """Represents a single backlog item with WSJF scoring."""
    id: str
    title: str
    type: TaskType
    description: str
    acceptance_criteria: List[str]
    effort: int  # 1-2-3-5-8-13 scale
    value: int  # 1-2-3-5-8-13 scale
    time_criticality: int  # 1-2-3-5-8-13 scale
    risk_reduction: int  # 1-2-3-5-8-13 scale
    status: TaskStatus
    risk_tier: RiskTier
    created_at: datetime
    links: List[str]
    aging_multiplier: float = 1.0
    
    @property
    def wsjf_score(self) -> float:
        """Calculate Weighted Shortest Job First score."""
        if self.effort == 0:
            return float('inf')
        return ((self.value + self.time_criticality + self.risk_reduction) / self.effort) * self.aging_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['type'] = self.type.value
        result['status'] = self.status.value
        result['risk_tier'] = self.risk_tier.value
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacklogItem':
        """Create from dictionary."""
        data['type'] = TaskType(data['type'])
        data['status'] = TaskStatus(data['status'])
        data['risk_tier'] = RiskTier(data['risk_tier'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class BacklogDiscovery:
    """Discovers tasks from various sources and converts them to backlog items."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.logger = get_core_logger()
    
    def discover_from_backlog_md(self, file_path: Path) -> List[BacklogItem]:
        """Parse existing BACKLOG.md file for items."""
        items = []
        
        if not file_path.exists():
            return items
        
        try:
            content = file_path.read_text(encoding='utf-8')
            # Parse markdown table format
            lines = content.split('\n')
            for line in lines:
                if '|' in line and not line.strip().startswith('|---'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 8:
                        task_name = parts[1]
                        if task_name and not task_name.startswith('Task') and '~~' not in task_name:
                            # Extract scores
                            try:
                                value = int(parts[2])
                                time_crit = int(parts[3])
                                risk_red = int(parts[4])
                                effort = int(parts[5])
                                status = TaskStatus.NEW if 'DONE' not in parts[8] else TaskStatus.DONE
                                
                                item = BacklogItem(
                                    id=f"backlog-{len(items)+1}",
                                    title=task_name,
                                    type=TaskType.FEATURE,
                                    description=f"Backlog item: {task_name}",
                                    acceptance_criteria=["Implementation completed", "Tests passing"],
                                    effort=effort,
                                    value=value,
                                    time_criticality=time_crit,
                                    risk_reduction=risk_red,
                                    status=status,
                                    risk_tier=RiskTier.MEDIUM,
                                    created_at=datetime.now(timezone.utc),
                                    links=[]
                                )
                                items.append(item)
                            except (ValueError, IndexError):
                                continue
                                
        except Exception as e:
            self.logger.error(f"Error parsing BACKLOG.md: {e}")
        
        return items
    
    def discover_todo_fixme_comments(self) -> List[BacklogItem]:
        """Scan codebase for TODO/FIXME comments."""
        items = []
        
        try:
            # Use git grep for efficient searching
            result = subprocess.run(
                ['git', 'grep', '-n', '-i', '-E', 'TODO|FIXME'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        file_path, line_num, content = line.split(':', 2)
                        if not any(exclude in file_path for exclude in ['.git', 'node_modules', '__pycache__']):
                            comment_type = 'TODO' if 'TODO' in content.upper() else 'FIXME'
                            
                            item = BacklogItem(
                                id=f"todo-{len(items)+1}",
                                title=f"{comment_type} in {file_path}:{line_num}",
                                type=TaskType.TECH_DEBT if comment_type == 'TODO' else TaskType.BUG,
                                description=content.strip(),
                                acceptance_criteria=[f"Resolve {comment_type} comment", "Update related tests"],
                                effort=2,  # Default small effort
                                value=3,
                                time_criticality=2,
                                risk_reduction=3,
                                status=TaskStatus.NEW,
                                risk_tier=RiskTier.LOW,
                                created_at=datetime.now(timezone.utc),
                                links=[f"{file_path}:{line_num}"]
                            )
                            items.append(item)
                            
        except Exception as e:
            self.logger.error(f"Error discovering TODO/FIXME comments: {e}")
        
        return items
    
    def discover_failing_tests(self) -> List[BacklogItem]:
        """Detect failing tests and create backlog items."""
        items = []
        
        try:
            # Run pytest to detect failures
            result = subprocess.run(
                ['python', '-m', 'pytest', '--tb=no', '-q'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 and 'FAILED' in result.stdout:
                item = BacklogItem(
                    id="failing-tests",
                    title="Fix failing tests",
                    type=TaskType.BUG,
                    description="Tests are currently failing and need to be fixed",
                    acceptance_criteria=["All tests pass", "CI pipeline succeeds"],
                    effort=5,
                    value=8,
                    time_criticality=8,
                    risk_reduction=7,
                    status=TaskStatus.NEW,
                    risk_tier=RiskTier.HIGH,
                    created_at=datetime.now(timezone.utc),
                    links=[]
                )
                items.append(item)
                
        except Exception as e:
            self.logger.error(f"Error checking test status: {e}")
        
        return items


class BacklogManager:
    """Manages backlog items with WSJF prioritization and execution tracking."""
    
    def __init__(self, repo_path: Path, config_path: Path):
        self.repo_path = repo_path
        self.config_path = config_path
        self.backlog_file = repo_path / "backlog.json"
        self.discovery = BacklogDiscovery(repo_path)
        self.logger = get_core_logger()
        self.items: List[BacklogItem] = []
        
        # Load automation scope config
        self.load_automation_config()
    
    def load_automation_config(self):
        """Load automation scope configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {
                    'automation_limits': {
                        'max_prs_per_day': 5,
                        'max_branch_age_hours': 24,
                        'max_loc_per_pr': 200
                    }
                }
        except Exception as e:
            self.logger.error(f"Error loading automation config: {e}")
            self.config = {}
    
    def load_backlog(self):
        """Load backlog from persistent storage."""
        if self.backlog_file.exists():
            try:
                with open(self.backlog_file, 'r') as f:
                    data = json.load(f)
                    self.items = [BacklogItem.from_dict(item) for item in data]
            except Exception as e:
                self.logger.error(f"Error loading backlog: {e}")
                self.items = []
        else:
            self.items = []
    
    def save_backlog(self):
        """Save backlog to persistent storage."""
        try:
            data = [item.to_dict() for item in self.items]
            with open(self.backlog_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving backlog: {e}")
    
    def discover_and_sync(self):
        """Discover new tasks and sync with existing backlog."""
        self.logger.info("Starting backlog discovery...")
        
        # Discover from various sources
        new_items = []
        
        # From BACKLOG.md
        backlog_md_items = self.discovery.discover_from_backlog_md(self.repo_path / "BACKLOG.md")
        new_items.extend(backlog_md_items)
        
        # From TODO/FIXME comments
        todo_items = self.discovery.discover_todo_fixme_comments()
        new_items.extend(todo_items)
        
        # From failing tests
        test_items = self.discovery.discover_failing_tests()
        new_items.extend(test_items)
        
        # Deduplicate and merge
        self._merge_items(new_items)
        
        # Apply aging multiplier
        self._apply_aging_multiplier()
        
        # Sort by WSJF score
        self.items.sort(key=lambda x: x.wsjf_score, reverse=True)
        
        self.logger.info(f"Discovered {len(new_items)} new items, total backlog size: {len(self.items)}")
    
    def _merge_items(self, new_items: List[BacklogItem]):
        """Merge new items with existing backlog, avoiding duplicates."""
        existing_titles = {item.title for item in self.items}
        
        for new_item in new_items:
            if new_item.title not in existing_titles:
                self.items.append(new_item)
    
    def _apply_aging_multiplier(self):
        """Apply aging multiplier to boost stale but valuable items."""
        now = datetime.now(timezone.utc)
        
        for item in self.items:
            days_old = (now - item.created_at).days
            if days_old > 7:  # After a week, start applying multiplier
                item.aging_multiplier = min(2.0, 1.0 + (days_old - 7) * 0.1)
    
    def get_next_ready_task(self) -> Optional[BacklogItem]:
        """Get the highest priority ready task."""
        ready_items = [item for item in self.items if item.status == TaskStatus.READY]
        return ready_items[0] if ready_items else None
    
    def get_actionable_items(self) -> List[BacklogItem]:
        """Get all actionable items (NEW, REFINED, READY)."""
        actionable_statuses = {TaskStatus.NEW, TaskStatus.REFINED, TaskStatus.READY}
        return [item for item in self.items if item.status in actionable_statuses]
    
    def update_item_status(self, item_id: str, new_status: TaskStatus):
        """Update the status of a backlog item."""
        for item in self.items:
            if item.id == item_id:
                item.status = new_status
                break
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate metrics and status report."""
        now = datetime.now(timezone.utc)
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len([item for item in self.items if item.status == status])
        
        # Calculate DORA metrics (simplified)
        completed_today = [
            item for item in self.items 
            if item.status == TaskStatus.DONE and 
            (now - item.created_at).days == 0
        ]
        
        return {
            "timestamp": now.isoformat(),
            "backlog_size_by_status": status_counts,
            "total_items": len(self.items),
            "actionable_items": len(self.get_actionable_items()),
            "completed_today": len(completed_today),
            "top_wsjf_items": [
                {"id": item.id, "title": item.title, "wsjf": item.wsjf_score}
                for item in self.items[:5]
            ],
            "ci_summary": "pending",  # To be filled by execution system
            "risks_or_blocks": []
        }


def main():
    """Main entry point for autonomous backlog management."""
    repo_path = Path.cwd()
    config_path = repo_path / ".automation-scope.yaml"
    
    manager = BacklogManager(repo_path, config_path)
    manager.load_backlog()
    manager.discover_and_sync()
    manager.save_backlog()
    
    # Generate report
    report = manager.generate_metrics_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()