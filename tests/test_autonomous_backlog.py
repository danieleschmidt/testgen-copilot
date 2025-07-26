#!/usr/bin/env python3
"""Tests for autonomous backlog management system."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone

from testgen_copilot.autonomous_backlog import (
    BacklogManager, BacklogItem, TaskType, TaskStatus, RiskTier, BacklogDiscovery
)


def test_backlog_item_wsjf_calculation():
    """Test WSJF score calculation."""
    item = BacklogItem(
        id="test-1",
        title="Test Task",
        type=TaskType.FEATURE,
        description="Test description",
        acceptance_criteria=["Complete implementation"],
        effort=5,
        value=8,
        time_criticality=6,
        risk_reduction=4,
        status=TaskStatus.NEW,
        risk_tier=RiskTier.MEDIUM,
        created_at=datetime.now(timezone.utc),
        links=[],
        aging_multiplier=1.0
    )
    
    # WSJF = (value + time_criticality + risk_reduction) / effort
    # WSJF = (8 + 6 + 4) / 5 = 18 / 5 = 3.6
    expected_wsjf = 3.6
    assert item.wsjf_score == expected_wsjf


def test_backlog_item_serialization():
    """Test backlog item to/from dict conversion."""
    item = BacklogItem(
        id="test-1",
        title="Test Task",
        type=TaskType.BUG,
        description="Test bug fix",
        acceptance_criteria=["Fix bug", "Add test"],
        effort=3,
        value=7,
        time_criticality=8,
        risk_reduction=5,
        status=TaskStatus.READY,
        risk_tier=RiskTier.HIGH,
        created_at=datetime.now(timezone.utc),
        links=["file.py:123"],
        aging_multiplier=1.2
    )
    
    # Convert to dict and back
    item_dict = item.to_dict()
    restored_item = BacklogItem.from_dict(item_dict)
    
    assert restored_item.id == item.id
    assert restored_item.title == item.title
    assert restored_item.type == item.type
    assert restored_item.status == item.status
    assert restored_item.wsjf_score == item.wsjf_score


def test_backlog_discovery_todo_comments():
    """Test discovery of TODO/FIXME comments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Create a file with TODO comments
        test_file = repo_path / "test.py"
        test_file.write_text("""
def test_function():
    # TODO: implement this function
    pass

def another_function():
    # FIXME: this has a bug
    return None
""")
        
        # Initialize git repo (required for git grep)
        import subprocess
        subprocess.run(['git', 'init'], cwd=repo_path, capture_output=True)
        subprocess.run(['git', 'add', '.'], cwd=repo_path, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_path, capture_output=True)
        
        discovery = BacklogDiscovery(repo_path)
        items = discovery.discover_todo_fixme_comments()
        
        assert len(items) >= 2  # Should find TODO and FIXME
        
        todo_items = [item for item in items if 'TODO' in item.title]
        fixme_items = [item for item in items if 'FIXME' in item.title]
        
        assert len(todo_items) >= 1
        assert len(fixme_items) >= 1


def test_backlog_manager_basic_functionality():
    """Test basic backlog manager functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        config_path = repo_path / ".automation-scope.yaml"
        
        # Create basic config
        config_path.write_text("""
automation_limits:
  max_prs_per_day: 5
  max_branch_age_hours: 24
""")
        
        manager = BacklogManager(repo_path, config_path)
        
        # Test empty backlog
        manager.load_backlog()
        assert len(manager.items) == 0
        
        # Add test item
        test_item = BacklogItem(
            id="test-1",
            title="Test Feature",
            type=TaskType.FEATURE,
            description="Test implementation",
            acceptance_criteria=["Implement feature"],
            effort=3,
            value=8,
            time_criticality=5,
            risk_reduction=2,
            status=TaskStatus.NEW,
            risk_tier=RiskTier.LOW,
            created_at=datetime.now(timezone.utc),
            links=[]
        )
        
        manager.items = [test_item]
        manager.save_backlog()
        
        # Test loading
        manager2 = BacklogManager(repo_path, config_path)
        manager2.load_backlog()
        
        assert len(manager2.items) == 1
        assert manager2.items[0].title == "Test Feature"


def test_backlog_prioritization():
    """Test WSJF-based prioritization."""
    items = [
        BacklogItem(
            id="low-priority",
            title="Low Priority Task",
            type=TaskType.FEATURE,
            description="Low priority",
            acceptance_criteria=["Complete"],
            effort=8,  # High effort
            value=3,   # Low value
            time_criticality=2,
            risk_reduction=1,
            status=TaskStatus.NEW,
            risk_tier=RiskTier.LOW,
            created_at=datetime.now(timezone.utc),
            links=[]
        ),
        BacklogItem(
            id="high-priority",
            title="High Priority Task",
            type=TaskType.SECURITY,
            description="High priority security fix",
            acceptance_criteria=["Fix vulnerability"],
            effort=2,  # Low effort
            value=8,   # High value
            time_criticality=8,
            risk_reduction=8,
            status=TaskStatus.NEW,
            risk_tier=RiskTier.CRITICAL,
            created_at=datetime.now(timezone.utc),
            links=[]
        )
    ]
    
    # Sort by WSJF (higher is better)
    items.sort(key=lambda x: x.wsjf_score, reverse=True)
    
    assert items[0].id == "high-priority"
    assert items[1].id == "low-priority"
    
    # High priority: (8 + 8 + 8) / 2 = 12.0
    # Low priority: (3 + 2 + 1) / 8 = 0.75
    assert items[0].wsjf_score == 12.0
    assert items[1].wsjf_score == 0.75


def test_metrics_generation():
    """Test backlog metrics generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        config_path = repo_path / ".automation-scope.yaml"
        
        config_path.write_text("automation_limits:\n  max_prs_per_day: 5")
        
        manager = BacklogManager(repo_path, config_path)
        
        # Add test items with different statuses
        manager.items = [
            BacklogItem(
                id="new-1", title="New Task", type=TaskType.FEATURE,
                description="New", acceptance_criteria=["Done"], effort=3,
                value=5, time_criticality=4, risk_reduction=3,
                status=TaskStatus.NEW, risk_tier=RiskTier.LOW,
                created_at=datetime.now(timezone.utc), links=[]
            ),
            BacklogItem(
                id="ready-1", title="Ready Task", type=TaskType.BUG,
                description="Ready", acceptance_criteria=["Fixed"], effort=2,
                value=7, time_criticality=6, risk_reduction=4,
                status=TaskStatus.READY, risk_tier=RiskTier.MEDIUM,
                created_at=datetime.now(timezone.utc), links=[]
            ),
            BacklogItem(
                id="done-1", title="Done Task", type=TaskType.TEST,
                description="Completed", acceptance_criteria=["Tested"], effort=1,
                value=4, time_criticality=2, risk_reduction=1,
                status=TaskStatus.DONE, risk_tier=RiskTier.LOW,
                created_at=datetime.now(timezone.utc), links=[]
            )
        ]
        
        report = manager.generate_metrics_report()
        
        assert report["total_items"] == 3
        assert report["actionable_items"] == 2  # NEW and READY
        assert report["backlog_size_by_status"]["NEW"] == 1
        assert report["backlog_size_by_status"]["READY"] == 1
        assert report["backlog_size_by_status"]["DONE"] == 1
        assert len(report["top_wsjf_items"]) <= 5