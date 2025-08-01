#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for TestGen Copilot
Implements comprehensive signal harvesting, WSJF/ICE scoring, and continuous execution.
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import yaml

from enum import Enum


class WorkItemType(Enum):
    SECURITY_FIX = "security_fix"
    TECHNICAL_DEBT = "technical_debt"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    DEPENDENCY_UPDATE = "dependency_update"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION = "documentation"
    TEST_IMPROVEMENT = "test_improvement"
    COMPLIANCE = "compliance"
    INFRASTRUCTURE = "infrastructure"
    HOUSEKEEPING = "housekeeping"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class WorkItem:
    id: str
    title: str
    description: str
    type: WorkItemType
    source: str
    priority: Priority
    
    # WSJF Components
    user_business_value: float
    time_criticality: float
    risk_reduction: float
    opportunity_enablement: float
    job_size: float
    
    # ICE Components
    impact: float
    confidence: float
    ease: float
    
    # Technical Debt Scoring
    debt_impact: float
    debt_interest: float
    hotspot_multiplier: float
    
    # Composite scoring
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    
    # Metadata
    files_affected: List[str]
    estimated_hours: float
    risk_level: float
    created_at: datetime
    dependencies: List[str]
    tags: List[str]


class ValueDiscoveryEngine:
    """Main value discovery and prioritization engine."""
    
    def __init__(self, repo_path: Path, config_path: Path):
        self.repo_path = repo_path
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path / "config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load analysis rules
        with open(config_path / "security-rules.yaml", 'r') as f:
            self.security_rules = yaml.safe_load(f)
        
        with open(config_path / "quality-rules.yaml", 'r') as f:
            self.quality_rules = yaml.safe_load(f)
        
        # Scoring weights from config
        weights = self.config['value_discovery']['scoring_weights']
        self.wsjf_weight = weights['wsjf']
        self.ice_weight = weights['ice'] 
        self.debt_weight = weights['technical_debt']
        self.security_weight = weights['security']
        
        # Discovered items cache
        self.discovered_items: List[WorkItem] = []
        self.completed_items: List[str] = []
        self.value_metrics = {}
    
    async def discover_work_items(self) -> List[WorkItem]:
        """Comprehensive signal harvesting from multiple sources."""
        self.logger.info("Starting comprehensive value discovery...")
        
        # Parallel signal harvesting
        tasks = [
            self._analyze_git_history(),
            self._run_static_analysis(),
            self._check_dependencies(),
            self._scan_security_vulnerabilities(),
            self._analyze_code_comments(),
            self._check_compliance_requirements(),
            self._analyze_test_coverage(),
            self._detect_performance_issues()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all discovered items
        all_items = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Discovery task failed: {result}")
        
        # Score and prioritize items
        scored_items = []
        for item in all_items:
            scored_item = self._calculate_composite_score(item)
            scored_items.append(scored_item)
        
        # Sort by composite score
        scored_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.discovered_items = scored_items
        self.logger.info(f"Discovered {len(scored_items)} work items")
        
        return scored_items
    
    async def _analyze_git_history(self) -> List[WorkItem]:
        """Extract work items from git history, comments, and commit patterns."""
        items = []
        
        try:
            # Get recent commits
            result = subprocess.run(
                ["git", "log", "--oneline", "-50", "--grep=TODO", "--grep=FIXME", "--grep=HACK"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            # Pattern analysis for technical debt markers
            debt_patterns = [
                r'TODO:?\s*(.+)',
                r'FIXME:?\s*(.+)',
                r'HACK:?\s*(.+)',
                r'XXX:?\s*(.+)',
                r'temporary:?\s*(.+)',
                r'quick.?fix:?\s*(.+)'
            ]
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    for pattern in debt_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            items.append(WorkItem(
                                id=f"git-{hash(line)}",
                                title=f"Address technical debt: {match.group(1)[:50]}",
                                description=f"Found in commit: {line}",
                                type=WorkItemType.TECHNICAL_DEBT,
                                source="git_history",
                                priority=Priority.MEDIUM,
                                user_business_value=3.0,
                                time_criticality=2.0,
                                risk_reduction=4.0,
                                opportunity_enablement=2.0,
                                job_size=3.0,
                                impact=4.0,
                                confidence=7.0,
                                ease=6.0,
                                debt_impact=15.0,
                                debt_interest=5.0,
                                hotspot_multiplier=1.0,
                                wsjf_score=0.0,
                                ice_score=0.0,
                                technical_debt_score=0.0,
                                composite_score=0.0,
                                files_affected=[],
                                estimated_hours=2.0,
                                risk_level=0.3,
                                created_at=datetime.now(timezone.utc),
                                dependencies=[],
                                tags=["technical-debt", "git-history"]
                            ))
            
        except Exception as e:
            self.logger.error(f"Git history analysis failed: {e}")
        
        return items
    
    async def _run_static_analysis(self) -> List[WorkItem]:
        """Run static analysis tools and extract improvement opportunities."""
        items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", "."],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues[:10]:  # Limit to top 10 issues
                    items.append(WorkItem(
                        id=f"ruff-{issue['code']}-{hash(issue['filename'])}",
                        title=f"Fix {issue['code']}: {issue['message'][:50]}",
                        description=f"Ruff found: {issue['message']} in {issue['filename']}",
                        type=WorkItemType.TECHNICAL_DEBT,
                        source="static_analysis",
                        priority=Priority.LOW,
                        user_business_value=2.0,
                        time_criticality=1.0,
                        risk_reduction=3.0,
                        opportunity_enablement=2.0,
                        job_size=1.0,
                        impact=3.0,
                        confidence=8.0,
                        ease=8.0,
                        debt_impact=5.0,
                        debt_interest=2.0,
                        hotspot_multiplier=1.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=0.0,
                        composite_score=0.0,
                        files_affected=[issue['filename']],
                        estimated_hours=0.5,
                        risk_level=0.1,
                        created_at=datetime.now(timezone.utc),
                        dependencies=[],
                        tags=["code-quality", "static-analysis"]
                    ))
                    
        except Exception as e:
            self.logger.error(f"Static analysis failed: {e}")
        
        return items
    
    async def _check_dependencies(self) -> List[WorkItem]:
        """Check for dependency updates and security vulnerabilities."""
        items = []
        
        # Check for outdated packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for package in outdated[:5]:  # Top 5 outdated packages
                    items.append(WorkItem(
                        id=f"dep-update-{package['name']}",
                        title=f"Update {package['name']} from {package['version']} to {package['latest_version']}",
                        description=f"Dependency update available: {package['name']}",
                        type=WorkItemType.DEPENDENCY_UPDATE,
                        source="dependency_check",
                        priority=Priority.LOW,
                        user_business_value=2.0,
                        time_criticality=1.0,
                        risk_reduction=3.0,
                        opportunity_enablement=3.0,
                        job_size=1.0,
                        impact=3.0,
                        confidence=6.0,
                        ease=7.0,
                        debt_impact=3.0,
                        debt_interest=1.0,
                        hotspot_multiplier=1.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=0.0,
                        composite_score=0.0,
                        files_affected=["requirements.txt", "pyproject.toml"],
                        estimated_hours=0.5,
                        risk_level=0.2,
                        created_at=datetime.now(timezone.utc),
                        dependencies=[],
                        tags=["dependencies", "updates"]
                    ))
                    
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
        
        return items
    
    async def _scan_security_vulnerabilities(self) -> List[WorkItem]:
        """Scan for security vulnerabilities using multiple tools."""
        items = []
        
        # Run bandit security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                bandit_report = json.loads(result.stdout)
                for issue in bandit_report.get('results', [])[:5]:
                    severity_map = {"LOW": Priority.LOW, "MEDIUM": Priority.HIGH, "HIGH": Priority.CRITICAL}
                    
                    items.append(WorkItem(
                        id=f"security-{issue['test_id']}-{hash(issue['filename'])}",
                        title=f"Security issue: {issue['issue_text'][:50]}",
                        description=f"Bandit found: {issue['issue_text']} in {issue['filename']}",
                        type=WorkItemType.SECURITY_FIX,
                        source="security_scan",
                        priority=severity_map.get(issue['issue_severity'], Priority.MEDIUM),
                        user_business_value=8.0,
                        time_criticality=7.0,
                        risk_reduction=9.0,
                        opportunity_enablement=5.0,
                        job_size=3.0,
                        impact=8.0,
                        confidence=9.0,
                        ease=5.0,
                        debt_impact=20.0,
                        debt_interest=10.0,
                        hotspot_multiplier=2.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=0.0,
                        composite_score=0.0,
                        files_affected=[issue['filename']],
                        estimated_hours=3.0,
                        risk_level=0.7,
                        created_at=datetime.now(timezone.utc),
                        dependencies=[],
                        tags=["security", "vulnerability"]
                    ))
                    
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
        
        return items
    
    async def _analyze_code_comments(self) -> List[WorkItem]:
        """Extract work items from TODO, FIXME, and other code comments."""
        items = []
        
        try:
            # Search for TODO/FIXME comments in Python files
            result = subprocess.run(
                ["grep", "-r", "-n", "-E", "(TODO|FIXME|HACK|XXX|BUG)", "src/", "--include=*.py"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n')[:10]:  # Limit to 10 items
                if line.strip():
                    parts = line.split(':', 3)
                    if len(parts) >= 3:
                        filename, line_num, comment = parts[0], parts[1], parts[2]
                        
                        items.append(WorkItem(
                            id=f"comment-{hash(line)}",
                            title=f"Address code comment: {comment[:50]}",
                            description=f"Found in {filename}:{line_num} - {comment}",
                            type=WorkItemType.TECHNICAL_DEBT,
                            source="code_comments",
                            priority=Priority.MEDIUM,
                            user_business_value=3.0,
                            time_criticality=2.0,
                            risk_reduction=4.0,
                            opportunity_enablement=3.0,
                            job_size=2.0,
                            impact=4.0,
                            confidence=7.0,
                            ease=6.0,
                            debt_impact=10.0,
                            debt_interest=3.0,
                            hotspot_multiplier=1.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            technical_debt_score=0.0,
                            composite_score=0.0,
                            files_affected=[filename],
                            estimated_hours=1.5,
                            risk_level=0.3,
                            created_at=datetime.now(timezone.utc),
                            dependencies=[],
                            tags=["technical-debt", "code-comments"]
                        ))
                        
        except Exception as e:
            self.logger.error(f"Code comment analysis failed: {e}")
        
        return items
    
    async def _check_compliance_requirements(self) -> List[WorkItem]:
        """Check for compliance and governance requirements."""
        items = []
        
        # Check for missing security files
        security_files = [
            "SECURITY.md", ".github/SECURITY.md",
            "CODE_OF_CONDUCT.md", ".github/CODE_OF_CONDUCT.md"
        ]
        
        for file_path in security_files:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                items.append(WorkItem(
                    id=f"compliance-{file_path.replace('/', '-')}",
                    title=f"Add missing compliance file: {file_path}",
                    description=f"Required compliance file {file_path} is missing",
                    type=WorkItemType.COMPLIANCE,
                    source="compliance_check",
                    priority=Priority.MEDIUM,
                    user_business_value=5.0,
                    time_criticality=3.0,
                    risk_reduction=6.0,
                    opportunity_enablement=4.0,
                    job_size=1.0,
                    impact=5.0,
                    confidence=8.0,
                    ease=9.0,
                    debt_impact=8.0,
                    debt_interest=2.0,
                    hotspot_multiplier=1.0,
                    wsjf_score=0.0,
                    ice_score=0.0,
                    technical_debt_score=0.0,
                    composite_score=0.0,
                    files_affected=[file_path],
                    estimated_hours=1.0,
                    risk_level=0.2,
                    created_at=datetime.now(timezone.utc),
                    dependencies=[],
                    tags=["compliance", "documentation"]
                ))
        
        return items
    
    async def _analyze_test_coverage(self) -> List[WorkItem]:
        """Analyze test coverage and identify gaps."""
        items = []
        
        try:
            # Run coverage analysis
            result = subprocess.run(
                ["pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            # Parse coverage report (simplified)
            if "missing" in result.stdout.lower():
                items.append(WorkItem(
                    id="test-coverage-improvement",
                    title="Improve test coverage for missing lines",
                    description="Test coverage analysis found missing coverage",
                    type=WorkItemType.TEST_IMPROVEMENT,
                    source="coverage_analysis",
                    priority=Priority.MEDIUM,
                    user_business_value=6.0,
                    time_criticality=3.0,
                    risk_reduction=7.0,
                    opportunity_enablement=5.0,
                    job_size=4.0,
                    impact=6.0,
                    confidence=7.0,
                    ease=5.0,
                    debt_impact=12.0,
                    debt_interest=4.0,
                    hotspot_multiplier=1.2,
                    wsjf_score=0.0,
                    ice_score=0.0,
                    technical_debt_score=0.0,
                    composite_score=0.0,
                    files_affected=["tests/"],
                    estimated_hours=4.0,
                    risk_level=0.3,
                    created_at=datetime.now(timezone.utc),
                    dependencies=[],
                    tags=["testing", "coverage"]
                ))
                
        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {e}")
        
        return items
    
    async def _detect_performance_issues(self) -> List[WorkItem]:
        """Detect potential performance optimization opportunities."""
        items = []
        
        # Simple heuristic-based performance issue detection
        performance_patterns = [
            r'for.*in.*:.*for.*in.*:',  # Nested loops
            r'\.append\(.*\).*in.*for',  # List append in loop
            r'open\(.*\).*in.*for',     # File operations in loop
        ]
        
        try:
            for pattern in performance_patterns:
                result = subprocess.run(
                    ["grep", "-r", "-E", pattern, "src/", "--include=*.py"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.stdout.strip():
                    items.append(WorkItem(
                        id=f"perf-{hash(pattern)}",
                        title=f"Performance optimization opportunity detected",
                        description=f"Pattern suggests performance issue: {pattern}",
                        type=WorkItemType.PERFORMANCE_OPTIMIZATION,
                        source="performance_analysis",
                        priority=Priority.LOW,
                        user_business_value=4.0,
                        time_criticality=2.0,
                        risk_reduction=3.0,
                        opportunity_enablement=6.0,
                        job_size=3.0,
                        impact=5.0,
                        confidence=6.0,
                        ease=4.0,
                        debt_impact=8.0,
                        debt_interest=3.0,
                        hotspot_multiplier=1.1,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=0.0,
                        composite_score=0.0,
                        files_affected=[],
                        estimated_hours=3.0,
                        risk_level=0.4,
                        created_at=datetime.now(timezone.utc),
                        dependencies=[],
                        tags=["performance", "optimization"]
                    ))
                    break  # Only add one performance item for now
                    
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
        
        return items
    
    def _calculate_composite_score(self, item: WorkItem) -> WorkItem:
        """Calculate comprehensive value score using WSJF, ICE, and technical debt."""
        
        # WSJF Calculation
        cost_of_delay = (
            item.user_business_value + 
            item.time_criticality + 
            item.risk_reduction + 
            item.opportunity_enablement
        )
        item.wsjf_score = cost_of_delay / max(item.job_size, 0.1)
        
        # ICE Calculation
        item.ice_score = item.impact * item.confidence * item.ease
        
        # Technical Debt Score
        item.technical_debt_score = (
            (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
        )
        
        # Composite Score with adaptive weights
        composite = (
            self.wsjf_weight * self._normalize_score(item.wsjf_score, 0, 50) +
            self.ice_weight * self._normalize_score(item.ice_score, 0, 1000) +
            self.debt_weight * self._normalize_score(item.technical_debt_score, 0, 100) +
            self.security_weight * (10 if item.type == WorkItemType.SECURITY_FIX else 0)
        )
        
        # Apply boost factors
        boost_factors = self.config['value_discovery']['boost_factors']
        if item.type == WorkItemType.SECURITY_FIX:
            composite *= boost_factors['security_vulnerability']
        elif item.type == WorkItemType.COMPLIANCE:
            composite *= boost_factors['compliance_blocking']
        elif item.type == WorkItemType.DEPENDENCY_UPDATE:
            composite *= boost_factors['dependency_update']
        elif item.type == WorkItemType.DOCUMENTATION:
            composite *= boost_factors['documentation_only']
        
        item.composite_score = composite
        return item
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val == min_val:
            return 0.0
        return max(0.0, min(100.0, ((score - min_val) / (max_val - min_val)) * 100))
    
    def select_next_best_value_item(self) -> Optional[WorkItem]:
        """Select the next highest-value work item for execution."""
        if not self.discovered_items:
            return None
        
        # Filter out completed items and apply constraints
        available_items = [
            item for item in self.discovered_items 
            if item.id not in self.completed_items
        ]
        
        if not available_items:
            return None
        
        # Apply selection criteria
        thresholds = self.config['value_discovery']['thresholds']
        
        for item in available_items:
            # Check minimum score threshold
            if item.composite_score < thresholds['min_composite_score']:
                continue
            
            # Check risk threshold  
            if item.risk_level > thresholds['max_execution_risk']:
                continue
            
            # Check confidence threshold
            if item.confidence < thresholds['min_confidence']:
                continue
            
            # Found suitable item
            return item
        
        # No items meet criteria - generate housekeeping task
        return self._generate_housekeeping_task()
    
    def _generate_housekeeping_task(self) -> WorkItem:
        """Generate a low-priority housekeeping task when no high-value items exist."""
        return WorkItem(
            id=f"housekeeping-{int(time.time())}",
            title="Dependency maintenance and cleanup",
            description="Routine maintenance: update dependencies, clean dead code",
            type=WorkItemType.HOUSEKEEPING,
            source="housekeeping_generator",
            priority=Priority.LOW,
            user_business_value=2.0,
            time_criticality=1.0,
            risk_reduction=2.0,
            opportunity_enablement=2.0,
            job_size=2.0,
            impact=3.0,
            confidence=8.0,
            ease=7.0,
            debt_impact=5.0,
            debt_interest=1.0,
            hotspot_multiplier=1.0,
            wsjf_score=3.5,
            ice_score=168.0,
            technical_debt_score=5.0,
            composite_score=12.0,
            files_affected=[],
            estimated_hours=2.0,
            risk_level=0.1,
            created_at=datetime.now(timezone.utc),
            dependencies=[],
            tags=["housekeeping", "maintenance"]
        )
    
    def save_value_metrics(self, metrics_path: Path):
        """Save comprehensive value discovery metrics."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "discovered_items_count": len(self.discovered_items),
            "completed_items_count": len(self.completed_items),
            "top_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "type": item.type.value,
                    "composite_score": item.composite_score,
                    "wsjf_score": item.wsjf_score,
                    "ice_score": item.ice_score,
                    "technical_debt_score": item.technical_debt_score
                }
                for item in self.discovered_items[:10]
            ],
            "items_by_type": {
                item_type.value: len([
                    item for item in self.discovered_items 
                    if item.type == item_type
                ])
                for item_type in WorkItemType
            },
            "average_scores": {
                "composite": sum(item.composite_score for item in self.discovered_items) / max(len(self.discovered_items), 1),
                "wsjf": sum(item.wsjf_score for item in self.discovered_items) / max(len(self.discovered_items), 1),
                "ice": sum(item.ice_score for item in self.discovered_items) / max(len(self.discovered_items), 1),
                "technical_debt": sum(item.technical_debt_score for item in self.discovered_items) / max(len(self.discovered_items), 1)
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Value metrics saved to {metrics_path}")


if __name__ == "__main__":
    # CLI for testing the value discovery engine
    import argparse
    
    parser = argparse.ArgumentParser(description="TestGen Copilot Value Discovery Engine")
    parser.add_argument("--repo-path", type=Path, default=Path.cwd())
    parser.add_argument("--config-path", type=Path, default=Path.cwd() / ".terragon")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "value-metrics.json")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run value discovery
    engine = ValueDiscoveryEngine(args.repo_path, args.config_path)
    
    async def main():
        items = await engine.discover_work_items()
        next_item = engine.select_next_best_value_item()
        
        if next_item:
            print(f"Next best value item: {next_item.title}")
            print(f"Composite score: {next_item.composite_score:.2f}")
            print(f"Type: {next_item.type.value}")
            print(f"Estimated hours: {next_item.estimated_hours}")
        
        engine.save_value_metrics(args.output)
    
    asyncio.run(main())