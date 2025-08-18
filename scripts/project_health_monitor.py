#!/usr/bin/env python3
"""
Project Health Monitor for TestGen-Copilot

Monitors overall project health including code quality, dependencies,
security, performance, and business metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import requests
from dataclasses import dataclass, asdict


@dataclass
class HealthMetric:
    """Represents a health metric."""
    name: str
    value: float
    unit: str
    status: str  # "healthy", "warning", "critical"
    threshold_warning: float
    threshold_critical: float
    trend: str  # "improving", "stable", "declining"
    last_updated: datetime


class ProjectHealthMonitor:
    """Monitors comprehensive project health."""
    
    def __init__(self, config_path: str = ".github/health-config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.metrics: List[HealthMetric] = []
        self.alerts: List[str] = []
        self.timestamp = datetime.now()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load health monitoring configuration."""
        default_config = {
            "thresholds": {
                "code_quality_score": {"warning": 85, "critical": 70},
                "test_coverage": {"warning": 80, "critical": 60},
                "security_score": {"warning": 90, "critical": 75},
                "dependency_freshness": {"warning": 30, "critical": 90},  # days
                "build_success_rate": {"warning": 95, "critical": 90},
                "performance_score": {"warning": 80, "critical": 60},
                "documentation_coverage": {"warning": 75, "critical": 50}
            },
            "github": {
                "enabled": True,
                "api_url": "https://api.github.com",
                "repo": os.getenv("GITHUB_REPOSITORY", ""),
                "token": os.getenv("GITHUB_TOKEN", "")
            },
            "notifications": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK", ""),
                "email_enabled": False
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for section, values in default_config.items():
                    if section not in config:
                        config[section] = values
                    else:
                        for key, value in values.items():
                            if key not in config[section]:
                                config[section][key] = value
                return config
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
        
        return default_config
    
    def run_command(self, command: List[str]) -> Dict[str, Any]:
        """Run command and return structured result."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": 124
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": 1
            }
    
    def check_code_quality_health(self) -> HealthMetric:
        """Check overall code quality health."""
        print("  📊 Analyzing code quality...")
        
        # Run comprehensive quality checks
        total_score = 100.0
        issues = []
        
        # Ruff linting
        result = self.run_command(["ruff", "check", "src/", "--output-format=json"])
        if result["success"] and result["stdout"]:
            try:
                ruff_issues = json.loads(result["stdout"])
                if ruff_issues:
                    total_score -= min(30, len(ruff_issues) * 2)
                    issues.extend(ruff_issues)
            except:
                pass
        
        # Black formatting
        result = self.run_command(["black", "--check", "src/", "tests/"])
        if not result["success"]:
            total_score -= 15
            issues.append("Code formatting issues detected")
        
        # MyPy type checking
        result = self.run_command(["mypy", "src/testgen_copilot"])
        if not result["success"]:
            error_count = result["stderr"].count("error:") if result["stderr"] else 1
            total_score -= min(25, error_count * 3)
            issues.append(f"{error_count} type checking errors")
        
        thresholds = self.config["thresholds"]["code_quality_score"]
        status = self._get_status(total_score, thresholds["warning"], thresholds["critical"])
        
        return HealthMetric(
            name="code_quality_score",
            value=total_score,
            unit="percent",
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            trend="stable",  # TODO: Calculate trend from historical data
            last_updated=self.timestamp
        )
    
    def check_test_coverage_health(self) -> HealthMetric:
        """Check test coverage health."""
        print("  🧪 Analyzing test coverage...")
        
        # Run tests with coverage
        result = self.run_command([
            "pytest", "tests/", "--cov=src/testgen_copilot", 
            "--cov-report=json", "--quiet"
        ])
        
        coverage_percent = 0.0
        try:
            if Path("coverage.json").exists():
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                    coverage_percent = coverage_data["totals"]["percent_covered"]
        except Exception as e:
            print(f"    Warning: Could not parse coverage: {e}")
        
        thresholds = self.config["thresholds"]["test_coverage"]
        status = self._get_status(coverage_percent, thresholds["warning"], thresholds["critical"])
        
        return HealthMetric(
            name="test_coverage",
            value=coverage_percent,
            unit="percent",
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            trend="stable",
            last_updated=self.timestamp
        )
    
    def check_security_health(self) -> HealthMetric:
        """Check security health."""
        print("  🔒 Analyzing security posture...")
        
        security_score = 100.0
        
        # Run bandit security scan
        result = self.run_command([
            "bandit", "-r", "src/", "-f", "json", "-o", "bandit-temp.json"
        ])
        
        try:
            if Path("bandit-temp.json").exists():
                with open("bandit-temp.json", "r") as f:
                    bandit_data = json.load(f)
                    
                    high_issues = len([r for r in bandit_data.get("results", []) 
                                     if r.get("issue_severity") == "HIGH"])
                    medium_issues = len([r for r in bandit_data.get("results", []) 
                                       if r.get("issue_severity") == "MEDIUM"])
                    
                    security_score -= high_issues * 25
                    security_score -= medium_issues * 10
                
                Path("bandit-temp.json").unlink()
        except Exception as e:
            print(f"    Warning: Could not analyze bandit results: {e}")
            security_score -= 20
        
        # Run safety check
        result = self.run_command(["safety", "check", "--json"])
        if not result["success"] and "vulnerabilities found" in result["stderr"]:
            # Parse number of vulnerabilities
            try:
                vulnerability_count = result["stderr"].count("vulnerability")
                security_score -= vulnerability_count * 15
            except:
                security_score -= 30
        
        security_score = max(0, security_score)
        
        thresholds = self.config["thresholds"]["security_score"]
        status = self._get_status(security_score, thresholds["warning"], thresholds["critical"])
        
        return HealthMetric(
            name="security_score",
            value=security_score,
            unit="percent",
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            trend="stable",
            last_updated=self.timestamp
        )
    
    def check_dependency_health(self) -> HealthMetric:
        """Check dependency health and freshness."""
        print("  📦 Analyzing dependencies...")
        
        # Check for outdated packages
        result = self.run_command(["pip", "list", "--outdated", "--format=json"])
        
        outdated_days = 0
        try:
            if result["success"] and result["stdout"]:
                outdated_packages = json.loads(result["stdout"])
                # Estimate average age (simplified calculation)
                outdated_days = len(outdated_packages) * 15  # Rough estimate
        except Exception as e:
            print(f"    Warning: Could not analyze outdated packages: {e}")
            outdated_days = 60  # Assume moderate staleness
        
        thresholds = self.config["thresholds"]["dependency_freshness"]
        status = self._get_status_inverse(outdated_days, thresholds["warning"], thresholds["critical"])
        
        return HealthMetric(
            name="dependency_freshness",
            value=outdated_days,
            unit="days",
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            trend="stable",
            last_updated=self.timestamp
        )
    
    def check_performance_health(self) -> HealthMetric:
        """Check performance health."""
        print("  ⚡ Analyzing performance...")
        
        performance_score = 100.0
        
        # Measure build time
        start_time = time.time()
        result = self.run_command(["python", "-m", "build", "--outdir", "temp-dist"])
        build_time = time.time() - start_time
        
        # Penalize slow builds
        if build_time > 120:  # 2 minutes
            performance_score -= min(40, (build_time - 120) / 10)
        
        if not result["success"]:
            performance_score -= 30
        
        # Clean up
        if Path("temp-dist").exists():
            import shutil
            shutil.rmtree("temp-dist")
        
        # Run simple performance test if available
        if Path("tests/performance").exists():
            result = self.run_command([
                "pytest", "tests/performance/", "--benchmark-only", "--quiet"
            ])
            if not result["success"]:
                performance_score -= 20
        
        performance_score = max(0, performance_score)
        
        thresholds = self.config["thresholds"]["performance_score"]
        status = self._get_status(performance_score, thresholds["warning"], thresholds["critical"])
        
        return HealthMetric(
            name="performance_score",
            value=performance_score,
            unit="percent",
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            trend="stable",
            last_updated=self.timestamp
        )
    
    def check_github_health(self) -> List[HealthMetric]:
        """Check GitHub repository health metrics."""
        print("  🐙 Analyzing GitHub metrics...")
        
        metrics = []
        
        if not self.config["github"]["enabled"] or not self.config["github"]["token"]:
            print("    Skipping GitHub metrics (not configured)")
            return metrics
        
        repo = self.config["github"]["repo"]
        token = self.config["github"]["token"]
        api_url = self.config["github"]["api_url"]
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Get repository statistics
            response = requests.get(f"{api_url}/repos/{repo}", headers=headers, timeout=10)
            if response.status_code == 200:
                repo_data = response.json()
                
                # Open issues health
                open_issues = repo_data.get("open_issues_count", 0)
                issues_status = "healthy" if open_issues < 10 else "warning" if open_issues < 25 else "critical"
                
                metrics.append(HealthMetric(
                    name="open_issues",
                    value=open_issues,
                    unit="count",
                    status=issues_status,
                    threshold_warning=10,
                    threshold_critical=25,
                    trend="stable",
                    last_updated=self.timestamp
                ))
                
                # Repository activity (stars, forks)
                stars = repo_data.get("stargazers_count", 0)
                forks = repo_data.get("forks_count", 0)
                
                metrics.append(HealthMetric(
                    name="github_stars",
                    value=stars,
                    unit="count",
                    status="healthy",
                    threshold_warning=0,
                    threshold_critical=0,
                    trend="stable",
                    last_updated=self.timestamp
                ))
            
            # Get recent workflow runs
            response = requests.get(f"{api_url}/repos/{repo}/actions/runs", 
                                  headers=headers, params={"per_page": 10}, timeout=10)
            if response.status_code == 200:
                workflows = response.json()
                
                recent_runs = workflows.get("workflow_runs", [])
                if recent_runs:
                    successful_runs = sum(1 for run in recent_runs 
                                        if run.get("conclusion") == "success")
                    success_rate = (successful_runs / len(recent_runs)) * 100
                    
                    thresholds = self.config["thresholds"]["build_success_rate"]
                    status = self._get_status(success_rate, thresholds["warning"], thresholds["critical"])
                    
                    metrics.append(HealthMetric(
                        name="build_success_rate",
                        value=success_rate,
                        unit="percent",
                        status=status,
                        threshold_warning=thresholds["warning"],
                        threshold_critical=thresholds["critical"],
                        trend="stable",
                        last_updated=self.timestamp
                    ))
        
        except Exception as e:
            print(f"    Warning: Could not fetch GitHub metrics: {e}")
        
        return metrics
    
    def _get_status(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get status based on value and thresholds (higher is better)."""
        if value >= warning_threshold:
            return "healthy"
        elif value >= critical_threshold:
            return "warning"
        else:
            return "critical"
    
    def _get_status_inverse(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get status based on value and thresholds (lower is better)."""
        if value <= warning_threshold:
            return "healthy"
        elif value <= critical_threshold:
            return "warning"
        else:
            return "critical"
    
    def collect_all_metrics(self) -> List[HealthMetric]:
        """Collect all health metrics."""
        print("🏥 Collecting Project Health Metrics")
        print("=" * 40)
        
        self.metrics = []
        
        # Core health checks
        self.metrics.append(self.check_code_quality_health())
        self.metrics.append(self.check_test_coverage_health())
        self.metrics.append(self.check_security_health())
        self.metrics.append(self.check_dependency_health())
        self.metrics.append(self.check_performance_health())
        
        # GitHub metrics
        github_metrics = self.check_github_health()
        self.metrics.extend(github_metrics)
        
        return self.metrics
    
    def generate_alerts(self) -> List[str]:
        """Generate alerts based on critical metrics."""
        self.alerts = []
        
        for metric in self.metrics:
            if metric.status == "critical":
                self.alerts.append(f"CRITICAL: {metric.name} is {metric.value}{metric.unit} "
                                 f"(threshold: {metric.threshold_critical})")
            elif metric.status == "warning":
                self.alerts.append(f"WARNING: {metric.name} is {metric.value}{metric.unit} "
                                 f"(threshold: {metric.threshold_warning})")
        
        return self.alerts
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        healthy_count = sum(1 for m in self.metrics if m.status == "healthy")
        warning_count = sum(1 for m in self.metrics if m.status == "warning")
        critical_count = sum(1 for m in self.metrics if m.status == "critical")
        
        overall_health = "healthy"
        if critical_count > 0:
            overall_health = "critical"
        elif warning_count > 0:
            overall_health = "warning"
        
        health_score = (healthy_count / len(self.metrics) * 100) if self.metrics else 0
        
        report = {
            "timestamp": self.timestamp.isoformat(),
            "overall_health": overall_health,
            "health_score": health_score,
            "metrics_total": len(self.metrics),
            "healthy_count": healthy_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "alerts": self.alerts,
            "metrics": [asdict(m) for m in self.metrics],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for metric in self.metrics:
            if metric.status == "critical":
                if metric.name == "code_quality_score":
                    recommendations.append("Fix code quality issues: run 'ruff check' and 'black .'")
                elif metric.name == "test_coverage":
                    recommendations.append("Improve test coverage: add more unit tests")
                elif metric.name == "security_score":
                    recommendations.append("Address security issues: run 'bandit -r src/' and 'safety check'")
                elif metric.name == "dependency_freshness":
                    recommendations.append("Update dependencies: run 'pip list --outdated'")
                elif metric.name == "performance_score":
                    recommendations.append("Optimize performance: profile slow operations")
                elif metric.name == "build_success_rate":
                    recommendations.append("Fix CI/CD pipeline: check recent workflow failures")
            elif metric.status == "warning":
                recommendations.append(f"Monitor {metric.name}: approaching threshold")
        
        if not recommendations:
            recommendations.append("Project health looks good! Keep up the great work.")
        
        return recommendations
    
    def send_notifications(self, report: Dict[str, Any]):
        """Send notifications for critical issues."""
        if not self.alerts:
            return
        
        webhook_url = self.config["notifications"]["slack_webhook"]
        if webhook_url:
            try:
                message = {
                    "text": f"🚨 TestGen-Copilot Health Alert",
                    "attachments": [
                        {
                            "color": "danger" if report["overall_health"] == "critical" else "warning",
                            "fields": [
                                {
                                    "title": "Overall Health",
                                    "value": report["overall_health"].upper(),
                                    "short": True
                                },
                                {
                                    "title": "Health Score",
                                    "value": f"{report['health_score']:.1f}%",
                                    "short": True
                                },
                                {
                                    "title": "Critical Issues",
                                    "value": str(report["critical_count"]),
                                    "short": True
                                },
                                {
                                    "title": "Warnings",
                                    "value": str(report["warning_count"]),
                                    "short": True
                                }
                            ],
                            "text": "\n".join(self.alerts[:5])  # Limit to 5 alerts
                        }
                    ]
                }
                
                response = requests.post(webhook_url, json=message, timeout=10)
                if response.status_code == 200:
                    print("  📢 Slack notification sent")
                else:
                    print(f"  ⚠️  Failed to send Slack notification: {response.status_code}")
            
            except Exception as e:
                print(f"  ⚠️  Error sending notification: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print health summary to console."""
        print("\n" + "=" * 50)
        print("🏥 PROJECT HEALTH SUMMARY")
        print("=" * 50)
        
        status_emoji = {
            "healthy": "💚",
            "warning": "⚠️",
            "critical": "🔴"
        }
        
        overall_status = report["overall_health"]
        print(f"Overall Health: {status_emoji[overall_status]} {overall_status.upper()}")
        print(f"Health Score: {report['health_score']:.1f}%")
        print(f"Metrics: {report['healthy_count']} healthy, {report['warning_count']} warnings, {report['critical_count']} critical")
        
        print("\nDetailed Metrics:")
        for metric in self.metrics:
            emoji = status_emoji[metric.status]
            print(f"  {emoji} {metric.name.replace('_', ' ').title()}: {metric.value}{metric.unit}")
        
        if self.alerts:
            print(f"\n🚨 Alerts ({len(self.alerts)}):")
            for alert in self.alerts[:5]:  # Show first 5 alerts
                print(f"  • {alert}")
            if len(self.alerts) > 5:
                print(f"  ... and {len(self.alerts) - 5} more")
        
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"][:3]:  # Show top 3 recommendations
                print(f"  • {rec}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor TestGen-Copilot project health")
    parser.add_argument("--config", help="Path to health configuration file")
    parser.add_argument("--output", help="Output file for report", default="health-report.json")
    parser.add_argument("--notify", action="store_true", help="Send notifications for critical issues")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    monitor = ProjectHealthMonitor(config_path=args.config or ".github/health-config.json")
    
    try:
        metrics = monitor.collect_all_metrics()
        alerts = monitor.generate_alerts()
        report = monitor.generate_report()
        
        # Save report
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        if not args.quiet:
            monitor.print_summary(report)
            print(f"\n📊 Full report saved to {args.output}")
        
        # Send notifications if requested and there are critical issues
        if args.notify and report["critical_count"] > 0:
            monitor.send_notifications(report)
        
        # Exit with error code if there are critical issues
        sys.exit(0 if report["overall_health"] != "critical" else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Health monitoring interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error monitoring project health: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()