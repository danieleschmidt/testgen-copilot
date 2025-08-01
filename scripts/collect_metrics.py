#!/usr/bin/env python3
"""
TestGen-Copilot Metrics Collection Script

This script collects various project metrics and generates reports for monitoring
project health, performance, and business KPIs.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from dataclasses import dataclass

@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    value: float
    unit: str
    timestamp: datetime
    source: str
    tags: Dict[str, str] = None

class MetricsCollector:
    """Main metrics collection class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or ".github/project-metrics.json"
        self.config = self._load_config()
        self.metrics = {}
        self.errors = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        print("🔍 Collecting project metrics...")
        
        # Collect different types of metrics
        self.collect_code_quality_metrics()
        self.collect_security_metrics()
        self.collect_performance_metrics()
        self.collect_development_metrics()
        self.collect_business_metrics()
        
        # Generate summary
        summary = self._generate_summary()
        
        if self.errors:
            print(f"⚠️  {len(self.errors)} errors occurred during collection:")
            for error in self.errors:
                print(f"   • {error}")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "summary": summary,
            "errors": self.errors
        }
    
    def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        print("📊 Collecting code quality metrics...")
        
        try:
            # Test coverage
            coverage = self._get_test_coverage()
            if coverage:
                self.metrics["test_coverage"] = MetricValue(
                    value=coverage,
                    unit="percent",
                    timestamp=datetime.utcnow(),
                    source="pytest-cov"
                )
            
            # Code complexity
            complexity = self._get_code_complexity()
            if complexity:
                self.metrics["code_complexity"] = MetricValue(
                    value=complexity,
                    unit="cyclomatic_complexity",
                    timestamp=datetime.utcnow(),
                    source="radon"
                )
            
            # Lines of code
            loc = self._get_lines_of_code()
            if loc:
                self.metrics["lines_of_code"] = MetricValue(
                    value=loc,
                    unit="lines",
                    timestamp=datetime.utcnow(),
                    source="cloc"
                )
                
        except Exception as e:
            self.errors.append(f"Code quality metrics: {e}")
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        print("🔒 Collecting security metrics...")
        
        try:
            # Security vulnerabilities
            vulns = self._get_security_vulnerabilities()
            for severity, count in vulns.items():
                self.metrics[f"vulnerabilities_{severity}"] = MetricValue(
                    value=count,
                    unit="count",
                    timestamp=datetime.utcnow(),
                    source="bandit",
                    tags={"severity": severity}
                )
            
            # Dependency vulnerabilities
            dep_vulns = self._get_dependency_vulnerabilities()
            if dep_vulns is not None:
                self.metrics["dependency_vulnerabilities"] = MetricValue(
                    value=dep_vulns,
                    unit="count",
                    timestamp=datetime.utcnow(),
                    source="safety"
                )
                
        except Exception as e:
            self.errors.append(f"Security metrics: {e}")
    
    def collect_performance_metrics(self):
        """Collect performance metrics."""
        print("⚡ Collecting performance metrics...")
        
        try:
            # Run performance benchmarks if available
            if Path("tests/performance").exists():
                perf_results = self._run_performance_tests()
                for test_name, duration in perf_results.items():
                    self.metrics[f"perf_{test_name}"] = MetricValue(
                        value=duration,
                        unit="seconds",
                        timestamp=datetime.utcnow(),
                        source="pytest-benchmark"
                    )
            
            # Memory usage analysis
            memory_usage = self._analyze_memory_usage()
            if memory_usage:
                self.metrics["memory_usage"] = MetricValue(
                    value=memory_usage,
                    unit="MB",
                    timestamp=datetime.utcnow(),
                    source="memory_profiler"
                )
                
        except Exception as e:
            self.errors.append(f"Performance metrics: {e}")
    
    def collect_development_metrics(self):
        """Collect development process metrics."""
        print("👥 Collecting development metrics...")
        
        try:
            # Git metrics
            git_metrics = self._get_git_metrics()
            for metric_name, value in git_metrics.items():
                self.metrics[f"git_{metric_name}"] = MetricValue(
                    value=value,
                    unit="count" if "count" in metric_name else "days",
                    timestamp=datetime.utcnow(),
                    source="git"
                )
            
            # GitHub metrics (if GitHub API token available)
            if os.getenv("GITHUB_TOKEN"):
                gh_metrics = self._get_github_metrics()
                for metric_name, value in gh_metrics.items():
                    self.metrics[f"github_{metric_name}"] = MetricValue(
                        value=value,
                        unit="count",
                        timestamp=datetime.utcnow(),
                        source="github_api"
                    )
                    
        except Exception as e:
            self.errors.append(f"Development metrics: {e}")
    
    def collect_business_metrics(self):
        """Collect business value metrics."""
        print("💼 Collecting business metrics...")
        
        try:
            # Package download stats (if published)
            if self._is_package_published():
                downloads = self._get_package_downloads()
                if downloads:
                    self.metrics["package_downloads"] = MetricValue(
                        value=downloads,
                        unit="downloads_per_month",
                        timestamp=datetime.utcnow(),
                        source="pypi_stats"
                    )
            
            # Docker image pulls
            docker_pulls = self._get_docker_pulls()
            if docker_pulls:
                self.metrics["docker_pulls"] = MetricValue(
                    value=docker_pulls,
                    unit="pulls_per_month",
                    timestamp=datetime.utcnow(),
                    source="docker_hub"
                )
                
        except Exception as e:
            self.errors.append(f"Business metrics: {e}")
    
    def _get_test_coverage(self) -> Optional[float]:
        """Get test coverage percentage."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src/testgen_copilot", "--cov-report=json", "--quiet"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and Path("coverage.json").exists():
                with open("coverage.json", 'r') as f:
                    cov_data = json.load(f)
                return cov_data.get("totals", {}).get("percent_covered", 0)
                
        except Exception as e:
            print(f"Warning: Could not get test coverage: {e}")
        return None
    
    def _get_code_complexity(self) -> Optional[float]:
        """Get average code complexity."""
        try:
            result = subprocess.run(
                ["python", "-m", "radon", "cc", "src/", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                complexities = []
                for file_data in complexity_data.values():
                    for item in file_data:
                        if isinstance(item, dict) and "complexity" in item:
                            complexities.append(item["complexity"])
                
                return sum(complexities) / len(complexities) if complexities else 0
                
        except Exception as e:
            print(f"Warning: Could not get code complexity: {e}")
        return None
    
    def _get_lines_of_code(self) -> Optional[int]:
        """Get total lines of code."""
        try:
            result = subprocess.run(
                ["find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_line = lines[-1] if lines else "0 total"
                return int(total_line.split()[0])
                
        except Exception as e:
            print(f"Warning: Could not get lines of code: {e}")
        return None
    
    def _get_security_vulnerabilities(self) -> Dict[str, int]:
        """Get security vulnerability counts by severity."""
        vulnerabilities = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        try:
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get("results", []):
                    severity = issue.get("issue_severity", "").lower()
                    if severity in vulnerabilities:
                        vulnerabilities[severity] += 1
                        
        except Exception as e:
            print(f"Warning: Could not get security vulnerabilities: {e}")
        
        return vulnerabilities
    
    def _get_dependency_vulnerabilities(self) -> Optional[int]:
        """Get dependency vulnerability count."""
        try:
            result = subprocess.run(
                ["python", "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                return len(safety_data) if isinstance(safety_data, list) else 0
                
        except Exception as e:
            print(f"Warning: Could not check dependency vulnerabilities: {e}")
        return None
    
    def _run_performance_tests(self) -> Dict[str, float]:
        """Run performance benchmarks."""
        perf_results = {}
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/performance/", "--benchmark-json=benchmark.json", "--quiet"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and Path("benchmark.json").exists():
                with open("benchmark.json", 'r') as f:
                    benchmark_data = json.load(f)
                
                for benchmark in benchmark_data.get("benchmarks", []):
                    name = benchmark.get("name", "unknown")
                    stats = benchmark.get("stats", {})
                    mean = stats.get("mean", 0)
                    perf_results[name] = mean
                    
        except Exception as e:
            print(f"Warning: Could not run performance tests: {e}")
        
        return perf_results
    
    def _analyze_memory_usage(self) -> Optional[float]:
        """Analyze memory usage of key operations."""
        try:
            # This is a simplified example - in practice, you'd run specific memory profiling
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
            
        except Exception as e:
            print(f"Warning: Could not analyze memory usage: {e}")
        return None
    
    def _get_git_metrics(self) -> Dict[str, int]:
        """Get Git repository metrics."""
        metrics = {}
        
        try:
            # Commit count (last 30 days)
            result = subprocess.run(
                ["git", "rev-list", "--count", f"--since=30 days ago", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                metrics["commits_30d"] = int(result.stdout.strip())
            
            # Contributors count
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                contributors = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                metrics["contributors_total"] = contributors
            
            # Files changed (last 30 days)
            result = subprocess.run(
                ["git", "diff", "--name-only", f"HEAD@{{30 days ago}}", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                files_changed = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                metrics["files_changed_30d"] = files_changed
                
        except Exception as e:
            print(f"Warning: Could not get Git metrics: {e}")
        
        return metrics
    
    def _get_github_metrics(self) -> Dict[str, int]:
        """Get GitHub repository metrics via API."""
        metrics = {}
        
        try:
            token = os.getenv("GITHUB_TOKEN")
            repo = self.config.get("project", {}).get("repository", "")
            
            if not token or not repo:
                return metrics
            
            headers = {"Authorization": f"token {token}"}
            base_url = f"https://api.github.com/repos/{repo}"
            
            # Repository stats
            response = requests.get(base_url, headers=headers, timeout=10)
            if response.status_code == 200:
                repo_data = response.json()
                metrics["stars"] = repo_data.get("stargazers_count", 0)
                metrics["forks"] = repo_data.get("forks_count", 0)
                metrics["open_issues"] = repo_data.get("open_issues_count", 0)
            
            # Pull requests (last 30 days)
            since = (datetime.utcnow() - timedelta(days=30)).isoformat()
            pr_url = f"{base_url}/pulls?state=all&since={since}"
            response = requests.get(pr_url, headers=headers, timeout=10)
            if response.status_code == 200:
                prs = response.json()
                metrics["prs_30d"] = len(prs)
                
        except Exception as e:
            print(f"Warning: Could not get GitHub metrics: {e}")
        
        return metrics
    
    def _is_package_published(self) -> bool:
        """Check if package is published to PyPI."""
        try:
            package_name = self.config.get("project", {}).get("name", "")
            if not package_name:
                return False
            
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            return response.status_code == 200
            
        except Exception:
            return False
    
    def _get_package_downloads(self) -> Optional[int]:
        """Get package download statistics."""
        try:
            package_name = self.config.get("project", {}).get("name", "")
            if not package_name:
                return None
            
            # Using pypistats API for download counts
            response = requests.get(
                f"https://pypistats.org/api/packages/{package_name}/recent",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("last_month", 0)
                
        except Exception as e:
            print(f"Warning: Could not get package downloads: {e}")
        return None
    
    def _get_docker_pulls(self) -> Optional[int]:
        """Get Docker image pull statistics."""
        try:
            # This would require Docker Hub API integration
            # Placeholder implementation
            return None
            
        except Exception as e:
            print(f"Warning: Could not get Docker pulls: {e}")
        return None
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate metrics summary."""
        summary = {
            "total_metrics": len(self.metrics),
            "collection_timestamp": datetime.utcnow().isoformat(),
            "status": "success" if not self.errors else "partial"
        }
        
        # Add key metric highlights
        if "test_coverage" in self.metrics:
            summary["test_coverage"] = self.metrics["test_coverage"].value
        
        if "vulnerabilities_critical" in self.metrics:
            summary["critical_vulnerabilities"] = self.metrics["vulnerabilities_critical"].value
        
        if "github_stars" in self.metrics:
            summary["github_stars"] = self.metrics["github_stars"].value
        
        return summary
    
    def save_metrics(self, output_path: str = "metrics_report.json"):
        """Save collected metrics to file."""
        report = self.collect_all_metrics()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Metrics report saved to {output_path}")
        return report

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect TestGen-Copilot project metrics")
    parser.add_argument("--config", help="Path to metrics configuration file")
    parser.add_argument("--output", default="metrics_report.json", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(config_path=args.config)
    
    try:
        report = collector.save_metrics(output_path=args.output)
        
        if args.verbose:
            print("\n📊 Metrics Summary:")
            print(f"   Total metrics collected: {report['summary']['total_metrics']}")
            print(f"   Collection status: {report['summary']['status']}")
            
            if "test_coverage" in report['summary']:
                print(f"   Test coverage: {report['summary']['test_coverage']:.1f}%")
            
            if "critical_vulnerabilities" in report['summary']:
                print(f"   Critical vulnerabilities: {report['summary']['critical_vulnerabilities']}")
        
        print("✅ Metrics collection completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error during metrics collection: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())