#!/usr/bin/env python3
"""
Automated Quality Gates for TestGen-Copilot

This script implements comprehensive quality gates that can be run in CI/CD
or locally to ensure code quality, security, and performance standards.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse


@dataclass
class QualityGateResult:
    """Represents the result of a quality gate check."""
    name: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime


class QualityGateRunner:
    """Orchestrates all quality gate checks."""
    
    def __init__(self, config_path: str = ".github/quality-gates.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load quality gates configuration."""
        default_config = {
            "code_quality": {
                "enabled": True,
                "ruff_max_errors": 0,
                "black_check": True,
                "mypy_check": True,
                "complexity_threshold": 10
            },
            "testing": {
                "enabled": True,
                "coverage_threshold": 80.0,
                "test_pass_rate": 95.0,
                "performance_regression_threshold": 10.0
            },
            "security": {
                "enabled": True,
                "bandit_severity_threshold": "medium",
                "safety_vulnerabilities": 0,
                "secrets_detected": 0
            },
            "dependencies": {
                "enabled": True,
                "outdated_threshold": 10,
                "security_vulnerabilities": 0,
                "license_compliance": True
            },
            "documentation": {
                "enabled": True,
                "docstring_coverage": 80.0,
                "broken_links": 0,
                "spelling_errors": 5
            },
            "performance": {
                "enabled": True,
                "build_time_threshold": 300,  # 5 minutes
                "memory_usage_threshold": 1024,  # 1GB
                "startup_time_threshold": 10.0  # 10 seconds
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
    
    def run_command(self, command: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
    
    def check_code_quality(self) -> QualityGateResult:
        """Check code quality metrics."""
        start_time = time.time()
        config = self.config.get("code_quality", {})
        
        if not config.get("enabled", True):
            return QualityGateResult(
                name="code_quality",
                passed=True,
                score=100.0,
                threshold=0.0,
                details={"skipped": True},
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        details = {}
        score = 100.0
        passed = True
        
        # Run ruff linting
        print("  Running ruff linting...")
        exit_code, stdout, stderr = self.run_command(["ruff", "check", "src/", "tests/", "--output-format=json"])
        
        if exit_code == 0:
            ruff_issues = 0
        else:
            try:
                ruff_output = json.loads(stdout) if stdout else []
                ruff_issues = len(ruff_output)
                details["ruff_issues"] = ruff_output
            except:
                ruff_issues = 1
        
        details["ruff_errors"] = ruff_issues
        if ruff_issues > config.get("ruff_max_errors", 0):
            passed = False
            score -= min(50, ruff_issues * 5)
        
        # Run black formatting check
        if config.get("black_check", True):
            print("  Checking black formatting...")
            exit_code, stdout, stderr = self.run_command(["black", "--check", "--diff", "src/", "tests/"])
            details["black_formatted"] = exit_code == 0
            if exit_code != 0:
                passed = False
                score -= 20
        
        # Run mypy type checking
        if config.get("mypy_check", True):
            print("  Running mypy type checking...")
            exit_code, stdout, stderr = self.run_command(["mypy", "src/testgen_copilot"])
            mypy_errors = stderr.count("error:") if stderr else 0
            details["mypy_errors"] = mypy_errors
            if mypy_errors > 0:
                passed = False
                score -= min(30, mypy_errors * 3)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="code_quality",
            passed=passed,
            score=max(0, score),
            threshold=90.0,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def check_testing(self) -> QualityGateResult:
        """Check testing metrics."""
        start_time = time.time()
        config = self.config.get("testing", {})
        
        if not config.get("enabled", True):
            return QualityGateResult(
                name="testing",
                passed=True,
                score=100.0,
                threshold=0.0,
                details={"skipped": True},
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        details = {}
        score = 100.0
        passed = True
        
        # Run tests with coverage
        print("  Running test suite with coverage...")
        exit_code, stdout, stderr = self.run_command([
            "pytest", "tests/", "--cov=src/testgen_copilot", 
            "--cov-report=json", "--cov-report=term-missing",
            "--junitxml=test-results.xml", "-v"
        ])
        
        # Parse coverage report
        try:
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)
                coverage_percent = coverage_data["totals"]["percent_covered"]
                details["coverage_percent"] = coverage_percent
                details["lines_covered"] = coverage_data["totals"]["covered_lines"]
                details["lines_total"] = coverage_data["totals"]["num_statements"]
                
                threshold = config.get("coverage_threshold", 80.0)
                if coverage_percent < threshold:
                    passed = False
                    score -= (threshold - coverage_percent) * 2
        except Exception as e:
            details["coverage_error"] = str(e)
            passed = False
            score -= 50
        
        # Parse test results
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse("test-results.xml")
            root = tree.getroot()
            
            tests_run = int(root.attrib.get("tests", 0))
            failures = int(root.attrib.get("failures", 0))
            errors = int(root.attrib.get("errors", 0))
            
            pass_rate = ((tests_run - failures - errors) / tests_run * 100) if tests_run > 0 else 0
            
            details["tests_run"] = tests_run
            details["test_failures"] = failures
            details["test_errors"] = errors
            details["test_pass_rate"] = pass_rate
            
            threshold = config.get("test_pass_rate", 95.0)
            if pass_rate < threshold:
                passed = False
                score -= (threshold - pass_rate) * 2
                
        except Exception as e:
            details["test_parsing_error"] = str(e)
            if exit_code != 0:
                passed = False
                score -= 30
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="testing",
            passed=passed,
            score=max(0, score),
            threshold=config.get("coverage_threshold", 80.0),
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def check_security(self) -> QualityGateResult:
        """Check security metrics."""
        start_time = time.time()
        config = self.config.get("security", {})
        
        if not config.get("enabled", True):
            return QualityGateResult(
                name="security",
                passed=True,
                score=100.0,
                threshold=0.0,
                details={"skipped": True},
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        details = {}
        score = 100.0
        passed = True
        
        # Run bandit security scan
        print("  Running bandit security scan...")
        exit_code, stdout, stderr = self.run_command([
            "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"
        ])
        
        try:
            with open("bandit-report.json", "r") as f:
                bandit_data = json.load(f)
                
                high_issues = len([r for r in bandit_data.get("results", []) 
                                 if r.get("issue_severity") == "HIGH"])
                medium_issues = len([r for r in bandit_data.get("results", []) 
                                   if r.get("issue_severity") == "MEDIUM"])
                low_issues = len([r for r in bandit_data.get("results", []) 
                                if r.get("issue_severity") == "LOW"])
                
                details["bandit_high"] = high_issues
                details["bandit_medium"] = medium_issues
                details["bandit_low"] = low_issues
                
                # Penalize based on severity
                if high_issues > 0:
                    passed = False
                    score -= high_issues * 30
                
                threshold = config.get("bandit_severity_threshold", "medium")
                if threshold == "medium" and medium_issues > 0:
                    passed = False
                    score -= medium_issues * 15
                elif threshold == "low" and low_issues > 0:
                    passed = False
                    score -= low_issues * 5
                    
        except Exception as e:
            details["bandit_error"] = str(e)
        
        # Run safety vulnerability check
        print("  Running safety vulnerability check...")
        exit_code, stdout, stderr = self.run_command([
            "safety", "check", "--json", "--output", "safety-report.json"
        ])
        
        try:
            with open("safety-report.json", "r") as f:
                safety_data = json.load(f)
                vulnerabilities = len(safety_data) if isinstance(safety_data, list) else 0
                details["safety_vulnerabilities"] = vulnerabilities
                
                threshold = config.get("safety_vulnerabilities", 0)
                if vulnerabilities > threshold:
                    passed = False
                    score -= vulnerabilities * 20
                    
        except Exception as e:
            details["safety_error"] = str(e)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="security",
            passed=passed,
            score=max(0, score),
            threshold=100.0,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def check_dependencies(self) -> QualityGateResult:
        """Check dependency health."""
        start_time = time.time()
        config = self.config.get("dependencies", {})
        
        if not config.get("enabled", True):
            return QualityGateResult(
                name="dependencies",
                passed=True,
                score=100.0,
                threshold=0.0,
                details={"skipped": True},
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        details = {}
        score = 100.0
        passed = True
        
        # Check for outdated packages
        print("  Checking for outdated packages...")
        exit_code, stdout, stderr = self.run_command(["pip", "list", "--outdated", "--format=json"])
        
        try:
            if stdout:
                outdated_packages = json.loads(stdout)
                outdated_count = len(outdated_packages)
                details["outdated_packages"] = outdated_count
                details["outdated_list"] = outdated_packages
                
                threshold = config.get("outdated_threshold", 10)
                if outdated_count > threshold:
                    passed = False
                    score -= min(50, (outdated_count - threshold) * 5)
        except Exception as e:
            details["outdated_error"] = str(e)
        
        # Check dependency tree
        print("  Checking dependency tree...")
        exit_code, stdout, stderr = self.run_command(["pip", "check"])
        
        if exit_code != 0:
            details["dependency_conflicts"] = stderr
            passed = False
            score -= 30
        else:
            details["dependency_conflicts"] = None
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="dependencies",
            passed=passed,
            score=max(0, score),
            threshold=90.0,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def check_performance(self) -> QualityGateResult:
        """Check performance metrics."""
        start_time = time.time()
        config = self.config.get("performance", {})
        
        if not config.get("enabled", True):
            return QualityGateResult(
                name="performance",
                passed=True,
                score=100.0,
                threshold=0.0,
                details={"skipped": True},
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
        details = {}
        score = 100.0
        passed = True
        
        # Measure build time
        print("  Measuring build performance...")
        build_start = time.time()
        exit_code, stdout, stderr = self.run_command(["python", "-m", "build"])
        build_time = time.time() - build_start
        
        details["build_time"] = build_time
        details["build_success"] = exit_code == 0
        
        threshold = config.get("build_time_threshold", 300)
        if build_time > threshold:
            passed = False
            score -= min(30, (build_time - threshold) / 10)
        
        if exit_code != 0:
            passed = False
            score -= 40
        
        # Run performance benchmarks if available
        if Path("tests/performance").exists():
            print("  Running performance benchmarks...")
            bench_start = time.time()
            exit_code, stdout, stderr = self.run_command([
                "pytest", "tests/performance/", "--benchmark-only", 
                "--benchmark-json=benchmark.json"
            ])
            bench_time = time.time() - bench_start
            
            details["benchmark_time"] = bench_time
            details["benchmark_success"] = exit_code == 0
            
            if exit_code != 0:
                score -= 20
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="performance",
            passed=passed,
            score=max(0, score),
            threshold=80.0,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def run_all_gates(self) -> List[QualityGateResult]:
        """Run all quality gates."""
        print("🚦 Running Quality Gates for TestGen-Copilot")
        print("=" * 50)
        
        gates = [
            ("Code Quality", self.check_code_quality),
            ("Testing", self.check_testing),
            ("Security", self.check_security),
            ("Dependencies", self.check_dependencies),
            ("Performance", self.check_performance),
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Running {gate_name} checks...")
            result = gate_func()
            self.results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} - Score: {result.score:.1f}/{result.threshold:.1f}")
            print(f"  Execution time: {result.execution_time:.2f}s")
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality gate report."""
        total_execution_time = time.time() - self.start_time
        
        passed_gates = sum(1 for r in self.results if r.passed)
        total_gates = len(self.results)
        overall_passed = passed_gates == total_gates
        
        average_score = sum(r.score for r in self.results) / len(self.results) if self.results else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_passed": overall_passed,
            "gates_passed": passed_gates,
            "gates_total": total_gates,
            "average_score": average_score,
            "execution_time": total_execution_time,
            "results": [asdict(r) for r in self.results],
            "summary": {
                "code_quality": next((r.passed for r in self.results if r.name == "code_quality"), None),
                "testing": next((r.passed for r in self.results if r.name == "testing"), None),
                "security": next((r.passed for r in self.results if r.name == "security"), None),
                "dependencies": next((r.passed for r in self.results if r.name == "dependencies"), None),
                "performance": next((r.passed for r in self.results if r.name == "performance"), None),
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = "quality-gates-report.json"):
        """Save report to file."""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📊 Report saved to {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print summary to console."""
        print("\n" + "=" * 50)
        print("🚦 QUALITY GATES SUMMARY")
        print("=" * 50)
        
        overall_status = "✅ ALL PASSED" if report["overall_passed"] else "❌ SOME FAILED"
        print(f"Overall Status: {overall_status}")
        print(f"Gates Passed: {report['gates_passed']}/{report['gates_total']}")
        print(f"Average Score: {report['average_score']:.1f}/100")
        print(f"Total Time: {report['execution_time']:.2f}s")
        
        print("\nDetailed Results:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name.replace('_', ' ').title()}: {result.score:.1f}/{result.threshold:.1f}")
        
        if not report["overall_passed"]:
            print("\n⚠️  Please fix the failing quality gates before proceeding.")
            return False
        else:
            print("\n🎉 All quality gates passed! Ready for deployment.")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run quality gates for TestGen-Copilot")
    parser.add_argument("--config", help="Path to quality gates configuration file")
    parser.add_argument("--output", help="Output file for report", default="quality-gates-report.json")
    parser.add_argument("--fail-fast", action="store_true", help="Exit on first failure")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    runner = QualityGateRunner(config_path=args.config or ".github/quality-gates.json")
    
    try:
        results = runner.run_all_gates()
        report = runner.generate_report()
        runner.save_report(report, args.output)
        
        if not args.quiet:
            success = runner.print_summary(report)
        else:
            success = report["overall_passed"]
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Quality gate execution interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error running quality gates: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()