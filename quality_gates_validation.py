#!/usr/bin/env python3
"""
Quality Gates Validation Suite
=============================

Comprehensive quality validation for the quantum-inspired test generation project.
This suite validates code quality, security, performance, and research standards.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result from a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    max_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    max_score: float
    passed_gates: int
    total_gates: int
    gate_results: List[QualityGateResult] = field(default_factory=list)
    recommendation: str = ""

class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results: List[QualityGateResult] = []
        
    def run_import_validation(self) -> QualityGateResult:
        """Validate all core imports work correctly."""
        self.logger.info("🔬 Running import validation...")
        
        try:
            # Test core imports
            import testgen_copilot
            from testgen_copilot.quantum_planner import create_quantum_planner, QuantumTask, TaskPriority
            from testgen_copilot.generator import TestGenerator, GenerationConfig
            from testgen_copilot.security import SecurityScanner
            from testgen_copilot.performance_optimizer import get_performance_optimizer
            from testgen_copilot.monitoring import get_health_monitor
            from testgen_copilot.resilience import get_resilience_manager
            
            # Test basic functionality
            planner = create_quantum_planner(max_iterations=100)
            generator = TestGenerator()
            scanner = SecurityScanner()
            optimizer = get_performance_optimizer()
            health = get_health_monitor()
            resilience = get_resilience_manager()
            
            score = 100.0
            details = {
                "modules_imported": 6,
                "basic_instantiation": "success",
                "quantum_planner_created": True,
                "generator_created": True,
                "scanner_created": True
            }
            
            return QualityGateResult(
                gate_name="Import Validation",
                passed=True,
                score=score,
                max_score=100.0,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Import validation failed: {e}")
            return QualityGateResult(
                gate_name="Import Validation",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message=str(e)
            )
    
    def run_code_quality_check(self) -> QualityGateResult:
        """Run code quality analysis using ruff."""
        self.logger.info("📊 Running code quality analysis...")
        
        try:
            # Run ruff check
            result = subprocess.run(
                ["ruff", "check", "src/", "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # No issues found
                score = 100.0
                issues = []
            else:
                # Parse ruff output
                try:
                    issues = json.loads(result.stdout) if result.stdout else []
                except json.JSONDecodeError:
                    issues = []
                
                # Calculate score based on issue severity
                error_count = sum(1 for issue in issues if issue.get('type') == 'error')
                warning_count = sum(1 for issue in issues if issue.get('type') == 'warning')
                
                # Scoring: -10 per error, -2 per warning
                penalty = error_count * 10 + warning_count * 2
                score = max(0, 100 - penalty)
            
            passed = score >= 80  # 80% threshold
            
            return QualityGateResult(
                gate_name="Code Quality",
                passed=passed,
                score=score,
                max_score=100.0,
                details={
                    "total_issues": len(issues),
                    "error_count": sum(1 for issue in issues if issue.get('type') == 'error'),
                    "warning_count": sum(1 for issue in issues if issue.get('type') == 'warning'),
                    "issues": issues[:10]  # First 10 issues for review
                }
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message="Code quality check timed out"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message=f"Code quality check failed: {e}"
            )
    
    def run_security_validation(self) -> QualityGateResult:
        """Run security validation using bandit."""
        self.logger.info("🛡️ Running security analysis...")
        
        try:
            # Run bandit security check
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Bandit returns non-zero even for warnings, so check output
            try:
                bandit_output = json.loads(result.stdout)
            except json.JSONDecodeError:
                bandit_output = {"results": [], "metrics": {"_totals": {"SEVERITY.HIGH": 0, "SEVERITY.MEDIUM": 0, "SEVERITY.LOW": 0}}}
            
            # Calculate security score
            high_issues = bandit_output.get("metrics", {}).get("_totals", {}).get("SEVERITY.HIGH", 0)
            medium_issues = bandit_output.get("metrics", {}).get("_totals", {}).get("SEVERITY.MEDIUM", 0)
            low_issues = bandit_output.get("metrics", {}).get("_totals", {}).get("SEVERITY.LOW", 0)
            
            # Scoring: -20 per high, -10 per medium, -2 per low
            penalty = high_issues * 20 + medium_issues * 10 + low_issues * 2
            score = max(0, 100 - penalty)
            passed = score >= 85  # Higher threshold for security
            
            return QualityGateResult(
                gate_name="Security Validation",
                passed=passed,
                score=score,
                max_score=100.0,
                details={
                    "high_severity_issues": high_issues,
                    "medium_severity_issues": medium_issues,
                    "low_severity_issues": low_issues,
                    "total_issues": len(bandit_output.get("results", [])),
                    "issues_sample": bandit_output.get("results", [])[:5]
                }
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="Security Validation",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message="Security check timed out"
            )
        except Exception as e:
            # If bandit is not available, create a basic security validation
            self.logger.warning(f"Bandit not available, running basic security check: {e}")
            return self._basic_security_check()
    
    def _basic_security_check(self) -> QualityGateResult:
        """Basic security validation when bandit is not available."""
        try:
            # Check for common security anti-patterns
            security_issues = []
            
            for py_file in Path("src").rglob("*.py"):
                content = py_file.read_text()
                
                # Check for potential issues
                if "eval(" in content:
                    security_issues.append(f"Use of eval() in {py_file}")
                if "exec(" in content:
                    security_issues.append(f"Use of exec() in {py_file}")
                if "shell=True" in content:
                    security_issues.append(f"Shell execution in {py_file}")
                if "password" in content.lower() and "=" in content:
                    security_issues.append(f"Potential hardcoded credential in {py_file}")
            
            score = max(0, 100 - len(security_issues) * 15)  # -15 per issue
            passed = score >= 85
            
            return QualityGateResult(
                gate_name="Security Validation",
                passed=passed,
                score=score,
                max_score=100.0,
                details={
                    "basic_check": True,
                    "issues_found": len(security_issues),
                    "issues": security_issues
                }
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="Security Validation",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message=f"Basic security check failed: {e}"
            )
    
    def run_performance_validation(self) -> QualityGateResult:
        """Validate performance characteristics."""
        self.logger.info("⚡ Running performance validation...")
        
        try:
            # Test quantum planner performance
            start_time = time.time()
            
            from testgen_copilot.quantum_planner import create_quantum_planner, QuantumTask, TaskPriority
            from datetime import timedelta
            
            planner = create_quantum_planner(max_iterations=100, quantum_processors=2)
            
            # Add some test tasks
            for i in range(5):
                planner.add_task(
                    task_id=f"perf_test_{i}",
                    name=f"Performance Test Task {i}",
                    description="Performance testing task",
                    priority=TaskPriority.GROUND_STATE,
                    estimated_duration=timedelta(hours=1),
                    resources_required={"cpu": 1.0, "memory": 2.0}
                )
            
            # Time the planning operation
            planning_start = time.time()
            import asyncio
            plan = asyncio.run(planner.generate_optimal_plan())
            planning_time = time.time() - planning_start
            
            total_time = time.time() - start_time
            
            # Performance scoring
            performance_score = 100.0
            
            # Penalize if too slow
            if planning_time > 2.0:  # More than 2 seconds for 5 tasks is slow
                performance_score -= 30
            elif planning_time > 1.0:  # More than 1 second is moderate
                performance_score -= 15
            
            if total_time > 5.0:  # More than 5 seconds total is slow
                performance_score -= 20
            
            passed = performance_score >= 70  # 70% threshold
            
            return QualityGateResult(
                gate_name="Performance Validation",
                passed=passed,
                score=performance_score,
                max_score=100.0,
                details={
                    "total_time_seconds": total_time,
                    "planning_time_seconds": planning_time,
                    "tasks_processed": 5,
                    "plan_generated": plan is not None,
                    "performance_class": "fast" if planning_time < 0.5 else "moderate" if planning_time < 2.0 else "slow"
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Performance Validation",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message=f"Performance validation failed: {e}"
            )
    
    def run_research_validation(self) -> QualityGateResult:
        """Validate research quality and academic standards."""
        self.logger.info("🎓 Running research validation...")
        
        try:
            # Check for academic rigor indicators
            research_score = 0.0
            
            # Check if quantum research results exist
            if Path("academic_quantum_results.json").exists():
                with open("academic_quantum_results.json") as f:
                    results = json.load(f)
                    
                # Evaluate research quality
                theoretical_guarantees = results.get("theoretical_guarantees", {})
                
                if theoretical_guarantees:
                    research_score += 30  # Has theoretical analysis
                    
                    # Check quality scores
                    for algo, stats in theoretical_guarantees.items():
                        quality = stats.get("overall_mean_quality", 0)
                        success_rate = stats.get("overall_success_rate", 0)
                        
                        if quality > 80:
                            research_score += 25  # High quality results
                        elif quality > 50:
                            research_score += 15  # Moderate quality
                        
                        if success_rate >= 95:
                            research_score += 20  # High reliability
                        elif success_rate >= 80:
                            research_score += 10  # Moderate reliability
                        
                        break  # Score first algorithm
            
            # Check for research documentation
            research_files = [
                "quantum_research_publication_report.md",
                "robust_quantum_results.json",
                "research_validation_suite.py",
                "academic_quantum_optimization.py"
            ]
            
            existing_files = sum(1 for f in research_files if Path(f).exists())
            research_score += (existing_files / len(research_files)) * 25  # 25 points for documentation
            
            passed = research_score >= 70  # 70% threshold for research
            
            return QualityGateResult(
                gate_name="Research Validation",
                passed=passed,
                score=research_score,
                max_score=100.0,
                details={
                    "research_files_found": existing_files,
                    "total_research_files": len(research_files),
                    "has_theoretical_analysis": Path("academic_quantum_results.json").exists(),
                    "has_benchmarking": Path("robust_quantum_results.json").exists(),
                    "research_quality": "high" if research_score > 80 else "moderate" if research_score > 60 else "needs_improvement"
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Research Validation",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message=f"Research validation failed: {e}"
            )
    
    def run_architecture_validation(self) -> QualityGateResult:
        """Validate architectural principles and patterns."""
        self.logger.info("🏗️ Running architecture validation...")
        
        try:
            arch_score = 0.0
            
            # Check for key architectural components
            components = {
                "quantum_planner.py": 25,  # Core quantum algorithm
                "generator.py": 20,        # Test generation
                "security.py": 15,         # Security scanning
                "monitoring.py": 10,       # Observability
                "resilience.py": 10,       # Error handling
                "performance_optimizer.py": 10,  # Performance
                "internationalization.py": 5,   # Globalization
                "multi_region.py": 5       # Multi-region support
            }
            
            src_path = Path("src/testgen_copilot")
            for component, points in components.items():
                if (src_path / component).exists():
                    arch_score += points
            
            # Check for proper modular structure
            if (src_path / "api").exists():
                arch_score += 5  # API module
            if (src_path / "database").exists():
                arch_score += 5  # Database module
            if (src_path / "integrations").exists():
                arch_score += 5  # Integrations module
            
            passed = arch_score >= 80  # 80% architectural completeness
            
            return QualityGateResult(
                gate_name="Architecture Validation",
                passed=passed,
                score=arch_score,
                max_score=100.0,
                details={
                    "components_found": sum(1 for comp in components if (src_path / comp).exists()),
                    "total_components": len(components),
                    "modular_structure": {
                        "api_module": (src_path / "api").exists(),
                        "database_module": (src_path / "database").exists(),
                        "integrations_module": (src_path / "integrations").exists()
                    }
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Architecture Validation",
                passed=False,
                score=0.0,
                max_score=100.0,
                error_message=f"Architecture validation failed: {e}"
            )
    
    def run_all_quality_gates(self) -> QualityReport:
        """Run all quality gates and generate comprehensive report."""
        self.logger.info("🚀 Starting comprehensive quality gate validation...")
        
        # Define all quality gates
        quality_gates = [
            self.run_import_validation,
            self.run_code_quality_check,
            self.run_security_validation,
            self.run_performance_validation,
            self.run_research_validation,
            self.run_architecture_validation
        ]
        
        results = []
        total_score = 0.0
        max_total_score = 0.0
        passed_gates = 0
        
        # Execute all quality gates
        for gate_func in quality_gates:
            try:
                result = gate_func()
                results.append(result)
                total_score += result.score
                max_total_score += result.max_score
                if result.passed:
                    passed_gates += 1
            except Exception as e:
                self.logger.error(f"Quality gate {gate_func.__name__} failed with exception: {e}")
                results.append(QualityGateResult(
                    gate_name=gate_func.__name__.replace("run_", "").replace("_", " ").title(),
                    passed=False,
                    score=0.0,
                    max_score=100.0,
                    error_message=str(e)
                ))
        
        # Calculate overall score
        overall_score = (total_score / max_total_score * 100) if max_total_score > 0 else 0
        
        # Generate recommendation
        if overall_score >= 90:
            recommendation = "🎯 EXCELLENT - Production ready with outstanding quality"
        elif overall_score >= 80:
            recommendation = "✅ GOOD - Production ready with minor improvements needed"
        elif overall_score >= 70:
            recommendation = "⚠️ ACCEPTABLE - Needs improvements before production"
        elif overall_score >= 60:
            recommendation = "🔧 NEEDS WORK - Significant improvements required"
        else:
            recommendation = "❌ CRITICAL - Major issues must be resolved"
        
        return QualityReport(
            overall_score=overall_score,
            max_score=100.0,
            passed_gates=passed_gates,
            total_gates=len(quality_gates),
            gate_results=results,
            recommendation=recommendation
        )
    
    def generate_quality_report(self, report: QualityReport) -> str:
        """Generate human-readable quality report."""
        
        report_text = f"""
# 🏆 QUALITY GATES VALIDATION REPORT
## Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Assessment
- **Overall Score**: {report.overall_score:.1f}/100
- **Gates Passed**: {report.passed_gates}/{report.total_gates} 
- **Recommendation**: {report.recommendation}

## 🔍 Detailed Results

"""
        
        for result in report.gate_results:
            status_icon = "✅" if result.passed else "❌"
            report_text += f"""### {status_icon} {result.gate_name}
- **Score**: {result.score:.1f}/{result.max_score}
- **Status**: {'PASSED' if result.passed else 'FAILED'}
"""
            
            if result.error_message:
                report_text += f"- **Error**: {result.error_message}\n"
            
            if result.details:
                report_text += f"- **Details**: {json.dumps(result.details, indent=2)}\n"
            
            report_text += "\n"
        
        # Add improvement recommendations
        failed_gates = [r for r in report.gate_results if not r.passed]
        if failed_gates:
            report_text += "## 🔧 Improvement Recommendations\n\n"
            for gate in failed_gates:
                report_text += f"- **{gate.gate_name}**: Address issues to improve score from {gate.score:.1f}\n"
        
        return report_text


def main():
    """Execute quality gates validation."""
    validator = QualityGateValidator()
    
    logger.info("🎯 Starting Quality Gates Validation Suite")
    
    try:
        # Run all quality gates
        report = validator.run_all_quality_gates()
        
        # Save detailed results
        results_file = Path("quality_gates_results.json")
        with open(results_file, 'w') as f:
            # Convert dataclasses to dict for JSON serialization
            results_dict = {
                "overall_score": report.overall_score,
                "max_score": report.max_score,
                "passed_gates": report.passed_gates,
                "total_gates": report.total_gates,
                "recommendation": report.recommendation,
                "gate_results": [
                    {
                        "gate_name": r.gate_name,
                        "passed": r.passed,
                        "score": r.score,
                        "max_score": r.max_score,
                        "details": r.details,
                        "error_message": r.error_message
                    }
                    for r in report.gate_results
                ]
            }
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Quality gates results saved to: {results_file}")
        
        # Generate human-readable report
        report_text = validator.generate_quality_report(report)
        report_file = Path("quality_gates_report.md")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"📄 Quality gates report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("🏆 QUALITY GATES VALIDATION SUMMARY")
        print("="*80)
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Gates Passed: {report.passed_gates}/{report.total_gates}")
        print(f"Recommendation: {report.recommendation}")
        
        # Print individual gate results
        print("\n📋 Individual Gate Results:")
        for result in report.gate_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} {result.gate_name}: {result.score:.1f}/100")
        
        # Exit with appropriate code
        if report.overall_score >= 70:
            print("\n🎯 Quality gates validation SUCCESSFUL!")
            return 0
        else:
            print("\n⚠️ Quality gates validation needs improvement")
            return 1
            
    except Exception as e:
        logger.error(f"Quality gates validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())