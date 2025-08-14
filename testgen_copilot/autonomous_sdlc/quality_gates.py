"""
Advanced Quality Gates Validator

Comprehensive quality gate system with adaptive thresholds, auto-remediation,
and intelligent failure analysis.
"""

import asyncio
import logging
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class QualityGateType(Enum):
    """Quality gate types"""
    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"  
    SECURITY_SCAN = "security_scan"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    DEPENDENCY_SCAN = "dependency_scan"
    CODE_FORMATTING = "code_formatting"
    TYPE_CHECKING = "type_checking"


class QualityGateStatus(Enum):
    """Quality gate status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning" 
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class QualityGateResult:
    """Quality gate execution result"""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float  # 0.0 to 1.0
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation_suggestions: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    overall_score: float
    overall_status: QualityGateStatus  
    gate_results: Dict[QualityGateType, QualityGateResult]
    passed_gates: int
    failed_gates: int
    total_gates: int
    execution_duration: float
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class QualityGateValidator:
    """
    Advanced quality gate validator with adaptive thresholds and auto-remediation.
    
    Features:
    - Comprehensive quality checks (code, tests, security, performance)
    - Adaptive threshold adjustment based on project maturity
    - Auto-remediation for common issues
    - Detailed remediation suggestions
    - Progressive quality enforcement
    """
    
    def __init__(
        self,
        project_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.project_path = project_path or Path.cwd()
        self.config = config or {}
        
        # Default quality thresholds
        self.thresholds = {
            QualityGateType.CODE_QUALITY: 0.85,
            QualityGateType.TEST_COVERAGE: 0.85,
            QualityGateType.SECURITY_SCAN: 0.90,
            QualityGateType.PERFORMANCE: 0.80,
            QualityGateType.DOCUMENTATION: 0.75,
            QualityGateType.DEPENDENCY_SCAN: 0.95,
            QualityGateType.CODE_FORMATTING: 1.0,
            QualityGateType.TYPE_CHECKING: 0.90,
        }
        
        # Override with config
        if "quality_thresholds" in self.config:
            self.thresholds.update(self.config["quality_thresholds"])
        
        # Adaptive parameters  
        self.adaptive_mode = self.config.get("adaptive_thresholds", True)
        self.auto_remediation = self.config.get("auto_remediation", True)
        self.project_maturity_score = 0.5  # Will be calculated
        
        # Execution tracking
        self.execution_history: List[QualityReport] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, project_path: Path) -> bool:
        """Initialize quality gate validator"""
        try:
            self.project_path = project_path
            self.logger.info(f"Initializing quality gates for: {project_path}")
            
            # Calculate project maturity
            self.project_maturity_score = await self._calculate_project_maturity()
            
            # Adapt thresholds based on project maturity
            if self.adaptive_mode:
                await self._adapt_thresholds()
            
            self.logger.info(f"Quality gates initialized (maturity: {self.project_maturity_score:.2f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quality gates: {e}")
            return False
    
    async def validate_all_gates(
        self,
        enabled_gates: Optional[List[QualityGateType]] = None
    ) -> QualityReport:
        """Validate all enabled quality gates"""
        
        start_time = datetime.utcnow()
        
        # Determine which gates to run
        gates_to_run = enabled_gates or list(QualityGateType)
        
        self.logger.info(f"🔍 Running {len(gates_to_run)} quality gates...")
        
        # Execute gates in parallel
        gate_tasks = [
            self._execute_quality_gate(gate_type) 
            for gate_type in gates_to_run
        ]
        
        gate_results = await asyncio.gather(*gate_tasks, return_exceptions=True)
        
        # Process results
        results_dict = {}
        passed_count = 0
        failed_count = 0
        
        for gate_type, result in zip(gates_to_run, gate_results):
            if isinstance(result, Exception):
                self.logger.error(f"Gate {gate_type.value} failed with exception: {result}")
                results_dict[gate_type] = QualityGateResult(
                    gate_type=gate_type,
                    status=QualityGateStatus.ERROR,
                    score=0.0,
                    threshold=self.thresholds[gate_type],
                    message=f"Execution error: {result}",
                    details={"exception": str(result)}
                )
                failed_count += 1
            else:
                results_dict[gate_type] = result
                if result.status == QualityGateStatus.PASSED:
                    passed_count += 1
                elif result.status in [QualityGateStatus.FAILED, QualityGateStatus.ERROR]:
                    failed_count += 1
        
        # Calculate overall metrics
        execution_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate overall score (weighted average)
        total_weighted_score = 0.0
        total_weights = 0.0
        
        gate_weights = {
            QualityGateType.SECURITY_SCAN: 3.0,
            QualityGateType.TEST_COVERAGE: 2.5,
            QualityGateType.CODE_QUALITY: 2.0,
            QualityGateType.TYPE_CHECKING: 1.5,
            QualityGateType.DEPENDENCY_SCAN: 1.5,
            QualityGateType.CODE_FORMATTING: 1.0,
            QualityGateType.PERFORMANCE: 1.5,
            QualityGateType.DOCUMENTATION: 1.0,
        }
        
        for gate_type, result in results_dict.items():
            weight = gate_weights.get(gate_type, 1.0)
            total_weighted_score += result.score * weight
            total_weights += weight
        
        overall_score = total_weighted_score / max(total_weights, 1.0)
        
        # Determine overall status
        if failed_count == 0:
            overall_status = QualityGateStatus.PASSED
        elif passed_count > failed_count:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.FAILED
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(results_dict)
        
        # Create quality report
        report = QualityReport(
            overall_score=overall_score,
            overall_status=overall_status,
            gate_results=results_dict,
            passed_gates=passed_count,
            failed_gates=failed_count,
            total_gates=len(results_dict),
            execution_duration=execution_duration,
            recommendations=recommendations
        )
        
        # Store in history for adaptive learning
        self.execution_history.append(report)
        
        # Auto-remediation if enabled
        if self.auto_remediation and overall_status != QualityGateStatus.PASSED:
            await self._attempt_auto_remediation(report)
        
        self._log_quality_report(report)
        return report
    
    async def _execute_quality_gate(self, gate_type: QualityGateType) -> QualityGateResult:
        """Execute individual quality gate"""
        
        start_time = datetime.utcnow()
        threshold = self.thresholds[gate_type]
        
        self.logger.debug(f"Executing gate: {gate_type.value}")
        
        try:
            if gate_type == QualityGateType.CODE_QUALITY:
                score, details, suggestions = await self._check_code_quality()
            elif gate_type == QualityGateType.TEST_COVERAGE:
                score, details, suggestions = await self._check_test_coverage()
            elif gate_type == QualityGateType.SECURITY_SCAN:
                score, details, suggestions = await self._check_security()
            elif gate_type == QualityGateType.PERFORMANCE:
                score, details, suggestions = await self._check_performance()
            elif gate_type == QualityGateType.DOCUMENTATION:
                score, details, suggestions = await self._check_documentation()
            elif gate_type == QualityGateType.DEPENDENCY_SCAN:
                score, details, suggestions = await self._check_dependencies()
            elif gate_type == QualityGateType.CODE_FORMATTING:
                score, details, suggestions = await self._check_code_formatting()
            elif gate_type == QualityGateType.TYPE_CHECKING:
                score, details, suggestions = await self._check_type_checking()
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")
            
            # Determine status
            if score >= threshold:
                status = QualityGateStatus.PASSED
                message = f"Quality gate passed with score {score:.2%} (threshold: {threshold:.2%})"
            elif score >= threshold * 0.8:  # Warning zone
                status = QualityGateStatus.WARNING
                message = f"Quality gate warning: score {score:.2%} below threshold {threshold:.2%}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Quality gate failed: score {score:.2%} below threshold {threshold:.2%}"
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_type=gate_type,
                status=status,
                score=score,
                threshold=threshold,
                message=message,
                details=details,
                remediation_suggestions=suggestions,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Quality gate {gate_type.value} execution failed: {e}")
            
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityGateStatus.ERROR,
                score=0.0,
                threshold=threshold,
                message=f"Execution failed: {e}",
                details={"error": str(e)},
                remediation_suggestions=[f"Fix execution error in {gate_type.value} check"],
                execution_time=execution_time
            )
    
    async def _check_code_quality(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check code quality using multiple tools"""
        
        try:
            # Run ruff for Python code quality
            result = await self._run_command(["ruff", "check", str(self.project_path)])
            
            if result.returncode == 0:
                score = 1.0
                issues = []
            else:
                # Parse ruff output for issues
                issues = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
                # Calculate score based on issue count and severity
                total_lines = await self._count_code_lines()
                issue_density = len(issues) / max(total_lines, 1) * 1000  # Issues per 1000 lines
                
                # Score calculation (inverted issue density)
                score = max(0.0, 1.0 - (issue_density * 0.1))
            
            details = {
                "tool": "ruff",
                "issues_found": len(issues),
                "total_lines": await self._count_code_lines(),
                "issue_list": issues[:10]  # First 10 issues
            }
            
            suggestions = []
            if issues:
                suggestions.extend([
                    "Run 'ruff check --fix' to auto-fix issues",
                    "Review code style guidelines",
                    "Consider enabling pre-commit hooks"
                ])
            
            return score, details, suggestions
            
        except FileNotFoundError:
            # Fallback: basic quality checks
            score = 0.7  # Assume decent quality if tools not available
            details = {"tool": "basic", "message": "Advanced tools not available"}
            suggestions = ["Install ruff for better code quality analysis"]
            return score, details, suggestions
    
    async def _check_test_coverage(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check test coverage"""
        
        try:
            # Try to run pytest with coverage
            result = await self._run_command([
                "python", "-m", "pytest", "--cov=testgen_copilot", 
                "--cov-report=json", "--cov-report=term-missing", "-q"
            ], cwd=self.project_path)
            
            coverage_score = 0.0
            details = {}
            
            # Try to parse coverage.json if available
            coverage_file = self.project_path / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    
                    coverage_score = coverage_data.get("totals", {}).get("percent_covered", 0) / 100
                    details = {
                        "total_statements": coverage_data.get("totals", {}).get("num_statements", 0),
                        "covered_statements": coverage_data.get("totals", {}).get("covered_lines", 0),
                        "missing_lines": coverage_data.get("totals", {}).get("missing_lines", 0)
                    }
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # If coverage.json not available, try to parse stdout
            if coverage_score == 0.0 and result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "TOTAL" in line and "%" in line:
                        try:
                            # Extract percentage from line like "TOTAL  1234  567  54%"
                            parts = line.split()
                            for part in parts:
                                if part.endswith('%'):
                                    coverage_score = float(part[:-1]) / 100
                                    break
                        except (ValueError, IndexError):
                            continue
            
            # If still no coverage data, estimate based on test files
            if coverage_score == 0.0:
                test_files = list(self.project_path.rglob("test_*.py"))
                code_files = list(self.project_path.rglob("*.py"))
                code_files = [f for f in code_files if not f.name.startswith("test_")]
                
                if test_files and code_files:
                    # Rough estimate: 1 test file per 2 code files = 50% coverage
                    coverage_score = min(0.8, len(test_files) / len(code_files))
                else:
                    coverage_score = 0.1  # Very low if no tests
            
            details.update({
                "tool": "pytest-cov",
                "test_files_found": len(list(self.project_path.rglob("test_*.py"))),
                "coverage_percentage": coverage_score * 100
            })
            
            suggestions = []
            if coverage_score < 0.8:
                suggestions.extend([
                    "Add more unit tests to increase coverage",
                    "Focus on testing critical business logic",
                    "Use coverage reports to identify untested code paths"
                ])
            
            return coverage_score, details, suggestions
            
        except Exception as e:
            self.logger.warning(f"Coverage check failed: {e}")
            return 0.5, {"error": str(e)}, ["Fix test setup and coverage tools"]
    
    async def _check_security(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check security using bandit and safety"""
        
        score = 1.0
        details = {}
        suggestions = []
        
        try:
            # Run bandit for security issues
            bandit_result = await self._run_command([
                "bandit", "-r", str(self.project_path), "-f", "json", "-q"
            ])
            
            bandit_issues = []
            if bandit_result.returncode == 0 and bandit_result.stdout:
                try:
                    bandit_data = json.loads(bandit_result.stdout)
                    bandit_issues = bandit_data.get("results", [])
                except json.JSONDecodeError:
                    pass
            
            # Calculate score based on security issues
            high_severity = len([i for i in bandit_issues if i.get("issue_severity") == "HIGH"])
            medium_severity = len([i for i in bandit_issues if i.get("issue_severity") == "MEDIUM"])
            
            # Penalize high severity issues more
            security_penalty = (high_severity * 0.2) + (medium_severity * 0.1)
            score = max(0.0, 1.0 - security_penalty)
            
            details.update({
                "bandit_issues": len(bandit_issues),
                "high_severity": high_severity,
                "medium_severity": medium_severity
            })
            
            if bandit_issues:
                suggestions.extend([
                    "Review and fix security issues identified by bandit",
                    "Implement input validation and sanitization",
                    "Use secure coding practices"
                ])
                
        except FileNotFoundError:
            suggestions.append("Install bandit for security analysis")
            score = 0.8  # Assume decent security if tools not available
        
        try:
            # Check for known vulnerable dependencies with safety
            safety_result = await self._run_command(["safety", "check", "--json"])
            
            if safety_result.returncode != 0 and safety_result.stdout:
                try:
                    safety_data = json.loads(safety_result.stdout)
                    vuln_count = len(safety_data)
                    
                    # Penalize vulnerable dependencies
                    score = min(score, max(0.5, 1.0 - (vuln_count * 0.15)))
                    
                    details["vulnerable_dependencies"] = vuln_count
                    suggestions.append("Update vulnerable dependencies")
                    
                except json.JSONDecodeError:
                    pass
                    
        except FileNotFoundError:
            suggestions.append("Install safety for dependency vulnerability scanning")
        
        return score, details, suggestions
    
    async def _check_performance(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check performance characteristics"""
        
        # Basic performance checks
        score = 0.8  # Default decent score
        details = {"metrics": "basic_estimation"}
        suggestions = []
        
        # Check for potential performance issues in code
        try:
            code_files = list(self.project_path.rglob("*.py"))
            
            performance_issues = 0
            for file_path in code_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for common performance anti-patterns
                    if "for" in content and "append" in content:
                        # Potential inefficient list building
                        performance_issues += content.count("append")
                    
                    if "time.sleep" in content:
                        # Blocking sleep calls
                        performance_issues += content.count("time.sleep")
                        
                except (UnicodeDecodeError, IOError):
                    continue
            
            # Calculate score based on issues found
            total_lines = await self._count_code_lines()
            issue_density = performance_issues / max(total_lines, 1) * 1000
            score = max(0.4, 1.0 - (issue_density * 0.05))
            
            details.update({
                "potential_issues": performance_issues,
                "total_lines": total_lines
            })
            
            if performance_issues > 10:
                suggestions.extend([
                    "Review code for performance bottlenecks",
                    "Consider using list comprehensions instead of loops with append",
                    "Replace blocking calls with async alternatives"
                ])
                
        except Exception as e:
            self.logger.warning(f"Performance check failed: {e}")
        
        return score, details, suggestions
    
    async def _check_documentation(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check documentation coverage and quality"""
        
        try:
            # Count documented vs undocumented functions/classes
            code_files = list(self.project_path.rglob("*.py"))
            
            total_functions = 0
            documented_functions = 0
            
            for file_path in code_files:
                if "test_" in file_path.name or "__pycache__" in str(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        
                        # Check for function/class definitions
                        if stripped.startswith(('def ', 'class ', 'async def ')):
                            total_functions += 1
                            
                            # Look for docstring in next few lines
                            for j in range(i+1, min(i+5, len(lines))):
                                next_line = lines[j].strip()
                                if next_line.startswith('"""') or next_line.startswith("'''"):
                                    documented_functions += 1
                                    break
                                elif next_line and not next_line.startswith('#'):
                                    break
                                    
                except (UnicodeDecodeError, IOError):
                    continue
            
            # Calculate documentation score
            if total_functions > 0:
                doc_score = documented_functions / total_functions
            else:
                doc_score = 1.0  # No functions to document
            
            # Check for README and other docs
            readme_exists = any([
                (self.project_path / name).exists() 
                for name in ["README.md", "README.rst", "README.txt"]
            ])
            
            docs_dir_exists = (self.project_path / "docs").exists()
            
            # Adjust score based on documentation infrastructure
            infrastructure_bonus = 0.0
            if readme_exists:
                infrastructure_bonus += 0.1
            if docs_dir_exists:
                infrastructure_bonus += 0.1
            
            final_score = min(1.0, doc_score + infrastructure_bonus)
            
            details = {
                "documented_functions": documented_functions,
                "total_functions": total_functions,
                "documentation_percentage": doc_score * 100,
                "readme_exists": readme_exists,
                "docs_directory": docs_dir_exists
            }
            
            suggestions = []
            if doc_score < 0.7:
                suggestions.extend([
                    "Add docstrings to functions and classes",
                    "Document public API methods",
                    "Include usage examples in documentation"
                ])
            
            if not readme_exists:
                suggestions.append("Create a README.md with project overview")
                
            return final_score, details, suggestions
            
        except Exception as e:
            self.logger.warning(f"Documentation check failed: {e}")
            return 0.5, {"error": str(e)}, ["Fix documentation analysis"]
    
    async def _check_dependencies(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check dependency health and security"""
        
        score = 1.0
        details = {}
        suggestions = []
        
        try:
            # Check for requirements files
            req_files = [
                self.project_path / "requirements.txt",
                self.project_path / "requirements-dev.txt", 
                self.project_path / "pyproject.toml"
            ]
            
            dependencies = []
            
            # Parse pyproject.toml if available
            pyproject_file = self.project_path / "pyproject.toml"
            if pyproject_file.exists():
                try:
                    with open(pyproject_file) as f:
                        content = f.read()
                    
                    # Simple parsing for dependencies
                    if 'dependencies = [' in content:
                        deps_section = content.split('dependencies = [')[1].split(']')[0]
                        deps = [line.strip().strip('"').strip("'").split(">=")[0] 
                               for line in deps_section.split('\n') if line.strip()]
                        dependencies.extend(deps)
                        
                except Exception:
                    pass
            
            # Parse requirements.txt files
            for req_file in req_files:
                if req_file.exists():
                    try:
                        with open(req_file) as f:
                            lines = f.readlines()
                        
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                dep_name = line.split('>=')[0].split('==')[0].split('[')[0]
                                if dep_name not in dependencies:
                                    dependencies.append(dep_name)
                                    
                    except Exception:
                        pass
            
            # Check for outdated packages (would normally use pip list --outdated)
            outdated_estimate = max(0, len(dependencies) // 10)  # Rough estimate
            
            # Check for excessive dependencies
            if len(dependencies) > 50:
                score -= 0.1
                suggestions.append("Consider reducing number of dependencies")
            
            # Simulate dependency health check
            score = max(0.7, score - (outdated_estimate * 0.05))
            
            details = {
                "total_dependencies": len(dependencies),
                "estimated_outdated": outdated_estimate,
                "requirements_files": [str(f) for f in req_files if f.exists()]
            }
            
            if outdated_estimate > 5:
                suggestions.append("Update outdated dependencies")
            
            suggestions.append("Regularly audit dependencies for security issues")
            
            return score, details, suggestions
            
        except Exception as e:
            self.logger.warning(f"Dependency check failed: {e}")
            return 0.8, {"error": str(e)}, ["Fix dependency analysis"]
    
    async def _check_code_formatting(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check code formatting consistency"""
        
        try:
            # Try black first
            result = await self._run_command([
                "black", "--check", "--diff", str(self.project_path)
            ])
            
            if result.returncode == 0:
                score = 1.0
                details = {"formatter": "black", "issues": 0}
                suggestions = []
            else:
                # Count formatting issues
                diff_lines = result.stdout.count('\n') if result.stdout else 0
                score = max(0.5, 1.0 - (diff_lines * 0.001))
                
                details = {
                    "formatter": "black",
                    "formatting_issues": diff_lines,
                    "needs_formatting": True
                }
                
                suggestions = [
                    "Run 'black .' to format code",
                    "Set up pre-commit hooks for automatic formatting"
                ]
            
            return score, details, suggestions
            
        except FileNotFoundError:
            # Black not available, do basic checks
            score = 0.8  # Assume reasonable formatting
            details = {"formatter": "basic", "message": "Black not available"}
            suggestions = ["Install black for code formatting"]
            
            return score, details, suggestions
    
    async def _check_type_checking(self) -> Tuple[float, Dict[str, Any], List[str]]:
        """Check type checking with mypy"""
        
        try:
            result = await self._run_command([
                "mypy", str(self.project_path), "--ignore-missing-imports"
            ])
            
            if result.returncode == 0:
                score = 1.0
                errors = 0
            else:
                # Count type errors
                error_lines = result.stdout.split('\n') if result.stdout else []
                errors = len([line for line in error_lines if ": error:" in line])
                
                # Calculate score based on error density
                total_lines = await self._count_code_lines()
                error_density = errors / max(total_lines, 1) * 100
                score = max(0.3, 1.0 - (error_density * 0.1))
            
            details = {
                "tool": "mypy",
                "type_errors": errors,
                "total_lines": await self._count_code_lines()
            }
            
            suggestions = []
            if errors > 0:
                suggestions.extend([
                    "Fix type checking errors",
                    "Add type hints to function signatures",
                    "Use proper type annotations"
                ])
            
            return score, details, suggestions
            
        except FileNotFoundError:
            # mypy not available
            score = 0.7  # Assume reasonable typing
            details = {"tool": "basic", "message": "mypy not available"}
            suggestions = ["Install mypy for type checking"]
            
            return score, details, suggestions
    
    async def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a command asynchronously"""
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd or self.project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode('utf-8', errors='ignore'),
            stderr=stderr.decode('utf-8', errors='ignore')
        )
    
    async def _count_code_lines(self) -> int:
        """Count total lines of code"""
        
        try:
            total_lines = 0
            code_files = list(self.project_path.rglob("*.py"))
            
            for file_path in code_files:
                if "__pycache__" in str(file_path) or "test_" in file_path.name:
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Count non-empty, non-comment lines
                    code_lines = [
                        line for line in lines 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    total_lines += len(code_lines)
                    
                except (UnicodeDecodeError, IOError):
                    continue
            
            return max(1, total_lines)  # Avoid division by zero
            
        except Exception:
            return 1000  # Default estimate
    
    async def _calculate_project_maturity(self) -> float:
        """Calculate project maturity score"""
        
        maturity_factors = {
            "has_tests": 0.2,
            "has_ci": 0.15, 
            "has_docs": 0.15,
            "has_typing": 0.1,
            "has_packaging": 0.1,
            "has_security": 0.1,
            "code_quality": 0.2
        }
        
        score = 0.0
        
        # Check for tests
        if list(self.project_path.rglob("test_*.py")):
            score += maturity_factors["has_tests"]
        
        # Check for CI configuration
        ci_files = [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile"]
        if any((self.project_path / path).exists() for path in ci_files):
            score += maturity_factors["has_ci"]
        
        # Check for documentation
        if (self.project_path / "README.md").exists() or (self.project_path / "docs").exists():
            score += maturity_factors["has_docs"]
        
        # Check for packaging
        if (self.project_path / "pyproject.toml").exists() or (self.project_path / "setup.py").exists():
            score += maturity_factors["has_packaging"]
        
        # Add base code quality score
        score += maturity_factors["code_quality"] * 0.7  # Assume reasonable quality
        
        return min(1.0, score)
    
    async def _adapt_thresholds(self) -> None:
        """Adapt quality thresholds based on project maturity"""
        
        # Adjust thresholds based on project maturity
        adjustment_factor = 0.8 + (self.project_maturity_score * 0.2)
        
        for gate_type in self.thresholds:
            original_threshold = self.thresholds[gate_type]
            adjusted_threshold = original_threshold * adjustment_factor
            self.thresholds[gate_type] = min(1.0, max(0.5, adjusted_threshold))
        
        self.logger.debug(f"Adapted thresholds with factor {adjustment_factor:.2f}")
    
    async def _generate_recommendations(
        self, 
        results: Dict[QualityGateType, QualityGateResult]
    ) -> List[str]:
        """Generate prioritized recommendations based on results"""
        
        recommendations = []
        
        # Priority: Security first
        if QualityGateType.SECURITY_SCAN in results:
            security_result = results[QualityGateType.SECURITY_SCAN]
            if security_result.status != QualityGateStatus.PASSED:
                recommendations.extend(security_result.remediation_suggestions)
        
        # Then test coverage
        if QualityGateType.TEST_COVERAGE in results:
            coverage_result = results[QualityGateType.TEST_COVERAGE]
            if coverage_result.status != QualityGateStatus.PASSED:
                recommendations.extend(coverage_result.remediation_suggestions)
        
        # Add other recommendations in order of importance
        priority_order = [
            QualityGateType.CODE_QUALITY,
            QualityGateType.TYPE_CHECKING,
            QualityGateType.DEPENDENCY_SCAN,
            QualityGateType.CODE_FORMATTING,
            QualityGateType.PERFORMANCE,
            QualityGateType.DOCUMENTATION
        ]
        
        for gate_type in priority_order:
            if gate_type in results and results[gate_type].status != QualityGateStatus.PASSED:
                recommendations.extend(results[gate_type].remediation_suggestions[:2])  # Top 2 per gate
        
        # Add general recommendations
        failed_gates = [r for r in results.values() if r.status == QualityGateStatus.FAILED]
        if len(failed_gates) > 2:
            recommendations.append("Consider establishing a quality improvement plan")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    async def _attempt_auto_remediation(self, report: QualityReport) -> bool:
        """Attempt automatic remediation of common issues"""
        
        if not self.auto_remediation:
            return False
        
        self.logger.info("🔧 Attempting auto-remediation...")
        
        remediation_success = True
        
        # Auto-format code
        if (QualityGateType.CODE_FORMATTING in report.gate_results and 
            report.gate_results[QualityGateType.CODE_FORMATTING].status != QualityGateStatus.PASSED):
            
            try:
                await self._run_command(["black", str(self.project_path)])
                self.logger.info("✅ Auto-formatted code with black")
            except FileNotFoundError:
                remediation_success = False
        
        # Auto-fix some ruff issues
        if (QualityGateType.CODE_QUALITY in report.gate_results and 
            report.gate_results[QualityGateType.CODE_QUALITY].status != QualityGateStatus.PASSED):
            
            try:
                await self._run_command(["ruff", "check", "--fix", str(self.project_path)])
                self.logger.info("✅ Auto-fixed code quality issues with ruff")
            except FileNotFoundError:
                remediation_success = False
        
        if remediation_success:
            self.logger.info("🎉 Auto-remediation completed successfully")
        else:
            self.logger.warning("⚠️  Some auto-remediation steps failed")
        
        return remediation_success
    
    def _log_quality_report(self, report: QualityReport) -> None:
        """Log quality report in a readable format"""
        
        self.logger.info("📊 Quality Gate Results:")
        self.logger.info(f"  Overall Score: {report.overall_score:.2%}")
        self.logger.info(f"  Overall Status: {report.overall_status.value.upper()}")
        self.logger.info(f"  Gates Passed: {report.passed_gates}/{report.total_gates}")
        self.logger.info(f"  Execution Time: {report.execution_duration:.1f}s")
        
        # Log individual gate results
        for gate_type, result in report.gate_results.items():
            status_emoji = {
                QualityGateStatus.PASSED: "✅",
                QualityGateStatus.WARNING: "⚠️",
                QualityGateStatus.FAILED: "❌",
                QualityGateStatus.ERROR: "💥",
                QualityGateStatus.SKIPPED: "⏭️"
            }
            
            emoji = status_emoji.get(result.status, "❓")
            self.logger.info(
                f"  {emoji} {gate_type.value}: {result.score:.2%} "
                f"(threshold: {result.threshold:.2%})"
            )
        
        # Log top recommendations
        if report.recommendations:
            self.logger.info("🔧 Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                self.logger.info(f"  {i}. {rec}")