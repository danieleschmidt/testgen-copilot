#!/usr/bin/env python3
"""
🧪 QUALITY GATES VALIDATION SYSTEM
==================================

Comprehensive quality gates with automatic validation and scoring.
Implements mandatory quality checks with 85%+ pass threshold.
"""

import ast
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

class QualityGate:
    """Base class for quality gates"""
    
    def __init__(self, name: str, threshold: float = 0.85):
        self.name = name
        self.threshold = threshold
        self.score = 0.0
        self.passed = False
        self.details = []
    
    def validate(self, project_path: Path) -> bool:
        """Validate quality gate - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_report(self) -> Dict[str, Any]:
        """Get quality gate report"""
        return {
            "name": self.name,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "details": self.details
        }


class CodeQualityGate(QualityGate):
    """Code quality validation gate"""
    
    def __init__(self):
        super().__init__("Code Quality", threshold=0.85)
    
    def validate(self, project_path: Path) -> bool:
        """Validate code quality metrics"""
        python_files = list(project_path.rglob("*.py"))
        if not python_files:
            self.score = 1.0
            self.passed = True
            self.details.append("No Python files found to analyze")
            return True
        
        total_score = 0.0
        file_count = 0
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
                
            file_score = self._analyze_file(py_file)
            total_score += file_score
            file_count += 1
        
        if file_count > 0:
            self.score = total_score / file_count
        else:
            self.score = 1.0
            
        self.passed = self.score >= self.threshold
        self.details.append(f"Analyzed {file_count} Python files")
        self.details.append(f"Average code quality score: {self.score:.2f}")
        
        return self.passed
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped from analysis"""
        skip_patterns = ["__pycache__", ".git", "venv", "env", "node_modules", "test_"]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> float:
        """Analyze individual file for code quality"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Quality metrics
            metrics = {
                "has_docstring": self._has_module_docstring(tree),
                "function_complexity": self._check_function_complexity(tree),
                "line_length": self._check_line_length(content),
                "naming_convention": self._check_naming_convention(tree),
                "imports_organized": self._check_import_organization(tree)
            }
            
            # Calculate weighted score
            weights = {
                "has_docstring": 0.2,
                "function_complexity": 0.3,
                "line_length": 0.2,
                "naming_convention": 0.2,
                "imports_organized": 0.1
            }
            
            score = sum(metrics[key] * weights[key] for key in metrics)
            return score
            
        except Exception as e:
            self.details.append(f"Error analyzing {file_path}: {e}")
            return 0.5  # Neutral score for unparseable files
    
    def _has_module_docstring(self, tree: ast.AST) -> float:
        """Check if module has docstring"""
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
            return 1.0
        return 0.0
    
    def _check_function_complexity(self, tree: ast.AST) -> float:
        """Check function complexity (simplified cyclomatic complexity)"""
        complexities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                complexities.append(complexity)
        
        if not complexities:
            return 1.0
        
        # Score based on average complexity (lower is better)
        avg_complexity = sum(complexities) / len(complexities)
        if avg_complexity <= 5:
            return 1.0
        elif avg_complexity <= 10:
            return 0.8
        elif avg_complexity <= 15:
            return 0.6
        else:
            return 0.4
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_line_length(self, content: str) -> float:
        """Check line length compliance"""
        lines = content.split('\n')
        long_lines = [line for line in lines if len(line) > 100]
        
        if not lines:
            return 1.0
        
        compliance_rate = 1.0 - (len(long_lines) / len(lines))
        return max(0.0, compliance_rate)
    
    def _check_naming_convention(self, tree: ast.AST) -> float:
        """Check naming convention compliance"""
        violations = 0
        total_names = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_names += 1
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    violations += 1
            elif isinstance(node, ast.ClassDef):
                total_names += 1
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    violations += 1
        
        if total_names == 0:
            return 1.0
        
        return 1.0 - (violations / total_names)
    
    def _check_import_organization(self, tree: ast.AST) -> float:
        """Check import organization"""
        imports = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
        
        if not imports:
            return 1.0
        
        # Simple check: imports should be at the top
        non_import_before_import = False
        for i, node in enumerate(tree.body):
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                for j in range(i + 1, len(tree.body)):
                    if isinstance(tree.body[j], (ast.Import, ast.ImportFrom)):
                        non_import_before_import = True
                        break
                break
        
        return 0.7 if non_import_before_import else 1.0


class SecurityGate(QualityGate):
    """Security validation gate"""
    
    def __init__(self):
        super().__init__("Security", threshold=0.90)
    
    def validate(self, project_path: Path) -> bool:
        """Validate security measures"""
        python_files = list(project_path.rglob("*.py"))
        vulnerabilities = []
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
            
            file_vulns = self._scan_file_security(py_file)
            vulnerabilities.extend(file_vulns)
        
        # Score based on vulnerabilities found
        if len(python_files) == 0:
            self.score = 1.0
        else:
            # Lower score for more vulnerabilities
            vuln_rate = len(vulnerabilities) / len(python_files)
            self.score = max(0.0, 1.0 - vuln_rate * 0.2)
        
        self.passed = self.score >= self.threshold
        self.details.append(f"Scanned {len(python_files)} files")
        self.details.append(f"Found {len(vulnerabilities)} potential security issues")
        
        if vulnerabilities:
            self.details.extend(vulnerabilities[:5])  # Show first 5 issues
        
        return self.passed
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped from security scan"""
        skip_patterns = ["__pycache__", ".git", "venv", "env", "test_"]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _scan_file_security(self, file_path: Path) -> List[str]:
        """Scan file for security vulnerabilities"""
        vulnerabilities = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check for common security issues
            security_patterns = [
                (r'eval\s*\(', "Use of eval() function"),
                (r'exec\s*\(', "Use of exec() function"),
                (r'os\.system\s*\(', "Use of os.system()"),
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection risk"),
                (r'pickle\.loads?\s*\(', "Unsafe deserialization"),
                (r'input\s*\([^)]*\)\s*.*os\.system', "Command injection risk"),
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
                (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key")
            ]
            
            for pattern, issue in security_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    vulnerabilities.append(f"{file_path.name}: {issue}")
            
        except Exception as e:
            vulnerabilities.append(f"{file_path.name}: Error scanning file - {e}")
        
        return vulnerabilities


class PerformanceGate(QualityGate):
    """Performance validation gate"""
    
    def __init__(self):
        super().__init__("Performance", threshold=0.80)
    
    def validate(self, project_path: Path) -> bool:
        """Validate performance characteristics"""
        python_files = list(project_path.rglob("*.py"))
        performance_issues = []
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
            
            issues = self._analyze_performance(py_file)
            performance_issues.extend(issues)
        
        # Score based on performance issues
        if len(python_files) == 0:
            self.score = 1.0
        else:
            issue_rate = len(performance_issues) / len(python_files)
            self.score = max(0.0, 1.0 - issue_rate * 0.3)
        
        self.passed = self.score >= self.threshold
        self.details.append(f"Analyzed {len(python_files)} files for performance")
        self.details.append(f"Found {len(performance_issues)} potential performance issues")
        
        if performance_issues:
            self.details.extend(performance_issues[:3])
        
        return self.passed
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ["__pycache__", ".git", "venv", "test_"]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_performance(self, file_path: Path) -> List[str]:
        """Analyze file for performance issues"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Check for performance anti-patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # Check for nested loops
                    for child in ast.walk(node):
                        if child != node and isinstance(child, ast.For):
                            issues.append(f"{file_path.name}: Nested loops detected - consider optimization")
                            break
                
                elif isinstance(node, ast.ListComp):
                    # Check for complex list comprehensions
                    if len(list(ast.walk(node))) > 10:
                        issues.append(f"{file_path.name}: Complex list comprehension - consider breaking down")
            
            # Check for inefficient patterns in source
            inefficient_patterns = [
                (r'\.append\s*\([^)]+\)\s*in\s+for', "List append in loop - consider list comprehension"),
                (r'range\s*\(\s*len\s*\(', "range(len()) - consider enumerate()"),
                (r'\.keys\s*\(\s*\)\s*in\s+for.*\[', "dict.keys() with indexing - use .items()")
            ]
            
            for pattern, issue in inefficient_patterns:
                if re.search(pattern, content):
                    issues.append(f"{file_path.name}: {issue}")
            
        except Exception as e:
            issues.append(f"{file_path.name}: Error analyzing performance - {e}")
        
        return issues


class TestCoverageGate(QualityGate):
    """Test coverage validation gate"""
    
    def __init__(self):
        super().__init__("Test Coverage", threshold=0.85)
    
    def validate(self, project_path: Path) -> bool:
        """Validate test coverage"""
        src_files = list((project_path / "src").rglob("*.py")) if (project_path / "src").exists() else []
        test_files = list((project_path / "tests").rglob("*.py")) if (project_path / "tests").exists() else []
        
        if not src_files:
            # Look for Python files in project root
            src_files = [f for f in project_path.rglob("*.py") if not self._is_test_file(f)]
        
        if not src_files:
            self.score = 1.0
            self.passed = True
            self.details.append("No source files found to test")
            return True
        
        # Calculate coverage estimate based on test files
        coverage_estimate = self._estimate_coverage(src_files, test_files)
        self.score = coverage_estimate
        self.passed = self.score >= self.threshold
        
        self.details.append(f"Found {len(src_files)} source files")
        self.details.append(f"Found {len(test_files)} test files")
        self.details.append(f"Estimated coverage: {self.score:.1%}")
        
        return self.passed
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file"""
        test_indicators = ["test_", "_test", "tests/", "conftest"]
        return any(indicator in str(file_path) for indicator in test_indicators)
    
    def _estimate_coverage(self, src_files: List[Path], test_files: List[Path]) -> float:
        """Estimate test coverage based on file analysis"""
        if not src_files:
            return 1.0
        
        if not test_files:
            return 0.0
        
        # Simple heuristic: ratio of test files to source files
        base_coverage = min(1.0, len(test_files) / len(src_files))
        
        # Bonus for test quality indicators
        test_content = ""
        for test_file in test_files:
            try:
                test_content += test_file.read_text(encoding='utf-8')
            except:
                continue
        
        # Look for testing patterns
        quality_indicators = [
            r'def test_',
            r'assert\s+',
            r'@pytest\.mark',
            r'unittest\.TestCase',
            r'mock\.',
            r'patch\('
        ]
        
        quality_bonus = 0.0
        for pattern in quality_indicators:
            if re.search(pattern, test_content):
                quality_bonus += 0.1
        
        # Cap at 1.0
        return min(1.0, base_coverage + quality_bonus * 0.1)


class DocumentationGate(QualityGate):
    """Documentation validation gate"""
    
    def __init__(self):
        super().__init__("Documentation", threshold=0.75)
    
    def validate(self, project_path: Path) -> bool:
        """Validate documentation completeness"""
        doc_score = 0.0
        doc_components = 0
        
        # Check for README
        readme_files = list(project_path.glob("README*"))
        if readme_files:
            doc_score += 0.3
            self.details.append("README file found")
        doc_components += 1
        
        # Check for API documentation
        docs_dir = project_path / "docs"
        if docs_dir.exists() and list(docs_dir.rglob("*.md")):
            doc_score += 0.2
            self.details.append("Documentation directory found")
        doc_components += 1
        
        # Check for docstrings in Python files
        python_files = list(project_path.rglob("*.py"))
        if python_files:
            docstring_score = self._check_docstrings(python_files)
            doc_score += docstring_score * 0.3
            self.details.append(f"Docstring coverage: {docstring_score:.1%}")
        doc_components += 1
        
        # Check for configuration documentation
        config_files = list(project_path.glob("*.toml")) + list(project_path.glob("*.yaml")) + list(project_path.glob("*.json"))
        if config_files:
            doc_score += 0.1
            self.details.append("Configuration files found")
        doc_components += 1
        
        # Check for examples
        example_indicators = ["example", "demo", "sample"]
        example_files = []
        for indicator in example_indicators:
            example_files.extend(project_path.rglob(f"*{indicator}*"))
        
        if example_files:
            doc_score += 0.1
            self.details.append("Example/demo files found")
        doc_components += 1
        
        self.score = doc_score
        self.passed = self.score >= self.threshold
        
        return self.passed
    
    def _check_docstrings(self, python_files: List[Path]) -> float:
        """Check docstring coverage in Python files"""
        total_functions = 0
        documented_functions = 0
        
        for py_file in python_files:
            if "test_" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        total_functions += 1
                        
                        # Check if it has a docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                            documented_functions += 1
            
            except Exception:
                continue
        
        if total_functions == 0:
            return 1.0
        
        return documented_functions / total_functions


class QualityGateValidator:
    """Main quality gate validation system"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.gates = [
            CodeQualityGate(),
            SecurityGate(),
            PerformanceGate(),
            TestCoverageGate(),
            DocumentationGate()
        ]
        self.results = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all quality gates"""
        print("🧪 QUALITY GATES VALIDATION")
        print("=" * 35)
        
        passed_gates = 0
        total_gates = len(self.gates)
        
        for gate in self.gates:
            print(f"\n📋 Validating {gate.name}...")
            
            start_time = time.time()
            passed = gate.validate(self.project_path)
            validation_time = time.time() - start_time
            
            self.results[gate.name] = gate.get_report()
            self.results[gate.name]["validation_time"] = validation_time
            
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status} - Score: {gate.score:.2f} (threshold: {gate.threshold:.2f})")
            
            if gate.details:
                for detail in gate.details[:3]:  # Show first 3 details
                    print(f"   • {detail}")
            
            if passed:
                passed_gates += 1
        
        # Calculate overall score
        overall_score = sum(gate.score for gate in self.gates) / len(self.gates)
        overall_passed = passed_gates >= total_gates * 0.8  # 80% of gates must pass
        
        print(f"\n🏆 QUALITY GATES SUMMARY")
        print("=" * 30)
        print(f"Passed Gates: {passed_gates}/{total_gates}")
        print(f"Overall Score: {overall_score:.2f}")
        print(f"Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        self.results["summary"] = {
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "overall_score": overall_score,
            "overall_passed": overall_passed,
            "pass_rate": passed_gates / total_gates
        }
        
        return self.results
    
    def save_report(self, output_path: Path) -> None:
        """Save quality gate report to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📄 Quality gate report saved to: {output_path}")


def main():
    """Main execution function"""
    project_path = Path("/root/repo")
    
    validator = QualityGateValidator(project_path)
    results = validator.validate_all()
    
    # Save report
    report_path = project_path / "quality_gates_report.json"
    validator.save_report(report_path)
    
    # Return exit code based on results
    return 0 if results["summary"]["overall_passed"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)