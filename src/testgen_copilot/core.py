"""Core business logic and orchestration for TestGen Copilot Assistant."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .coverage import CoverageAnalyzer, CoverageResult
from .generator import GenerationConfig, TestGenerator
from .logging_config import LogContext, get_core_logger
from .metrics_collector import MetricsCollector
from .quality import TestQualityScorer
from .security import SecurityReport, SecurityScanner


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisPhase(Enum):
    """Phases of code analysis."""
    PARSING = "parsing"
    SECURITY_SCAN = "security_scan"
    TEST_GENERATION = "test_generation"
    COVERAGE_ANALYSIS = "coverage_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"


@dataclass
class ProjectMetrics:
    """Comprehensive project metrics."""
    files_analyzed: int = 0
    tests_generated: int = 0
    security_issues_found: int = 0
    coverage_percentage: float = 0.0
    quality_score: float = 0.0
    processing_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProcessingResult:
    """Result of processing a code file or project."""
    file_path: Path
    status: ProcessingStatus
    tests_generated: Optional[Path] = None
    security_report: Optional[SecurityReport] = None
    coverage_result: Optional[CoverageResult] = None
    quality_score: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    phase: Optional[AnalysisPhase] = None


class TestGenOrchestrator:
    """Main orchestration engine for TestGen Copilot operations."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        enable_security: bool = True,
        enable_coverage: bool = True,
        enable_quality: bool = True,
        concurrent_limit: int = 4
    ):
        self.config = config or GenerationConfig()
        self.enable_security = enable_security
        self.enable_coverage = enable_coverage
        self.enable_quality = enable_quality
        self.concurrent_limit = concurrent_limit

        # Initialize components
        self.generator = TestGenerator(config)
        self.security_scanner = SecurityScanner() if enable_security else None
        self.coverage_analyzer = CoverageAnalyzer() if enable_coverage else None
        self.quality_scorer = TestQualityScorer() if enable_quality else None
        self.metrics_collector = MetricsCollector()

        self.logger = get_core_logger()

    async def process_file(
        self,
        file_path: Union[str, Path],
        output_dir: Union[str, Path],
        phases: Optional[List[AnalysisPhase]] = None,
        retry_attempts: int = 3
    ) -> ProcessingResult:
        """Process a single file through all analysis phases."""
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        start_time = datetime.now()

        phases = phases or list(AnalysisPhase)
        result = ProcessingResult(file_path=file_path, status=ProcessingStatus.IN_PROGRESS)

        with LogContext(self.logger, "process_file", {
            "file_path": str(file_path),
            "output_dir": str(output_dir),
            "phases": [p.value for p in phases]
        }):
            try:
                # Phase 1: Test Generation (with retry logic)
                if AnalysisPhase.TEST_GENERATION in phases:
                    result.phase = AnalysisPhase.TEST_GENERATION
                    self.logger.info("Starting test generation phase", {
                        "file": str(file_path)
                    })

                    for attempt in range(retry_attempts):
                        try:
                            result.tests_generated = self.generator.generate_tests(file_path, output_dir)
                            self.logger.info("Test generation completed", {
                                "output_file": str(result.tests_generated),
                                "attempt": attempt + 1
                            })
                            break
                        except Exception as gen_error:
                            if attempt == retry_attempts - 1:
                                raise gen_error
                            self.logger.warning(f"Test generation attempt {attempt + 1} failed, retrying", {
                                "error": str(gen_error),
                                "remaining_attempts": retry_attempts - attempt - 1
                            })
                            await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

                # Phase 2: Security Analysis
                if AnalysisPhase.SECURITY_SCAN in phases and self.security_scanner:
                    result.phase = AnalysisPhase.SECURITY_SCAN
                    self.logger.info("Starting security analysis phase")

                    result.security_report = self.security_scanner.scan_file(str(file_path))
                    self.logger.info("Security analysis completed", {
                        "issues_found": len(result.security_report.issues) if result.security_report else 0
                    })

                # Phase 3: Coverage Analysis
                if AnalysisPhase.COVERAGE_ANALYSIS in phases and self.coverage_analyzer and result.tests_generated:
                    result.phase = AnalysisPhase.COVERAGE_ANALYSIS
                    self.logger.info("Starting coverage analysis phase")

                    result.coverage_result = self.coverage_analyzer.analyze_file(
                        str(file_path),
                        str(result.tests_generated.parent)
                    )
                    self.logger.info("Coverage analysis completed", {
                        "coverage_percentage": result.coverage_result.percentage if result.coverage_result else 0
                    })

                # Phase 4: Quality Assessment
                if AnalysisPhase.QUALITY_ASSESSMENT in phases and self.quality_scorer and result.tests_generated:
                    result.phase = AnalysisPhase.QUALITY_ASSESSMENT
                    self.logger.info("Starting quality assessment phase")

                    result.quality_score = self.quality_scorer.score_test_file(str(result.tests_generated))
                    self.logger.info("Quality assessment completed", {
                        "quality_score": result.quality_score
                    })

                result.status = ProcessingStatus.COMPLETED
                result.processing_time = (datetime.now() - start_time).total_seconds()

                self.logger.info("File processing completed successfully", {
                    "file": str(file_path),
                    "processing_time": result.processing_time,
                    "status": result.status.value
                })

            except Exception as e:
                result.status = ProcessingStatus.FAILED
                result.errors.append(str(e))
                result.processing_time = (datetime.now() - start_time).total_seconds()

                self.logger.error("File processing failed", {
                    "file": str(file_path),
                    "error": str(e),
                    "phase": result.phase.value if result.phase else "unknown",
                    "processing_time": result.processing_time
                })

        return result

    async def process_project(
        self,
        project_path: Union[str, Path],
        output_dir: Union[str, Path],
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, ProcessingResult]:
        """Process an entire project with parallel file processing."""
        project_path = Path(project_path)
        output_dir = Path(output_dir)

        file_patterns = file_patterns or ["*.py", "*.js", "*.ts", "*.java", "*.cs", "*.go", "*.rs"]
        exclude_patterns = exclude_patterns or ["*test*", "*spec*", "__pycache__", "node_modules"]

        with LogContext(self.logger, "process_project", {
            "project_path": str(project_path),
            "output_dir": str(output_dir),
            "file_patterns": file_patterns,
            "exclude_patterns": exclude_patterns
        }):
            # Discover source files
            source_files = self._discover_source_files(project_path, file_patterns, exclude_patterns)
            self.logger.info("Discovered source files", {
                "file_count": len(source_files),
                "files": [str(f) for f in source_files[:10]]  # Log first 10 files
            })

            # Process files concurrently
            semaphore = asyncio.Semaphore(self.concurrent_limit)
            tasks = []

            async def process_with_semaphore(file_path: Path) -> ProcessingResult:
                async with semaphore:
                    return await self.process_file(file_path, output_dir)

            for file_path in source_files:
                task = asyncio.create_task(process_with_semaphore(file_path))
                tasks.append((str(file_path), task))

            # Wait for all tasks to complete
            results = {}
            for file_path_str, task in tasks:
                try:
                    result = await task
                    results[file_path_str] = result
                except Exception as e:
                    self.logger.error("Task failed", {
                        "file": file_path_str,
                        "error": str(e)
                    })
                    results[file_path_str] = ProcessingResult(
                        file_path=Path(file_path_str),
                        status=ProcessingStatus.FAILED,
                        errors=[str(e)]
                    )

            # Generate project metrics
            metrics = self._calculate_project_metrics(results)
            self.logger.info("Project processing completed", {
                "total_files": len(results),
                "successful_files": sum(1 for r in results.values() if r.status == ProcessingStatus.COMPLETED),
                "failed_files": sum(1 for r in results.values() if r.status == ProcessingStatus.FAILED),
                "metrics": {
                    "coverage_percentage": metrics.coverage_percentage,
                    "quality_score": metrics.quality_score,
                    "security_issues": metrics.security_issues_found
                }
            })

        # Store metrics for monitoring
        self.metrics_collector.record_batch_metrics(metrics)
        return results

    def generate_comprehensive_report(
        self,
        results: Dict[str, ProcessingResult],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive analysis report for processed files."""
        metrics = self._calculate_project_metrics(results)

        report = {
            "summary": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_files_processed": len(results),
                "successful_files": sum(1 for r in results.values() if r.status == ProcessingStatus.COMPLETED),
                "failed_files": sum(1 for r in results.values() if r.status == ProcessingStatus.FAILED),
                "total_processing_time": sum(r.processing_time for r in results.values()),
            },
            "metrics": {
                "files_analyzed": metrics.files_analyzed,
                "tests_generated": metrics.tests_generated,
                "security_issues_found": metrics.security_issues_found,
                "average_coverage_percentage": metrics.coverage_percentage,
                "average_quality_score": metrics.quality_score,
                "processing_time_seconds": metrics.processing_time_seconds
            },
            "file_results": [],
            "security_summary": {
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            },
            "recommendations": []
        }

        # Process individual file results
        for file_path, result in results.items():
            file_report = {
                "file_path": file_path,
                "status": result.status.value,
                "processing_time": result.processing_time,
                "tests_generated": str(result.tests_generated) if result.tests_generated else None,
                "coverage_percentage": result.coverage_result.percentage if result.coverage_result else None,
                "quality_score": result.quality_score,
                "security_issues": len(result.security_report.issues) if result.security_report else 0,
                "errors": result.errors,
                "warnings": result.warnings
            }
            report["file_results"].append(file_report)

            # Aggregate security issues
            if result.security_report:
                for issue in result.security_report.issues:
                    report["security_summary"]["total_issues"] += 1
                    severity_key = f"{issue.severity.lower()}_issues"
                    if severity_key in report["security_summary"]:
                        report["security_summary"][severity_key] += 1

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(results, metrics)

        # Save report if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info("Comprehensive report saved", {
                "output_path": str(output_path)
            })

        return report

    def _discover_source_files(
        self,
        project_path: Path,
        patterns: List[str],
        exclude_patterns: List[str]
    ) -> List[Path]:
        """Discover source files matching patterns while excluding specified patterns."""
        source_files = []

        for pattern in patterns:
            for file_path in project_path.rglob(pattern):
                if file_path.is_file():
                    # Check exclude patterns
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if exclude_pattern in str(file_path):
                            should_exclude = True
                            break

                    if not should_exclude:
                        source_files.append(file_path)

        return sorted(set(source_files))

    def _calculate_project_metrics(self, results: Dict[str, ProcessingResult]) -> ProjectMetrics:
        """Calculate comprehensive project metrics from processing results."""
        successful_results = [r for r in results.values() if r.status == ProcessingStatus.COMPLETED]

        if not successful_results:
            return ProjectMetrics()

        coverage_scores = [r.coverage_result.percentage for r in successful_results if r.coverage_result]
        quality_scores = [r.quality_score for r in successful_results if r.quality_score is not None]
        security_issues = sum(
            len(r.security_report.issues) for r in successful_results
            if r.security_report
        )

        return ProjectMetrics(
            files_analyzed=len(successful_results),
            tests_generated=sum(1 for r in successful_results if r.tests_generated),
            security_issues_found=security_issues,
            coverage_percentage=sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0,
            quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            processing_time_seconds=sum(r.processing_time for r in results.values())
        )

    def _generate_recommendations(
        self,
        results: Dict[str, ProcessingResult],
        metrics: ProjectMetrics
    ) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []

        # Coverage recommendations
        if metrics.coverage_percentage < 80:
            recommendations.append(
                f"Consider improving test coverage (currently {metrics.coverage_percentage:.1f}%). "
                "Aim for at least 80% coverage for better code reliability."
            )

        # Quality recommendations
        if metrics.quality_score < 75:
            recommendations.append(
                f"Test quality score is {metrics.quality_score:.1f}%. "
                "Consider adding more edge cases, error handling tests, and assertions."
            )

        # Security recommendations
        if metrics.security_issues_found > 0:
            recommendations.append(
                f"Found {metrics.security_issues_found} security issues. "
                "Review and address security vulnerabilities to improve code safety."
            )

        # Performance recommendations
        failed_files = sum(1 for r in results.values() if r.status == ProcessingStatus.FAILED)
        if failed_files > 0:
            recommendations.append(
                f"{failed_files} files failed processing. "
                "Check error logs and ensure files are accessible and syntactically correct."
            )

        if metrics.processing_time_seconds > 300:  # 5 minutes
            recommendations.append(
                "Processing time is high. Consider running analysis on smaller batches "
                "or increasing concurrent processing limits."
            )

        return recommendations


# Legacy function for backward compatibility
def identity(value):
    """Return the input value as-is."""
    return value
