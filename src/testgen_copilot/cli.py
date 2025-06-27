"""Command line interface for TestGen Copilot."""

from __future__ import annotations

import argparse

from pathlib import Path

from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner
from .vscode import scaffold_extension
from .coverage import CoverageAnalyzer
from .quality import TestQualityScorer


def _coverage_failures(
    project_dir: str | Path, target: float, tests_dir: str | Path | None = None
) -> list[tuple[str, float, set[str]]]:
    """Return modules below ``target`` with their uncovered functions."""
    analyzer = CoverageAnalyzer()
    project = Path(project_dir)
    tests_dir = Path(tests_dir) if tests_dir else Path("tests")
    if not tests_dir.is_absolute():
        tests_dir = project / tests_dir
    failures: list[tuple[str, float, set[str]]] = []
    for path in project.rglob("*.py"):
        if tests_dir in path.parents:
            continue
        cov = analyzer.analyze(path, tests_dir)
        if cov < target:
            missing = analyzer.uncovered_functions(path, tests_dir)
            failures.append((str(path.relative_to(project)), cov, missing))
    return failures


def _check_project_coverage(
    project_dir: str | Path, target: float, tests_dir: str | Path | None = None
) -> list[str]:
    """Return list of modules that fail the coverage ``target``."""
    return [
        f"{m}: {c:.1f}%" for m, c, _ in _coverage_failures(project_dir, target, tests_dir)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate unit tests for a file")
    parser.add_argument("--file", help="Source file to analyze")
    parser.add_argument(
        "--language",
        default="python",
        help="Programming language of the source file",
    )
    parser.add_argument("--project", help="Project directory to scan")
    parser.add_argument(
        "--output",
        help="Directory where generated tests should be written",
    )
    parser.add_argument(
        "--scaffold-vscode",
        help="Create a VS Code extension scaffold in the given directory",
    )
    parser.add_argument(
        "--security-scan", action="store_true", help="Run security analysis"
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        help="Fail if project coverage is below this percentage",
    )
    parser.add_argument(
        "--quality-target",
        type=float,
        help="Fail if test quality is below this percentage",
    )
    parser.add_argument(
        "--tests-dir",
        default="tests",
        help="Location of test files relative to the project root",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="List uncovered functions for modules below the coverage target",
    )

    args = parser.parse_args(argv)

    if args.scaffold_vscode:
        path = scaffold_extension(args.scaffold_vscode)
        print(f"VS Code extension scaffolded at {path.parent}")
        return

    is_generating = args.file and args.output
    analysis_only = (
        (args.coverage_target is not None or args.quality_target is not None)
        and args.project
        and not is_generating
    )

    if not is_generating and not analysis_only:
        parser.error(
            "--file and --output are required unless --scaffold-vscode or "
            "--coverage-target/--quality-target with --project is used"
        )

    if analysis_only:
        if args.coverage_target is not None:
            failures = _coverage_failures(args.project, args.coverage_target, args.tests_dir)
            if failures:
                print("Coverage below target:")
                for mod, cov, missing in failures:
                    print(f"  {mod}: {cov:.1f}%")
                    if args.show_missing and missing:
                        names = ", ".join(sorted(missing))
                        print(f"    Missing: {names}")
                parser.exit(status=1, message="Coverage target not met\n")
            print("Coverage target satisfied")
        if args.quality_target is not None:
            tests_dir = Path(args.tests_dir)
            if not tests_dir.is_absolute():
                tests_dir = Path(args.project) / tests_dir
            scorer = TestQualityScorer()
            score = scorer.score(tests_dir)
            if score < args.quality_target:
                print(f"Test quality: {score:.1f}%")
                lacking = scorer.low_quality_tests(tests_dir)
                if lacking:
                    names = ", ".join(sorted(lacking))
                    print(f"  Missing asserts in: {names}")
                parser.exit(status=1, message="Quality target not met\n")
            print("Quality target satisfied")
        return

    config = GenerationConfig(language=args.language)
    generator = TestGenerator(config)
    scanner = SecurityScanner()
    output_path = generator.generate_tests(args.file, args.output)
    if args.security_scan:
        target = args.project if args.project else args.file
        reports = (
            scanner.scan_project(target)
            if args.project
            else [scanner.scan_file(target)]
        )
        for rep in reports:
            print(rep.to_text())
    if args.coverage_target is not None and args.project:
        failures = _coverage_failures(args.project, args.coverage_target, args.tests_dir)
        if failures:
            print("Coverage below target:")
            for mod, cov, missing in failures:
                print(f"  {mod}: {cov:.1f}%")
                if args.show_missing and missing:
                    names = ", ".join(sorted(missing))
                    print(f"    Missing: {names}")
            parser.exit(status=1, message="Coverage target not met\n")
        print("Coverage target satisfied")
    if args.quality_target is not None and args.project:
        tests_dir = Path(args.tests_dir)
        if not tests_dir.is_absolute():
            tests_dir = Path(args.project) / tests_dir
        scorer = TestQualityScorer()
        score = scorer.score(tests_dir)
        if score < args.quality_target:
            print(f"Test quality: {score:.1f}%")
            lacking = scorer.low_quality_tests(tests_dir)
            if lacking:
                names = ", ".join(sorted(lacking))
                print(f"  Missing asserts in: {names}")
            parser.exit(status=1, message="Quality target not met\n")
        print("Quality target satisfied")
    print(f"Generated tests -> {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
