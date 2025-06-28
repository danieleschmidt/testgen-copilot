"""Command line interface for TestGen Copilot."""

from __future__ import annotations

import argparse
import time

from pathlib import Path

from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner
from .vscode import scaffold_extension
from .coverage import CoverageAnalyzer
from .quality import TestQualityScorer


LANG_PATTERNS = {
    "python": "*.py",
    "py": "*.py",
    "javascript": "*.js",
    "js": "*.js",
    "typescript": "*.ts",
    "ts": "*.ts",
    "java": "*.java",
    "c#": "*.cs",
    "csharp": "*.cs",
    "go": "*.go",
    "rust": "*.rs",
}


def _language_pattern(language: str) -> str:
    """Return glob pattern for ``language`` source files."""
    return LANG_PATTERNS.get(language.lower(), f"*{language}")


def _load_config(path: Path) -> dict:
    """Load configuration dictionary from ``path`` if it exists."""
    if path.exists():
        try:
            import json

            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


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


def _watch_for_changes(
    directory: Path,
    generator: TestGenerator,
    out_dir: Path,
    pattern: str,
    poll: float = 1.0,
    *,
    auto_generate: bool = False,
    max_cycles: int | None = None,
) -> None:
    """Watch ``directory`` for updates and optionally generate tests."""
    seen = {p: p.stat().st_mtime for p in directory.rglob(pattern)}
    cycles = 0
    try:
        while True:
            for path in directory.rglob(pattern):
                if out_dir in path.parents:
                    continue
                mtime = path.stat().st_mtime
                if path not in seen or mtime > seen[path]:
                    if auto_generate:
                        generator.generate_tests(path, out_dir)
                    else:
                        print(f"Change detected: {path}")
                    seen[path] = mtime
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        pass


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
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Skip generating edge case tests",
    )
    parser.add_argument(
        "--no-error-tests",
        action="store_true",
        help="Skip generating error path tests",
    )
    parser.add_argument(
        "--no-benchmark-tests",
        action="store_true",
        help="Skip generating performance benchmark tests",
    )
    parser.add_argument(
        "--no-integration-tests",
        action="store_true",
        help="Skip generating integration tests",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (defaults to .testgen.config.json)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate tests for all project files (requires --project and --output)",
    )
    parser.add_argument(
        "--watch",
        help="Watch a directory for changes and regenerate tests",
    )
    parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="Automatically generate tests when watching for changes",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Polling interval in seconds when watching for changes",
    )

    args = parser.parse_args(argv)

    base = Path(args.project) if args.project else Path.cwd()
    cfg_path = Path(args.config) if args.config else base / ".testgen.config.json"
    cfg = _load_config(cfg_path)

    if args.scaffold_vscode:
        path = scaffold_extension(args.scaffold_vscode)
        print(f"VS Code extension scaffolded at {path.parent}")
        return

    if args.batch:
        if args.file:
            parser.error("--file cannot be used with --batch")
        if not args.project:
            parser.error("--batch requires --project")
        if not args.output:
            parser.error("--batch requires --output")

    if args.watch:
        if args.file:
            parser.error("--file cannot be used with --watch")
        if args.batch:
            parser.error("--watch cannot be used with --batch")
        if not args.output:
            parser.error("--watch requires --output")

    is_batch = args.batch and args.project and args.output
    is_watch = args.watch and args.output
    is_generating = (args.file and args.output) or is_batch or is_watch
    analysis_only = (
        (args.coverage_target is not None or args.quality_target is not None)
        and args.project
        and not ((args.file and args.output) or is_batch or is_watch)
    )

    if not is_generating and not analysis_only:
        parser.error(
            "--file/--output or --batch with --project and --output are required "
            "unless --scaffold-vscode or --coverage-target/--quality-target with "
            "--project is used"
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

    language = args.language if args.language != "python" else cfg.get("language", args.language)
    include_edge_cases = cfg.get("include_edge_cases", True)
    include_error_paths = cfg.get("include_error_paths", True)
    include_benchmarks = cfg.get("include_benchmarks", True)
    include_integration_tests = cfg.get("include_integration_tests", True)
    if args.no_edge_cases:
        include_edge_cases = False
    if args.no_error_tests:
        include_error_paths = False
    if args.no_benchmark_tests:
        include_benchmarks = False
    if args.no_integration_tests:
        include_integration_tests = False

    config = GenerationConfig(
        language=language,
        include_edge_cases=include_edge_cases,
        include_error_paths=include_error_paths,
        include_benchmarks=include_benchmarks,
        include_integration_tests=include_integration_tests,
    )
    generator = TestGenerator(config)
    scanner = SecurityScanner()

    if args.watch:
        if args.file or args.batch:
            parser.error("--watch cannot be used with --file or --batch")
        if not args.output:
            parser.error("--watch requires --output")
        watch_dir = Path(args.watch)
        out_dir = Path(args.output)
        pattern = _language_pattern(config.language)
        _watch_for_changes(
            watch_dir,
            generator,
            out_dir,
            pattern,
            poll=args.poll,
            auto_generate=args.auto_generate,
        )
        return

    if is_batch:
        project = Path(args.project)
        out_dir = Path(args.output)
        pattern = _language_pattern(config.language)
        files = [
            p
            for p in project.rglob(pattern)
            if out_dir not in p.parents
        ]
        for f in files:
            generator.generate_tests(f, out_dir)
        output_path = out_dir
    else:
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
