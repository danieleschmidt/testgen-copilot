"""Command line interface for TestGen Copilot."""

from __future__ import annotations

import argparse
import time

from pathlib import Path
import logging

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


logger = logging.getLogger(__name__)


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
    logger.info("Watching %s for changes", directory)
    seen = {p: p.stat().st_mtime for p in directory.rglob(pattern)}
    dir_times = {d: d.stat().st_mtime for d in directory.rglob("*") if d.is_dir()}
    cycles = 0
    try:
        while True:
            for d in list(dir_times):
                try:
                    mtime = d.stat().st_mtime
                except FileNotFoundError:
                    dir_times.pop(d, None)
                    continue
                if mtime != dir_times[d]:
                    dir_times[d] = mtime
                    for path in d.glob(pattern):
                        if out_dir in path.parents:
                            continue
                        m = path.stat().st_mtime
                        if path not in seen or m > seen[path]:
                            if auto_generate:
                                generator.generate_tests(path, out_dir)
                                logger.info("Generated tests for %s", path)
                            else:
                                logger.info("Change detected: %s", path)
                            seen[path] = m
            for sub in directory.rglob("*"):
                if sub.is_dir() and sub not in dir_times:
                    dir_times[sub] = sub.stat().st_mtime
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        pass



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TestGen Copilot CLI")
    parser.add_argument("--log-level", default="info", help="Logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate tests for files")
    gen.add_argument("--file", help="Source file to analyze")
    gen.add_argument("--language", default="python", help="Source language")
    gen.add_argument("--project", help="Project directory to scan")
    gen.add_argument("--output", help="Directory to write generated tests")
    gen.add_argument("--security-scan", action="store_true", help="Run security analysis")
    gen.add_argument("--coverage-target", type=float, help="Fail if coverage below this")
    gen.add_argument("--quality-target", type=float, help="Fail if test quality below this")
    gen.add_argument("--tests-dir", default="tests", help="Location of tests relative to project")
    gen.add_argument("--show-missing", action="store_true", help="List uncovered functions")
    gen.add_argument("--no-edge-cases", action="store_true", help="Skip edge case tests")
    gen.add_argument("--no-error-tests", action="store_true", help="Skip error path tests")
    gen.add_argument("--no-benchmark-tests", action="store_true", help="Skip benchmark tests")
    gen.add_argument("--no-integration-tests", action="store_true", help="Skip integration tests")
    gen.add_argument("--config", help="Path to config file")
    gen.add_argument("--batch", action="store_true", help="Generate tests for all project files")
    gen.add_argument("--watch", help="Watch directory for changes")
    gen.add_argument("--auto-generate", action="store_true", help="Generate automatically when watching")
    gen.add_argument("--poll", type=float, default=1.0, help="Polling interval for watch mode")

    analyze = sub.add_parser("analyze", help="Analyze coverage and quality")
    analyze.add_argument("--project", required=True, help="Project directory to analyze")
    analyze.add_argument("--coverage-target", type=float, help="Coverage threshold")
    analyze.add_argument("--quality-target", type=float, help="Quality threshold")
    analyze.add_argument("--tests-dir", default="tests", help="Location of tests relative to project")
    analyze.add_argument("--show-missing", action="store_true", help="List uncovered functions")

    scaffold = sub.add_parser("scaffold", help="Create VS Code extension scaffold")
    scaffold.add_argument("directory", help="Destination directory")

    return parser


def _generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    logger.info("Starting generation")
    base = Path(args.project) if args.project else Path.cwd()
    cfg_path = Path(args.config) if args.config else base / ".testgen.config.json"
    cfg = _load_config(cfg_path)

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
    if not is_generating:
        parser.error("--file/--output or --batch with --project and --output are required")

    language = args.language if args.language != "python" else cfg.get("language", args.language)
    include_edge_cases = cfg.get("include_edge_cases", True) and not args.no_edge_cases
    include_error_paths = cfg.get("include_error_paths", True) and not args.no_error_tests
    include_benchmarks = cfg.get("include_benchmarks", True) and not args.no_benchmark_tests
    include_integration_tests = cfg.get("include_integration_tests", True) and not args.no_integration_tests

    config = GenerationConfig(
        language=language,
        include_edge_cases=include_edge_cases,
        include_error_paths=include_error_paths,
        include_benchmarks=include_benchmarks,
        include_integration_tests=include_integration_tests,
    )
    generator = TestGenerator(config)
    scanner = SecurityScanner()

    if is_watch:
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
        files = [p for p in project.rglob(pattern) if out_dir not in p.parents]
        for f in files:
            logger.info("Generating tests for %s", f)
            generator.generate_tests(f, out_dir)
        output_path = out_dir
    else:
        logger.info("Generating tests for %s", args.file)
        output_path = generator.generate_tests(args.file, args.output)

    if args.security_scan:
        target = args.project if args.project else args.file
        reports = scanner.scan_project(target) if args.project else [scanner.scan_file(target)]
        for rep in reports:
            logger.info(rep.to_text())
            print(rep.to_text())

    if args.coverage_target is not None and args.project:
        failures = _coverage_failures(args.project, args.coverage_target, args.tests_dir)
        if failures:
            logger.warning("Coverage below target:")
            print("Coverage below target:")
            for mod, cov, missing in failures:
                logger.warning("  %s: %.1f%%", mod, cov)
                print(f"  {mod}: {cov:.1f}%")
                if args.show_missing and missing:
                    names = ", ".join(sorted(missing))
                    logger.warning("    Missing: %s", names)
                    print(f"    Missing: {names}")
            parser.exit(status=1, message="Coverage target not met\n")
        logger.info("Coverage target satisfied")
        print("Coverage target satisfied")

    if args.quality_target is not None and args.project:
        tests_dir = Path(args.tests_dir)
        if not tests_dir.is_absolute():
            tests_dir = Path(args.project) / tests_dir
        scorer = TestQualityScorer()
        score = scorer.score(tests_dir)
        if score < args.quality_target:
            logger.warning("Test quality: %.1f%%", score)
            print(f"Test quality: {score:.1f}%")
            lacking = scorer.low_quality_tests(tests_dir)
            if lacking:
                names = ", ".join(sorted(lacking))
                logger.warning("  Missing asserts in: %s", names)
                print(f"  Missing asserts in: {names}")
            parser.exit(status=1, message="Quality target not met\n")
        logger.info("Quality target satisfied")
        print("Quality target satisfied")

    logger.info("Generated tests -> %s", output_path)
    print(f"Generated tests -> {output_path}")


def _analyze(args: argparse.Namespace) -> None:
    logger.info("Starting analysis")
    if args.coverage_target is None and args.quality_target is None:
        raise SystemExit("No analysis target specified")

    if args.coverage_target is not None:
        failures = _coverage_failures(args.project, args.coverage_target, args.tests_dir)
        if failures:
            logger.warning("Coverage below target:")
            print("Coverage below target:")
            for mod, cov, missing in failures:
                logger.warning("  %s: %.1f%%", mod, cov)
                print(f"  {mod}: {cov:.1f}%")
                if args.show_missing and missing:
                    names = ", ".join(sorted(missing))
                    logger.warning("    Missing: %s", names)
                    print(f"    Missing: {names}")
            raise SystemExit(1)
        logger.info("Coverage target satisfied")
        print("Coverage target satisfied")

    if args.quality_target is not None:
        tests_dir = Path(args.tests_dir)
        if not tests_dir.is_absolute():
            tests_dir = Path(args.project) / tests_dir
        scorer = TestQualityScorer()
        score = scorer.score(tests_dir)
        if score < args.quality_target:
            logger.warning("Test quality: %.1f%%", score)
            print(f"Test quality: {score:.1f}%")
            lacking = scorer.low_quality_tests(tests_dir)
            if lacking:
                names = ", ".join(sorted(lacking))
                logger.warning("  Missing asserts in: %s", names)
                print(f"  Missing asserts in: {names}")
            raise SystemExit(1)
        logger.info("Quality target satisfied")
        print("Quality target satisfied")


def _scaffold(args: argparse.Namespace) -> None:
    logger.info("Scaffolding VS Code extension")
    path = scaffold_extension(args.directory)
    logger.info("VS Code extension scaffolded at %s", path.parent)
    print(f"VS Code extension scaffolded at {path.parent}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s:%(message)s")

    if args.command == "generate":
        _generate(args, parser)
    elif args.command == "analyze":
        _analyze(args)
    elif args.command == "scaffold":
        _scaffold(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
