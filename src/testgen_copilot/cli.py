"""Command line interface for TestGen Copilot."""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path
import logging

from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner
from .vscode import scaffold_extension
from .coverage import CoverageAnalyzer, ParallelCoverageAnalyzer, CoverageResult
from .quality import TestQualityScorer
from .logging_config import configure_logging, get_cli_logger, LogContext


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



logger = get_cli_logger()


def _validate_config_schema(cfg: dict) -> dict:
    """Validate configuration dictionary schema."""
    allowed = {
        "language": str,
        "include_edge_cases": bool,
        "include_error_paths": bool,
        "include_benchmarks": bool,
        "include_integration_tests": bool,
    }
    
    # Check for dangerous keys that could indicate code injection attempts
    dangerous_keys = {"__import__", "eval", "exec", "open", "compile", "globals", "locals"}
    for key in cfg.keys():
        if key in dangerous_keys:
            raise ValueError(f"Dangerous config option detected: {key}")
    
    for key, value in cfg.items():
        if key not in allowed:
            raise ValueError(f"Unknown config option: {key}")
        if not isinstance(value, allowed[key]):
            raise ValueError(f"Invalid type for {key}: expected {allowed[key].__name__}, got {type(value).__name__}")
    
    # Validate specific values
    if "language" in cfg:
        valid_languages = {"python", "py", "javascript", "js", "typescript", "ts", "java", "c#", "csharp", "go", "rust"}
        if cfg["language"].lower() not in valid_languages:
            raise ValueError(f"Unsupported language: {cfg['language']}. Supported: {', '.join(sorted(valid_languages))}")
    
    return cfg


def _is_dangerous_path(path: Path) -> bool:
    """Check if path accesses dangerous system locations."""
    # Convert to string for easier checking
    path_str = str(path).lower()
    
    # Dangerous system directories
    dangerous_dirs = {
        "/etc", "/sys", "/proc", "/dev", "/boot", "/root",
        "/var/log", "/var/run", "/var/lib", "/usr/bin", "/usr/sbin",
        "/bin", "/sbin", "/lib", "/lib64"
    }
    
    # Check if path starts with any dangerous directory
    for dangerous in dangerous_dirs:
        if path_str.startswith(dangerous.lower()):
            return True
    
    # Check for path traversal attempts
    if "../" in path_str or "..\\" in path_str:
        return True
        
    return False


def _validate_numeric_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate numeric argument ranges."""
    if hasattr(args, 'coverage_target') and args.coverage_target is not None:
        if not 0 <= args.coverage_target <= 100:
            parser.error(f"--coverage-target must be between 0 and 100, got {args.coverage_target}")
    
    if hasattr(args, 'quality_target') and args.quality_target is not None:
        if not 0 <= args.quality_target <= 100:
            parser.error(f"--quality-target must be between 0 and 100, got {args.quality_target}")
    
    if hasattr(args, 'poll') and args.poll is not None:
        if args.poll <= 0:
            parser.error(f"--poll interval must be positive, got {args.poll}")
        if args.poll > 300:  # 5 minutes max
            parser.error(f"--poll interval too large (max 300 seconds), got {args.poll}")


def _language_pattern(language: str) -> str:
    """Return glob pattern for ``language`` source files."""
    return LANG_PATTERNS.get(language.lower(), f"*{language}")


def _load_config(path: Path) -> dict:
    """Load and validate configuration from ``path`` if it exists."""
    if path.exists():
        cfg = json.loads(path.read_text())
        return _validate_config_schema(cfg)
    return {}


def _validate_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate and normalize path arguments."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            parser.error(f"--file {args.file} is not a valid file")
        # Security: Ensure path is within reasonable bounds and not accessing system files
        resolved_path = file_path.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.file} is not allowed")
        args.file = str(resolved_path)
        
    if args.project:
        project = Path(args.project)
        if not project.is_dir():
            parser.error(f"--project {args.project} is not a directory")
        resolved_path = project.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.project} is not allowed")
        args.project = str(resolved_path)
        
    if args.watch:
        watch_dir = Path(args.watch)
        if not watch_dir.is_dir():
            parser.error(f"--watch {args.watch} is not a directory")
        resolved_path = watch_dir.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.watch} is not allowed")
        args.watch = str(resolved_path)
        
    if args.output:
        out_dir = Path(args.output)
        if out_dir.exists() and not out_dir.is_dir():
            parser.error(f"--output {args.output} is not a directory")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            parser.error(f"Cannot create output directory {args.output}: {e}")
        resolved_path = out_dir.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.output} is not allowed")
        args.output = str(resolved_path)
        
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            parser.error(f"Config file {args.config} not found")
        resolved_path = cfg_path.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to config file {args.config} is not allowed")
        args.config = str(resolved_path)


def _coverage_failures(
    project_dir: str | Path, target: float, tests_dir: str | Path | None = None
) -> list[tuple[str, float, set[str]]]:
    """Return modules below ``target`` with their uncovered functions using parallel processing."""
    project = Path(project_dir)
    tests_dir = Path(tests_dir) if tests_dir else Path("tests")
    if not tests_dir.is_absolute():
        tests_dir = project / tests_dir
    
    # Use parallel coverage analyzer for improved performance
    parallel_analyzer = ParallelCoverageAnalyzer()
    
    def progress_callback(completed: int, total: int):
        """Progress callback for large projects."""
        if total > 20:  # Only show progress for large projects
            logger.info(f"Coverage analysis progress: {completed}/{total} files ({100*completed/total:.1f}%)")
    
    try:
        coverage_results = parallel_analyzer.analyze_project_parallel(
            project_dir=project,
            tests_dir=tests_dir,
            target_coverage=target,
            progress_callback=progress_callback
        )
        
        # Transform CoverageResult objects to the expected format
        failures = [
            (str(Path(result.file_path).relative_to(project)), result.coverage_percentage, result.uncovered_functions)
            for result in coverage_results
        ]
        
        logger.info(f"Parallel coverage analysis found {len(failures)} files below {target}% coverage")
        return failures
        
    except Exception as e:
        logger.warning(f"Parallel coverage analysis failed, falling back to sequential: {e}")
        # Fallback to original sequential implementation
        analyzer = CoverageAnalyzer()
        failures: list[tuple[str, float, set[str]]] = []
        for path in project.rglob("*.py"):
            if tests_dir in path.parents:
                continue
            try:
                cov = analyzer.analyze(path, tests_dir)
                if cov < target:
                    missing = analyzer.uncovered_functions(path, tests_dir)
                    failures.append((str(path.relative_to(project)), cov, missing))
            except Exception as file_error:
                logger.warning(f"Failed to analyze {path}: {file_error}")
                continue
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
    logger = get_cli_logger()
    
    with LogContext(logger, "generate_command", {
        "file": args.file,
        "project": args.project,
        "output": args.output,
        "language": args.language,
        "batch": getattr(args, 'batch', False),
        "watch": getattr(args, 'watch', False)
    }):
        logger.info("Starting test generation", {
            "operation_type": "generate",
            "source": args.file or args.project,
            "output_dir": args.output
        })
    base = Path(args.project) if args.project else Path.cwd()
    cfg_path = Path(args.config) if args.config else base / ".testgen.config.json"
    try:
        cfg = _load_config(cfg_path)
    except Exception as exc:  # json or validation errors
        parser.error(str(exc))

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

    _validate_paths(args, parser)
    _validate_numeric_args(args, parser)

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
    
    # Validate paths and numeric arguments for analyze command
    parser = _build_parser()  # We need parser for error handling
    _validate_paths(args, parser)
    _validate_numeric_args(args, parser)

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
    
    # Validate scaffold directory path
    directory_path = Path(args.directory)
    if _is_dangerous_path(directory_path.resolve()):
        raise SystemExit(f"Access to path {args.directory} is not allowed")
    
    path = scaffold_extension(args.directory)
    logger.info("VS Code extension scaffolded at %s", path.parent)
    print(f"VS Code extension scaffolded at {path.parent}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure structured logging
    configure_logging(
        level=args.log_level.upper(),
        format_type="structured",
        enable_console=True
    )
    
    logger = get_cli_logger()
    
    # Create operation context for the entire CLI session
    with LogContext(logger, f"cli_{args.command}", {
        "command": args.command,
        "log_level": args.log_level,
        "arguments": vars(args)
    }):
        logger.info("Starting TestGen Copilot CLI", {
            "command": args.command,
            "version": "0.0.1",  # Could be extracted from package metadata
            "log_level": args.log_level
        })
        
        try:
            if args.command == "generate":
                _generate(args, parser)
            elif args.command == "analyze":
                _analyze(args)
            elif args.command == "scaffold":
                _scaffold(args)
            
            logger.info("CLI operation completed successfully", {
                "command": args.command
            })
            
        except Exception as e:
            logger.error("CLI operation failed", {
                "command": args.command,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
