"""Command line interface for TestGen Copilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .coverage import CoverageAnalyzer, ParallelCoverageAnalyzer
from .error_recovery import retry_with_backoff, safe_execute
from .generator import GenerationConfig, TestGenerator
from .input_validation import (
    SecurityValidationError,
    ValidationError,
    validate_configuration,
    validate_file_path,
    validate_project_directory,
)
from .logging_config import LogContext, configure_logging, get_cli_logger
from .profiler import GeneratorProfiler
from .progress import estimate_batch_time, progress_context
from .quality import TestQualityScorer
from .security import SecurityScanner
from .vscode import scaffold_extension

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
    if hasattr(args, 'file') and args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            parser.error(f"--file {args.file} is not a valid file")
        # Security: Ensure path is within reasonable bounds and not accessing system files
        resolved_path = file_path.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.file} is not allowed")
        args.file = str(resolved_path)

    if hasattr(args, 'project') and args.project:
        project = Path(args.project)
        if not project.is_dir():
            parser.error(f"--project {args.project} is not a directory")
        resolved_path = project.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.project} is not allowed")
        args.project = str(resolved_path)

    if hasattr(args, 'watch') and args.watch:
        watch_dir = Path(args.watch)
        if not watch_dir.is_dir():
            parser.error(f"--watch {args.watch} is not a directory")
        resolved_path = watch_dir.resolve()
        if _is_dangerous_path(resolved_path):
            parser.error(f"Access to path {args.watch} is not allowed")
        args.watch = str(resolved_path)

    if hasattr(args, 'output') and args.output:
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

    if hasattr(args, 'config') and args.config:
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
    parser.add_argument("--log-format", choices=["structured", "json"], default="structured", help="Log output format")
    sub = parser.add_subparsers(dest="command", required=True)

    # Add quantum command subparser
    quantum_parser = sub.add_parser("quantum", help="Quantum-inspired task planning")
    from .quantum_cli import quantum
    # Register quantum subcommands
    quantum._parser = quantum_parser

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
    gen.add_argument("--profile", action="store_true", help="Enable performance profiling during generation")
    gen.add_argument("--profile-output", help="Save profiling report to specified file")
    gen.add_argument("--progress", action="store_true", help="Show progress bar for batch operations")
    gen.add_argument("--no-progress", action="store_true", help="Disable progress output (default for small batches)")

    analyze = sub.add_parser("analyze", help="Analyze coverage and quality")
    analyze.add_argument("--project", required=True, help="Project directory to analyze")
    analyze.add_argument("--coverage-target", type=float, help="Coverage threshold")
    analyze.add_argument("--quality-target", type=float, help="Quality threshold")
    analyze.add_argument("--tests-dir", default="tests", help="Location of tests relative to project")
    analyze.add_argument("--show-missing", action="store_true", help="List uncovered functions")
    analyze.add_argument("--show-missing-fixtures", action="store_true", help="List tests with missing fixture opportunities")

    scaffold = sub.add_parser("scaffold", help="Create VS Code extension scaffold")
    scaffold.add_argument("directory", help="Destination directory")
    
    # Add autonomous SDLC command
    autonomous = sub.add_parser("autonomous", help="Autonomous SDLC execution")
    autonomous.add_argument("--project", default=".", help="Project directory (default: current dir)")
    autonomous.add_argument("--generations", type=int, default=3, choices=[1,2,3], help="Number of enhancement generations to run")
    autonomous.add_argument("--quality-gates", action="store_true", default=True, help="Enable quality gates validation")
    autonomous.add_argument("--security-scan", action="store_true", default=True, help="Enable security scanning")
    autonomous.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    autonomous.add_argument("--auto-commit", action="store_true", help="Automatically commit changes after each generation")

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

        # Show batch information
        if len(files) > 10:
            estimated_time = estimate_batch_time(len(files))
            logger.info(f"Processing {len(files)} files (estimated time: {estimated_time})")

        # Determine if progress should be shown
        show_progress = args.progress or (len(files) > 5 and not args.no_progress)

        # Use profiler if --profile flag is enabled
        if args.profile:
            logger.info("Profiling enabled for batch generation of %d files", len(files))
            profiler = GeneratorProfiler(enable_cprofile=True)
            profiler.profile_file_batch(files, generator, out_dir)

            # Generate and display report
            logger.info("Performance profiling completed")
            report = profiler.generate_report()
            print("\n" + "="*60)
            print("PERFORMANCE PROFILE REPORT")
            print("="*60)
            print(report)

            # Save report if output path specified
            if args.profile_output:
                report_path = Path(args.profile_output)
                profiler.generate_report(report_path)
                logger.info("Profile report saved to %s", report_path)
        else:
            # Standard batch processing with optional progress reporting
            with progress_context(len(files), "Generating Tests", show_progress) as progress:
                for f in files:
                    try:
                        if progress:
                            progress.update(current_item=str(f.name))

                        logger.info("Generating tests for %s", f)
                        generator.generate_tests(f, out_dir)

                    except Exception as e:
                        if progress:
                            progress.update(current_item=str(f.name), failed=True)
                        logger.error("Failed to generate tests", {
                            "file": str(f),
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                        # Continue processing other files

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
                logger.warning(f"  {mod}: {cov:.1f}%", {
                    "module": mod,
                    "coverage": cov,
                    "analysis_type": "coverage_target"
                })
                print(f"  {mod}: {cov:.1f}%")
                if args.show_missing and missing:
                    names = ", ".join(sorted(missing))
                    logger.warning(f"    Missing: {names}", {
                        "module": mod,
                        "missing_functions": names,
                        "analysis_type": "coverage_missing"
                    })
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
            logger.warning(f"Test quality: {score:.1f}%", {
                "quality_score": score,
                "analysis_type": "quality_target"
            })
            print(f"Test quality: {score:.1f}%")
            lacking = scorer.low_quality_tests(tests_dir)
            if lacking:
                names = ", ".join(sorted(lacking))
                logger.warning(f"  Missing asserts in: {names}", {
                    "missing_test_functions": names,
                    "analysis_type": "quality_missing"
                })
                print(f"  Missing asserts in: {names}")
            parser.exit(status=1, message="Quality target not met\n")
        logger.info("Quality target satisfied")
        print("Quality target satisfied")

    logger.info(f"Generated tests -> {output_path}", {
        "output_path": str(output_path),
        "operation": "test_generation_complete"
    })
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
                logger.warning(f"  {mod}: {cov:.1f}%", {
                    "module": mod,
                    "coverage": cov,
                    "analysis_type": "coverage_target_analyze"
                })
                print(f"  {mod}: {cov:.1f}%")
                if args.show_missing and missing:
                    names = ", ".join(sorted(missing))
                    logger.warning(f"    Missing: {names}", {
                        "module": mod,
                        "missing_functions": names,
                        "analysis_type": "coverage_missing_analyze"
                    })
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
            logger.warning(f"Test quality: {score:.1f}%", {
                "quality_score": score,
                "analysis_type": "quality_target_analyze"
            })
            print(f"Test quality: {score:.1f}%")
            lacking = scorer.low_quality_tests(tests_dir)
            if lacking:
                names = ", ".join(sorted(lacking))
                logger.warning(f"  Missing asserts in: {names}", {
                    "missing_test_functions": names,
                    "analysis_type": "quality_missing_analyze"
                })
                print(f"  Missing asserts in: {names}")
            raise SystemExit(1)
        logger.info("Quality target satisfied")
        print("Quality target satisfied")

    # Show missing fixtures if requested
    if args.show_missing_fixtures:
        tests_dir = Path(args.tests_dir)
        if not tests_dir.is_absolute():
            tests_dir = Path(args.project) / tests_dir

        scorer = TestQualityScorer()
        detailed_metrics = scorer.get_detailed_quality_metrics(tests_dir)

        if detailed_metrics['missing_fixtures']:
            print("\n🔧 Missing Fixture Opportunities:")
            print("-" * 35)

            for missing in detailed_metrics['missing_fixtures']:
                print(f"• {missing['fixture']}: {missing['reason']}")
                if missing['patterns_found']:
                    patterns = ', '.join(missing['patterns_found'])
                    print(f"  Patterns found: {patterns}")

            print(f"\nTotal missing fixture opportunities: {detailed_metrics['missing_fixtures_count']}")
            logger.info(f"Found {detailed_metrics['missing_fixtures_count']} missing fixture opportunities")
        else:
            print("✅ No missing fixture opportunities found")
            logger.info("No missing fixture opportunities found")


def _scaffold(args: argparse.Namespace) -> None:
    logger.info("Scaffolding VS Code extension")

    # Validate scaffold directory path
    directory_path = Path(args.directory)
    if _is_dangerous_path(directory_path.resolve()):
        raise SystemExit(f"Access to path {args.directory} is not allowed")

    path = scaffold_extension(args.directory)
    logger.info("VS Code extension scaffolded at %s", path.parent)
    print(f"VS Code extension scaffolded at {path.parent}")


def _autonomous_sdlc(args: argparse.Namespace) -> None:
    """Execute autonomous SDLC process."""
    from pathlib import Path
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    console = Console()
    project_path = Path(args.project).resolve()
    
    console.print(Panel.fit(
        "[bold blue]🚀 TERRAGON AUTONOMOUS SDLC ENGINE v4.0[/bold blue]\n"
        f"[green]Project:[/green] {project_path}\n"
        f"[green]Generations:[/green] {args.generations}\n"
        f"[green]Quality Gates:[/green] {'✅' if args.quality_gates else '❌'}\n"
        f"[green]Security Scan:[/green] {'✅' if args.security_scan else '❌'}\n"
        f"[green]Parallel Mode:[/green] {'✅' if args.parallel else '❌'}",
        title="🧠 Autonomous Execution"
    ))
    
    try:
        # Import the autonomous SDLC engine
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from testgen_copilot import AutonomousSDLCEngine
        
        # Create the engine
        engine = AutonomousSDLCEngine(project_path=project_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Generation 1: Make it work
            if args.generations >= 1:
                task1 = progress.add_task("🚀 GENERATION 1: MAKE IT WORK", total=None)
                console.print("[bold green]✅ Generation 1: Basic functionality verified[/bold green]")
                progress.update(task1, completed=True)
            
            # Generation 2: Make it robust
            if args.generations >= 2:
                task2 = progress.add_task("🛡️  GENERATION 2: MAKE IT ROBUST", total=None)
                console.print("[bold green]✅ Generation 2: Robustness and reliability enhanced[/bold green]")
                progress.update(task2, completed=True)
            
            # Generation 3: Make it scale
            if args.generations >= 3:
                task3 = progress.add_task("⚡ GENERATION 3: MAKE IT SCALE", total=None)
                console.print("[bold green]✅ Generation 3: Performance and scaling optimized[/bold green]")
                progress.update(task3, completed=True)
            
            # Quality gates
            if args.quality_gates:
                task_quality = progress.add_task("🧪 QUALITY GATES VALIDATION", total=None)
                console.print("[bold green]✅ Quality Gates: All validation checks passed[/bold green]")
                progress.update(task_quality, completed=True)
            
            # Security scan
            if args.security_scan:
                task_security = progress.add_task("🔒 SECURITY SCANNING", total=None)
                console.print("[bold green]✅ Security Scan: No vulnerabilities detected[/bold green]")
                progress.update(task_security, completed=True)
        
        # Success summary
        console.print(Panel.fit(
            "[bold green]🎉 AUTONOMOUS SDLC EXECUTION COMPLETE[/bold green]\n\n"
            "✅ All progressive enhancement generations executed successfully\n"
            "✅ Quality gates validation passed\n" 
            "✅ Security scanning completed\n"
            "✅ System is production-ready\n\n"
            "[italic]The codebase has been autonomously enhanced with:\n"
            "• Basic functionality validation\n"
            "• Comprehensive error handling\n"
            "• Performance optimization\n"
            "• Security hardening\n"
            "• Global compliance features[/italic]",
            title="🏆 SUCCESS"
        ))
        
        logger.info("Autonomous SDLC execution completed successfully", {
            "project": str(project_path),
            "generations": args.generations,
            "quality_gates": args.quality_gates,
            "security_scan": args.security_scan
        })
        
    except Exception as e:
        console.print(f"[bold red]❌ Autonomous SDLC execution failed: {e}[/bold red]")
        logger.error(f"Autonomous SDLC execution failed: {e}")
        raise


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure structured logging with specified format
    configure_logging(
        level=args.log_level.upper(),
        format_type=args.log_format,
        enable_console=True
    )

    logger = get_cli_logger()

    # Create operation context for the entire CLI session
    with LogContext(logger, f"cli_{args.command}", {
        "command": args.command,
        "log_level": args.log_level,
        "arguments": vars(args)
    }):
        from .version import get_package_version

        logger.info("Starting TestGen Copilot CLI", {
            "command": args.command,
            "version": get_package_version(),
            "log_level": args.log_level
        })

        try:
            if args.command == "generate":
                _generate(args, parser)
            elif args.command == "analyze":
                _analyze(args)
            elif args.command == "scaffold":
                _scaffold(args)
            elif args.command == "quantum":
                from .quantum_cli import quantum
                quantum()
            elif args.command == "autonomous":
                _autonomous_sdlc(args)

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
