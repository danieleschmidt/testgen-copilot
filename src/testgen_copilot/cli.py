"""Command line interface for TestGen Copilot."""

from __future__ import annotations

import argparse

from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner
from .vscode import scaffold_extension


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

    args = parser.parse_args(argv)

    if args.scaffold_vscode:
        path = scaffold_extension(args.scaffold_vscode)
        print(f"VS Code extension scaffolded at {path.parent}")
        return

    if not args.file or not args.output:
        parser.error(
            "--file and --output are required unless --scaffold-vscode is used"
        )

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
    print(f"Generated tests -> {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
