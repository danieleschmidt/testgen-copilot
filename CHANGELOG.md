# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Added proper packaging configuration with development dependencies (pytest, pytest-cov)
- Fixed pytest collection warnings for TestGenerator class
- Added CLI script entry point for `testgen` command

### Enhanced
- Improved cross-platform timeout handling for Windows compatibility
- Enhanced `safe_parse_ast_with_timeout` with multiprocessing fallback for Windows systems
- Updated cross-platform timeout tests to handle threading limitations correctly
- Maintained Unix signal-based timeout for optimal performance on Linux/macOS
- Enhanced cross-platform memory monitoring with improved error handling
- Added comprehensive exception handling to prevent false memory limit detections
- Verified memory monitoring fallbacks work correctly across all platforms

## [0.2.0] - 2025-06-29
### Added
- GitHub Actions workflow running linting, security scans, tests, and uploading coverage artifacts
- Modular CLI with `generate`, `analyze`, and `scaffold` subcommands
- Configurable logging level and faster file watching using per-directory modification times
- Security scanner detects shell injection and insecure temp file usage
- VS Code extension commands to run security scans and show coverage
- CONTRIBUTING guidelines and CODEOWNERS file

## [0.1.0] - 2025-06-29
- Initial release with test generation CLI and VS Code extension scaffolding.
