# ADR-001: Python CLI Architecture Decision

## Status
Accepted

## Context
TestGen Copilot Assistant needs to provide both a command-line interface and VS Code extension integration. We need to decide on the core architecture and technology stack.

## Decision
We will implement the core functionality as a Python CLI tool with the following architecture:

1. **Core Engine**: Python-based for robust AST parsing and LLM integration
2. **Modular Design**: Separate modules for generation, security, coverage, and quality
3. **VS Code Integration**: Language Server Protocol (LSP) bridge to Python backend
4. **Multi-language Support**: Extensible parser system for different programming languages

## Rationale

### Python as Core Language
- **AST Parsing**: Excellent built-in AST modules and third-party libraries
- **LLM Integration**: Rich ecosystem of AI/ML libraries and API clients
- **Cross-platform**: Native cross-platform support with pip distribution
- **Community**: Large developer community for open-source contributions

### Modular Architecture
- **Maintainability**: Clear separation of concerns reduces complexity
- **Testability**: Individual modules can be tested in isolation
- **Extensibility**: New features can be added without affecting existing code
- **Performance**: Selective loading of modules based on required functionality

### CLI-First Design
- **Flexibility**: Can be integrated into any development workflow
- **CI/CD Integration**: Easy integration with build pipelines
- **IDE Agnostic**: Core functionality works regardless of editor choice
- **Automation**: Scriptable for batch processing and automation

## Consequences

### Positive
- Robust foundation for complex code analysis
- Easy integration with existing Python development tools
- Strong ecosystem for testing and security analysis
- Clear upgrade path for AI/ML enhancements

### Negative
- Requires Python runtime on target systems
- Additional complexity for VS Code extension (requires Python bridge)
- Potential performance overhead for very large codebases

### Mitigation Strategies
- Provide standalone executable builds to eliminate Python dependency
- Implement caching and incremental processing for performance
- Use efficient AST parsing libraries and parallel processing where possible

## Implementation Notes
- Use `click` or `argparse` for CLI interface
- Implement plugin system for language-specific parsers
- Use `ast` module for Python parsing, language-specific parsers for others
- Structure code for easy packaging and distribution

## Related ADRs
- ADR-002: Security Rule Engine Design
- ADR-003: Test Generation Strategy
- ADR-004: VS Code Extension Architecture