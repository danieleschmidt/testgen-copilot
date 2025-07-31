# TestGen Copilot VS Code Extension

AI-powered test generation and security analysis for Python projects, integrated directly into your VS Code workflow.

## Features

### 🧪 Intelligent Test Generation
- **One-click test generation** for entire projects or individual files
- **AI-powered test cases** using GPT-4/Claude-3 for comprehensive coverage
- **Real-time suggestions** for functions that need tests
- **Code lens integration** showing test generation options inline

### 🛡️ Advanced Security Analysis
- **Multi-layer security scanning** with Bandit, Semgrep, and custom rules
- **Real-time vulnerability detection** as you type
- **Security diagnostics** integrated into VS Code's problem panel
- **OWASP compliance checking** for enterprise security standards

### 📊 Coverage Visualization
- **Interactive HTML coverage reports** in VS Code webview
- **Line-by-line coverage highlighting** directly in the editor
- **Coverage metrics** in the TestGen Explorer panel
- **Missing coverage indicators** with hover tooltips

### ⚡ Developer Experience
- **Command palette integration** for quick access to all features
- **Context menu actions** for Python files in explorer
- **Keyboard shortcuts** (Ctrl+Shift+T for tests, Ctrl+Shift+S for security)
- **TreeView explorer** showing project test health at a glance

## Installation

1. Install the TestGen Copilot Python package:
   ```bash
   pip install testgen-copilot
   ```

2. Install this VS Code extension from the marketplace or package it locally:
   ```bash
   npm install
   npm run package
   code --install-extension testgen-copilot-0.0.1.vsix
   ```

## Configuration

Configure the extension through VS Code settings:

### Core Settings
- `testgen.pythonPath` - Path to Python executable (default: "python")
- `testgen.outputDirectory` - Directory for generated test files (default: "tests")
- `testgen.aiModel` - AI model for test generation (gpt-4, gpt-3.5-turbo, claude-3)

### Feature Toggles
- `testgen.enableRealTimeSuggestions` - Enable real-time test suggestions (default: true)
- `testgen.enableCodeLens` - Show TestGen actions in code lens (default: true)
- `testgen.securityScanLevel` - Security scanning sensitivity (basic, standard, comprehensive)

## Usage

### Quick Start
1. Open a Python project in VS Code
2. Right-click on a Python file → "Generate Tests for Active File"
3. Or use Ctrl+Shift+T keyboard shortcut
4. Generated tests appear in the configured output directory

### Command Palette
Access all features via Command Palette (Ctrl+Shift+P):
- `TestGen: Generate Tests with TestGen`
- `TestGen: Generate Tests for Active File`  
- `TestGen: Run Security Scan`
- `TestGen: Show Coverage Report`
- `TestGen: Open TestGen Settings`

### TestGen Explorer
The TestGen Explorer panel (in the Activity Bar) provides:
- **Test Coverage** - Quick access to coverage reports
- **Security Scan** - One-click security analysis
- **Test Files** - Browse and manage test files
- **Files Needing Tests** - Track coverage gaps

### Real-time Features
- **Code Lens** - Generate test actions appear above function definitions
- **Diagnostics** - Security warnings and test suggestions in Problems panel
- **Coverage Highlighting** - Uncovered lines highlighted in red

## AI Model Configuration

### OpenAI Integration
Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

### Anthropic Integration  
Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Model Selection
Choose your preferred model in settings:
- **GPT-4** - Most comprehensive, best for complex test scenarios
- **GPT-3.5 Turbo** - Faster, good for simple test generation
- **Claude-3** - Excellent for security-focused test generation

## Security Features

### Multi-Layer Scanning
- **Static Analysis** - Bandit for Python security patterns
- **Dependency Scanning** - Known vulnerability detection
- **Custom Rules** - TestGen-specific security patterns
- **OWASP Compliance** - Enterprise security standards

### Security Diagnostics
Security issues appear as:
- **Errors** - Critical and high-severity issues
- **Warnings** - Medium-severity issues  
- **Information** - Low-severity suggestions
- **Hints** - Best practice recommendations

## Advanced Features

### Workspace Integration
- **Multi-root workspace** support
- **Project-specific settings** via workspace configuration
- **Team sharing** of TestGen configurations

### Performance Optimization
- **Incremental analysis** - Only scan changed files
- **Background processing** - Non-blocking operations
- **Caching** - Intelligent caching of AI responses and scan results

### Enterprise Features
- **SBOM Generation** - Software Bill of Materials
- **Compliance Reporting** - Automated compliance reports
- **Team Metrics** - Coverage and security metrics dashboards

## Troubleshooting

### Common Issues

**Extension not activating:**
- Ensure Python and testgen-copilot package are installed
- Check VS Code Output panel for error messages
- Verify Python path in settings

**Test generation failing:**
- Check AI API key configuration
- Verify internet connection for AI model access
- Review Python path and project structure

**Security scan errors:**
- Ensure all dependencies are installed
- Check file permissions for project directory
- Review scan level configuration

### Getting Help
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive guides and examples
- **Community**: Join our Discord for support and tips

## Development

### Building from Source
```bash
git clone https://github.com/terragonlabs/testgen-copilot.git
cd testgen-copilot/vscode-extension
npm install
npm run compile
```

### Contributing
We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](../LICENSE) file for details.

---

🚀 **Powered by TestGen Copilot** - Intelligent test generation and security analysis for modern Python development.