# Manual Setup Requirements

This document outlines setup steps that require elevated permissions.

## GitHub Repository Settings

### Branch Protection
1. Go to Settings → Branches
2. Add rule for `main` branch:
   • Require pull request reviews
   • Require status checks
   • Restrict pushes to specific users/teams

### Security Settings
1. Enable Dependabot alerts
2. Enable secret scanning
3. Configure code scanning with CodeQL

### Repository Configuration
```yaml
# Repository settings to configure manually:
topics: ["python", "cli", "testing", "automation", "ai"]
description: "CLI tool for automated test generation and security analysis"
homepage: "https://github.com/terragonlabs/testgen-copilot"
```

## GitHub Actions Workflows

Copy workflow templates from `docs/workflow-templates/` to `.github/workflows/`:
• `ci.yml` - Continuous integration
• `security-scan.yml` - Security scanning
• `release.yml` - Automated releases

## External Integrations

### Required Services
• **Codecov** - Coverage reporting
• **PyPI** - Package distribution
• **GitHub Apps** - Dependabot, security scanning

### Optional Services  
• **Sentry** - Error monitoring
• **DataDog** - Performance monitoring

## Automation Scope
See `.automation-scope.yaml` for detailed automation boundaries.