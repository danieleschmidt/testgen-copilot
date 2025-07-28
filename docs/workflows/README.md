# Workflow Requirements

## Manual Setup Required

Due to security restrictions, the following GitHub workflows require manual setup:

### Required Workflows
• **CI/CD Pipeline** - `.github/workflows/ci.yml`
• **Security Scanning** - `.github/workflows/security.yml`  
• **Dependency Updates** - `.github/workflows/dependabot.yml`
• **Release Automation** - `.github/workflows/release.yml`

### Templates Available
Workflow templates are available in `docs/workflow-templates/`:
• Copy templates to `.github/workflows/`
• Configure secrets in repository settings
• Enable branch protection rules

### Repository Settings
• Enable branch protection for `main`
• Require status checks before merging
• Enable security alerts and dependency updates
• Configure repository topics and description

### Secrets Configuration
Add these secrets in GitHub repository settings:
• `PYPI_TOKEN` - For package publishing
• `CODECOV_TOKEN` - For coverage reporting

### References
• [GitHub Actions Documentation](https://docs.github.com/en/actions)
• [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
• [Repository Security Settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)