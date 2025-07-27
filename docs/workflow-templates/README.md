# GitHub Actions Workflow Templates

This directory contains comprehensive GitHub Actions workflow templates for implementing a complete SDLC automation framework. Due to GitHub App permissions, these workflows cannot be automatically installed and must be manually copied to `.github/workflows/` by repository maintainers.

## Available Workflows

### 1. `comprehensive-ci.yml` - Comprehensive CI/CD Pipeline
- **Purpose**: Complete CI/CD pipeline with multi-stage testing, security scanning, and quality gates
- **Features**:
  - Code quality and linting (Ruff, MyPy, pre-commit)
  - Security scanning (Bandit, Safety, Semgrep)
  - Testing matrix across multiple OS and Python versions
  - Coverage analysis with Codecov integration
  - Performance testing and benchmarking
  - Build and package creation
  - Docker security scanning with Trivy
  - SBOM generation
  - End-to-end testing
  - Automated release creation

### 2. `dependency-update.yml` - Automated Dependency Management
- **Purpose**: Automated dependency updates with security prioritization
- **Features**:
  - Security vulnerability scanning and updates
  - Regular dependency updates (minor/patch)
  - Pre-commit hook updates
  - GitHub Actions version updates
  - Docker base image updates
  - Automated PR creation for updates

### 3. `security-scan.yml` - Comprehensive Security Scanning
- **Purpose**: Multi-layered security scanning and compliance
- **Features**:
  - Secret detection (TruffleHog, GitLeaks)
  - Dependency vulnerability scanning (Safety, pip-audit)
  - Static code analysis (Bandit, Semgrep, PyLint)
  - Container security scanning (Trivy, Grype)
  - License compliance checking
  - SBOM generation and validation
  - Security report aggregation

### 4. `release.yml` - Automated Release Management
- **Purpose**: Automated semantic versioning and release publishing
- **Features**:
  - Semantic version determination
  - Automated changelog generation
  - Multi-platform package building
  - PyPI publishing
  - Docker image building and publishing
  - GitHub release creation with artifacts
  - Post-release documentation updates

## Installation Instructions

### Step 1: Copy Workflow Files

```bash
# Navigate to your repository root
cd your-repository

# Copy the desired workflow files
cp docs/workflow-templates/comprehensive-ci.yml .github/workflows/
cp docs/workflow-templates/dependency-update.yml .github/workflows/
cp docs/workflow-templates/security-scan.yml .github/workflows/
cp docs/workflow-templates/release.yml .github/workflows/
```

### Step 2: Configure Secrets

Add the following secrets to your GitHub repository (`Settings > Secrets and variables > Actions`):

#### Required Secrets
- `CODECOV_TOKEN` - For coverage reporting (get from codecov.io)
- `PYPI_API_TOKEN` - For publishing to PyPI (get from pypi.org)

#### Optional Secrets (for enhanced functionality)
- `GITLEAKS_LICENSE` - For GitLeaks Pro features
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `TEAMS_WEBHOOK_URL` - For Microsoft Teams notifications

### Step 3: Configure Repository Settings

#### Branch Protection Rules
Navigate to `Settings > Branches` and add protection rules for `main`:

- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Include administrators

#### Required Status Checks
Add these status checks to your branch protection:
- `Code Quality`
- `Security Scan` 
- `Test Suite`
- `Coverage Analysis`
- `Build Package`

#### Repository Permissions
Ensure the following permissions are enabled:
- Actions: Read and write permissions
- Contents: Read and write permissions
- Pull requests: Write permissions
- Issues: Write permissions
- Security events: Write permissions

### Step 4: Customize Configuration

#### Environment Variables
Update the workflows with your specific configuration:

```yaml
env:
  PYTHON_VERSION: '3.11'  # Your preferred Python version
  NODE_VERSION: '18'      # For semantic-release
```

#### Coverage Thresholds
Adjust coverage requirements in `comprehensive-ci.yml`:

```yaml
--cov-fail-under=85  # Your minimum coverage percentage
```

#### Security Scan Settings
Customize security scanning in `security-scan.yml`:

```yaml
# Add/remove security tools as needed
bandit -r src/ --severity-level medium
```

## Workflow Dependencies

### Required Files
These workflows expect the following configuration files to exist:

- `.releaserc.json` - Semantic release configuration
- `pyproject.toml` - Python project configuration with tool settings
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `Dockerfile` - Container build configuration

### Required Directory Structure
```
.github/
├── workflows/           # GitHub Actions workflows
├── ISSUE_TEMPLATE/     # Issue templates
├── PULL_REQUEST_TEMPLATE.md
└── SECURITY.md

docs/
├── DEVELOPMENT.md      # Development guide
└── runbooks/          # Operational procedures

tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── e2e/              # End-to-end tests
└── performance/      # Performance tests
```

## Troubleshooting

### Common Issues

#### 1. Workflow Permission Errors
**Error**: `refusing to allow a GitHub App to create or update workflow`
**Solution**: Workflows must be manually copied by repository maintainers with appropriate permissions.

#### 2. Missing Secrets
**Error**: `Secret CODECOV_TOKEN not found`
**Solution**: Add required secrets in repository settings.

#### 3. Failed Status Checks
**Error**: Branch protection preventing merge
**Solution**: Ensure all required status checks are passing before merge.

#### 4. Docker Build Failures
**Error**: `dockerfile parse error`
**Solution**: Ensure `Dockerfile` exists and is properly formatted.

### Getting Help

- **Documentation**: Check the development guide in `docs/DEVELOPMENT.md`
- **Issues**: Create an issue with the `ci/cd` label
- **Security**: Report security issues via `SECURITY.md` procedures

## Customization Guide

### Adding New Workflow Jobs

1. **Create new job** in appropriate workflow file
2. **Add dependencies** using `needs:` directive
3. **Configure environment** variables and secrets
4. **Add status check** to branch protection rules

### Modifying Security Scans

1. **Update scan commands** in `security-scan.yml`
2. **Adjust severity thresholds** as needed
3. **Add new security tools** following existing patterns
4. **Update report aggregation** logic

### Customizing Release Process

1. **Modify semantic release config** in `.releaserc.json`
2. **Update version bump rules** for commit types
3. **Customize release notes** format and content
4. **Add deployment steps** for your infrastructure

## Best Practices

### Workflow Maintenance
- **Regular updates**: Keep workflows updated with latest action versions
- **Performance monitoring**: Monitor workflow execution times
- **Cost optimization**: Use appropriate runner types and parallel execution
- **Security**: Regularly audit workflow permissions and secrets

### Testing Strategy
- **Test workflows** in feature branches before merging
- **Use workflow_dispatch** for manual testing
- **Monitor workflow runs** for failures and performance issues
- **Maintain test data** for consistent workflow testing

### Security Considerations
- **Principle of least privilege** for workflow permissions
- **Regular secret rotation** for API keys and tokens
- **Audit trail** for all workflow modifications
- **Secure artifact handling** for build outputs

---

🚀 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>