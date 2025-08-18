# GitHub Workflows Documentation

## Overview

This document provides comprehensive guidance for setting up GitHub workflows for TestGen-Copilot. Due to GitHub App permission limitations, all workflow files must be manually created by repository maintainers.

## Manual Setup Required

⚠️ **IMPORTANT**: GitHub workflows cannot be automatically created due to security restrictions. Repository maintainers must manually implement these workflows using the provided templates.

## Required Workflows

### 1. Continuous Integration (CI)
**File**: `.github/workflows/ci.yml`
**Purpose**: Automated testing, linting, and quality checks on all PRs and pushes

**Key Features**:
- Multi-Python version testing (3.8, 3.9, 3.10, 3.11, 3.12)
- Cross-platform testing (Ubuntu, macOS, Windows)
- Code quality checks (ruff, black, mypy)
- Security scanning (bandit, safety)
- Test coverage reporting
- Dependency caching for faster builds

### 2. Security Scanning
**File**: `.github/workflows/security.yml`
**Purpose**: Comprehensive security analysis and vulnerability detection

**Key Features**:
- SAST (Static Application Security Testing)
- Dependency vulnerability scanning
- Container security scanning
- SBOM (Software Bill of Materials) generation
- Security advisories integration
- CodeQL analysis

### 3. Release Automation
**File**: `.github/workflows/release.yml`
**Purpose**: Automated package building, testing, and publishing

**Key Features**:
- Semantic version management
- Automated changelog generation
- PyPI package publishing
- Docker image building and pushing
- GitHub release creation
- Asset upload (wheels, source distributions)

### 4. Dependency Management
**File**: `.github/workflows/dependency-update.yml`
**Purpose**: Automated dependency updates and security patches

**Key Features**:
- Dependabot integration
- Automated PR creation for updates
- Security patch prioritization
- License compliance checking
- Breaking change detection

### 5. Performance Monitoring
**File**: `.github/workflows/performance.yml`
**Purpose**: Performance regression detection and benchmarking

**Key Features**:
- Automated performance testing
- Benchmark comparison with main branch
- Performance metrics collection
- Regression alerts
- Memory profiling

### 6. Documentation Updates
**File**: `.github/workflows/docs.yml`
**Purpose**: Automated documentation building and deployment

**Key Features**:
- Sphinx documentation building
- API documentation generation
- Documentation deployment to GitHub Pages
- Link checking and validation
- Multi-format output (HTML, PDF)

## Template Locations

All workflow templates are available in `docs/workflow-templates/` directory:

```
docs/workflow-templates/
├── ci.yml                    # Main CI/CD pipeline
├── security-scan.yml         # Security scanning workflow
├── release.yml              # Release automation
├── dependency-update.yml    # Dependency management
├── performance-monitoring.yml # Performance testing
├── docs-build.yml           # Documentation builds
├── container-security.yml   # Container scanning
└── supply-chain-security.yml # Supply chain security
```

## Setup Instructions

### Step 1: Copy Workflow Files
```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy all workflow templates
cp docs/workflow-templates/*.yml .github/workflows/

# Commit the workflow files
git add .github/workflows/
git commit -m "Add GitHub workflows for CI/CD automation"
git push origin main
```

### Step 2: Configure Repository Secrets

Add the following secrets in **Repository Settings > Secrets and variables > Actions**:

#### Required Secrets
- `PYPI_TOKEN` - PyPI API token for package publishing
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or access token

#### Optional Secrets (for enhanced features)
- `SLACK_WEBHOOK` - Slack webhook for notifications
- `HONEYCOMB_API_KEY` - Honeycomb API key for observability
- `SENTRY_DSN` - Sentry DSN for error tracking
- `SONAR_TOKEN` - SonarCloud token for code quality

### Step 3: Configure Repository Settings

#### Branch Protection Rules
Navigate to **Settings > Branches** and configure:

1. **Protect main branch**:
   - Require a pull request before merging
   - Require status checks to pass before merging
   - Required status checks:
     - `test (3.8, ubuntu-latest)`
     - `test (3.9, ubuntu-latest)` 
     - `test (3.10, ubuntu-latest)`
     - `test (3.11, ubuntu-latest)`
     - `test (3.12, ubuntu-latest)`
     - `lint`
     - `security-scan`
     - `type-check`
   - Require up-to-date branches before merging
   - Require signed commits (recommended)
   - Include administrators

2. **Additional Protection**:
   - Restrict pushes that create files
   - Restrict force pushes
   - Allow deletions: ❌

#### Security Settings
Navigate to **Settings > Security & analysis** and enable:

- Dependabot alerts
- Dependabot security updates  
- Dependabot version updates
- Code scanning alerts
- Secret scanning alerts
- Private repository forking (disable)

#### Repository Topics
Add relevant topics in **Repository Settings**:
```
python, testing, security, ai, llm, automation, cli, vscode-extension, 
quantum-computing, devops, ci-cd, monitoring, observability
```

### Step 4: Configure Dependabot

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "maintainer-team"
    assignees:
      - "lead-maintainer"
    labels:
      - "dependencies"
      - "python"
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "docker"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    labels:
      - "dependencies"
      - "github-actions"
```

## Workflow Configurations

### Environment Variables

Set the following environment variables in workflow files:

```yaml
env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"
  TESTGEN_ENV: "ci"
  CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
  FORCE_COLOR: "1"
  PIP_DISABLE_PIP_VERSION_CHECK: "1"
  PIP_NO_CACHE_DIR: "1"
```

### Caching Strategy

Configure caching for dependencies:

```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache pre-commit
  uses: actions/cache@v3
  with:
    path: ~/.cache/pre-commit
    key: ${{ runner.os }}-pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

### Matrix Testing

Configure comprehensive testing matrix:

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
    os: [ubuntu-latest, macos-latest, windows-latest]
    include:
      - python-version: "3.11"
        os: ubuntu-latest
        coverage: true
    exclude:
      - python-version: "3.12"
        os: macos-latest  # Exclude if not yet supported
```

## Quality Gates

### Required Checks
All workflows must pass these quality gates:

1. **Code Quality**
   - Linting with ruff (no errors)
   - Formatting with black (properly formatted)
   - Type checking with mypy (no type errors)
   - Import sorting with isort (properly sorted)

2. **Security**
   - SAST scanning with bandit (no high-severity issues)
   - Dependency scanning with safety (no known vulnerabilities)
   - Secret scanning (no secrets detected)
   - Container scanning (no critical vulnerabilities)

3. **Testing**
   - Unit tests (90%+ pass rate)
   - Integration tests (95%+ pass rate)
   - Coverage threshold (80%+ line coverage)
   - Performance benchmarks (no regressions > 10%)

4. **Documentation**
   - Documentation builds successfully
   - No broken links
   - API documentation up-to-date
   - Changelog updated for releases

### Failure Handling

Configure appropriate failure handling:

```yaml
- name: Upload test results
  uses: actions/upload-artifact@v3
  if: failure()
  with:
    name: test-results-${{ matrix.python-version }}-${{ matrix.os }}
    path: |
      test-results/
      htmlcov/
      .coverage

- name: Comment PR with results
  uses: actions/github-script@v6
  if: failure() && github.event_name == 'pull_request'
  with:
    script: |
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: '❌ CI failed. Check the [workflow run](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}) for details.'
      })
```

## Monitoring and Alerts

### Workflow Monitoring

Set up monitoring for workflow health:

1. **GitHub Insights**
   - Monitor workflow success rates
   - Track build duration trends
   - Analyze failure patterns

2. **External Monitoring**
   - Slack notifications for failures
   - Email alerts for security issues
   - Dashboard integration

3. **Performance Tracking**
   - Build time metrics
   - Test execution time
   - Resource usage patterns

### Notification Configuration

Configure notifications for different events:

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    fields: repo,message,commit,author,action,eventName,ref,workflow

- name: Notify on success (releases only)
  if: success() && startsWith(github.ref, 'refs/tags/')
  uses: 8398a7/action-slack@v3
  with:
    status: success
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    text: "🚀 New release deployed successfully!"
```

## Best Practices

### Security Best Practices

1. **Secret Management**
   - Use GitHub secrets for sensitive data
   - Rotate secrets regularly
   - Use least-privilege access
   - Audit secret usage

2. **Workflow Security**
   - Pin action versions to specific commits
   - Avoid using third-party actions from untrusted sources
   - Use `pull_request_target` carefully
   - Validate inputs and outputs

3. **Container Security**
   - Use official base images
   - Scan containers for vulnerabilities
   - Sign container images
   - Use multi-stage builds

### Performance Best Practices

1. **Optimization**
   - Use matrix builds efficiently
   - Cache dependencies appropriately
   - Parallelize independent tasks
   - Skip unnecessary steps

2. **Resource Management**
   - Use appropriate runner sizes
   - Set reasonable timeouts
   - Clean up artifacts
   - Monitor quota usage

### Maintenance Best Practices

1. **Regular Updates**
   - Keep actions up-to-date
   - Update Python versions
   - Review and update dependencies
   - Monitor security advisories

2. **Documentation**
   - Document workflow changes
   - Maintain README files
   - Update troubleshooting guides
   - Keep examples current

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```bash
   # Check token permissions
   # Verify secret configuration
   # Ensure token hasn't expired
   ```

2. **Build Failures**
   ```bash
   # Check dependency conflicts
   # Verify environment setup
   # Review test failures
   ```

3. **Permission Errors**
   ```bash
   # Check repository permissions
   # Verify workflow permissions
   # Review security settings
   ```

### Debug Strategies

1. **Enable Debug Logging**
   ```yaml
   env:
     ACTIONS_RUNNER_DEBUG: true
     ACTIONS_STEP_DEBUG: true
   ```

2. **Use Debugging Actions**
   ```yaml
   - name: Debug environment
     run: |
       echo "Python version: $(python --version)"
       echo "Pip version: $(pip --version)"
       echo "Environment variables:"
       env | sort
   ```

3. **Local Testing**
   ```bash
   # Use act to test locally
   act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
   ```

## Resources

### Official Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

### Community Resources
- [Awesome GitHub Actions](https://github.com/sdras/awesome-actions)
- [GitHub Actions Examples](https://github.com/actions/example-workflows)
- [Security Best Practices](https://github.com/ossf/wg-best-practices-os-developers)

### Testing Tools
- [act](https://github.com/nektos/act) - Run GitHub Actions locally
- [github-actions-runner](https://github.com/actions/runner) - Self-hosted runners
- [workflow-telemetry-action](https://github.com/runforesight/workflow-telemetry-action) - Workflow monitoring

---

**Next Steps**: After setting up workflows, monitor their performance and adjust configurations based on project needs and team feedback.