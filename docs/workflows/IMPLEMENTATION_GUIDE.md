# GitHub Workflows Implementation Guide

This guide provides step-by-step instructions for implementing the CI/CD workflows for TestGen-Copilot.

## Overview

Due to GitHub App permission limitations, workflows must be manually created by repository maintainers. This guide provides all necessary templates and configuration.

## Required Permissions

Before implementing workflows, ensure the GitHub repository has:

### Repository Settings
- [ ] Actions enabled
- [ ] Allow GitHub Actions to create and approve pull requests
- [ ] Allow GitHub Actions to submit approving pull request reviews

### Secrets Required
Navigate to Settings → Secrets and variables → Actions:

```bash
# Required secrets:
OPENAI_API_KEY          # OpenAI API key for testing
ANTHROPIC_API_KEY       # Anthropic API key for testing
CODECOV_TOKEN          # Code coverage reporting
DOCKER_USERNAME        # Docker Hub username
DOCKER_PASSWORD        # Docker Hub password/token
PYPI_API_TOKEN         # PyPI publishing token
SLACK_WEBHOOK_URL      # Slack notifications (optional)

# Optional secrets:
SONAR_TOKEN           # SonarQube analysis
SNYK_TOKEN            # Security scanning
```

### Environment Variables
```bash
# Repository variables:
DOCKER_REGISTRY       # Docker registry URL (default: docker.io)
IMAGE_NAME            # Docker image name (default: testgen-copilot)
PYTHON_VERSION        # Python version matrix (default: "3.8,3.9,3.10,3.11")
NODE_VERSION          # Node.js version (default: "18")
```

## Workflow Implementation Steps

### Step 1: Create .github/workflows Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Implement Core CI/CD Workflow

Copy the template from `docs/workflow-templates/comprehensive-ci.yml`:

**File: `.github/workflows/ci.yml`**

This workflow provides:
- Multi-platform testing (Linux, macOS, Windows)
- Multi-Python version testing (3.8, 3.9, 3.10, 3.11)
- Code quality checks (linting, type checking, security)
- Test execution with coverage reporting
- Docker image building and scanning
- Dependency vulnerability scanning

**Triggers:**
- Pull requests to main branch
- Push to main branch
- Manual workflow dispatch

### Step 3: Implement Security Scanning

**File: `.github/workflows/security.yml`**

Copy from `docs/workflow-templates/security-scan.yml`:

Features:
- SAST with CodeQL
- Dependency scanning with Snyk
- Container image scanning
- SBOM generation
- Security advisory checks

**Schedule:** Daily at 2 AM UTC

### Step 4: Implement Dependency Management

**File: `.github/workflows/dependency-update.yml`**

Copy from `docs/workflow-templates/dependency-update.yml`:

Features:
- Automated dependency updates
- Security vulnerability patches
- Automated testing of updates
- Pull request creation

**Schedule:** Weekly on Mondays

### Step 5: Implement Release Automation

**File: `.github/workflows/release.yml`**

Copy from `docs/workflow-templates/release.yml`:

Features:
- Semantic version bumping
- Changelog generation
- PyPI package publishing
- Docker image publishing
- GitHub release creation

**Triggers:**
- Push of version tags (v*.*.*)
- Manual workflow dispatch

### Step 6: Configure Branch Protection

Navigate to Settings → Branches → Add rule:

**Branch name pattern:** `main`

**Protect matching branches:**
- [ ] Require a pull request before merging
  - [ ] Require approvals: 1
  - [ ] Dismiss stale PR approvals when new commits are pushed
  - [ ] Require review from code owners
- [ ] Require status checks to pass before merging
  - [ ] Require branches to be up to date before merging
  - **Required status checks:**
    - `test (ubuntu-latest, 3.11)`
    - `test (ubuntu-latest, 3.10)`
    - `test (ubuntu-latest, 3.9)`
    - `test (ubuntu-latest, 3.8)`
    - `lint`
    - `type-check`
    - `security-scan`
    - `docker-build`
- [ ] Require conversation resolution before merging
- [ ] Require signed commits
- [ ] Include administrators

## Workflow Templates

### 1. Comprehensive CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"

jobs:
  test:
    name: Test (${{ matrix.os }}, Python ${{ matrix.python-version }})
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11"]
        exclude:
          # Reduce matrix size for efficiency
          - os: macos-latest
            python-version: "3.8"
          - os: windows-latest
            python-version: "3.8"

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,ai]"

    - name: Run tests
      run: |
        pytest --cov=src/testgen_copilot --cov-report=xml --cov-report=term-missing
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml

  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run linting
      run: |
        ruff check . --output-format=github
        ruff format --check .

    - name: Run type checking
      run: mypy src/testgen_copilot

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit[toml] safety

    - name: Run Bandit security scan
      run: bandit -r src/testgen_copilot -f json -o bandit-report.json

    - name: Run Safety check
      run: safety check --json --output safety-report.json

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  docker:
    name: Docker Build & Scan
    runs-on: ubuntu-latest
    needs: [test, lint, security]
    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        load: true
        tags: testgen-copilot:test
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Scan Docker image
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: testgen-copilot:test
        format: sarif
        output: trivy-results.sarif

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: trivy-results.sarif
```

### 2. Security Workflow

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python, javascript

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
```

### 3. Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release'
        required: true
        type: string

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/testgen-copilot:latest
          ${{ secrets.DOCKER_USERNAME }}/testgen-copilot:${{ github.ref_name }}

    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
```

## Advanced Configuration

### Matrix Strategy Optimization

Reduce CI costs while maintaining coverage:

```yaml
strategy:
  matrix:
    include:
      # Full test suite on Ubuntu with latest Python
      - os: ubuntu-latest
        python-version: "3.11"
        test-suite: full
      
      # Basic tests on other platforms
      - os: macos-latest
        python-version: "3.11"
        test-suite: basic
      
      - os: windows-latest
        python-version: "3.11"
        test-suite: basic
      
      # Multiple Python versions on Ubuntu only
      - os: ubuntu-latest
        python-version: "3.10"
        test-suite: basic
      
      - os: ubuntu-latest
        python-version: "3.9"
        test-suite: basic
      
      - os: ubuntu-latest
        python-version: "3.8"
        test-suite: basic
```

### Conditional Steps

Skip expensive operations on draft PRs:

```yaml
- name: Run full test suite
  if: github.event.pull_request.draft == false
  run: pytest --cov=src/testgen_copilot

- name: Run basic tests (draft PR)
  if: github.event.pull_request.draft == true
  run: pytest tests/unit/ -x
```

### Caching Strategy

Optimize build times:

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pre-commit
      node_modules
    key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements*.txt', 'package-lock.json', '.pre-commit-config.yaml') }}
    restore-keys: |
      ${{ runner.os }}-deps-
```

## Monitoring and Notifications

### Slack Integration

Add Slack notifications for important events:

```yaml
- name: Notify Slack on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#ci-cd'
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Email Notifications

Configure email alerts for security issues:

```yaml
- name: Email security report
  if: steps.security-scan.outcome == 'failure'
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 587
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: Security vulnerabilities found
    body: Check the CI logs for details
    to: security@company.com
```

## Troubleshooting

### Common Issues

**Permission denied errors:**
- Check repository permissions
- Verify GITHUB_TOKEN has necessary scopes
- Review branch protection rules

**Test failures in CI but not locally:**
- Check environment variables
- Review matrix configuration
- Verify dependencies are properly cached

**Docker build failures:**
- Check Dockerfile syntax
- Verify base image availability
- Review build context size

**Security scan false positives:**
- Update security scanning rules
- Add exceptions for known safe code
- Review dependency versions

### Debug Steps

Enable debug logging:

```yaml
- name: Debug information
  run: |
    echo "Runner OS: ${{ runner.os }}"
    echo "Python version: ${{ matrix.python-version }}"
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    env | sort
```

## Maintenance

### Regular Updates

**Monthly:**
- [ ] Update action versions in workflows
- [ ] Review security scan results
- [ ] Update Python/Node versions in matrix

**Quarterly:**
- [ ] Review workflow performance metrics
- [ ] Optimize build times and costs
- [ ] Update documentation

**As Needed:**
- [ ] Add new test environments
- [ ] Update secrets and tokens
- [ ] Modify branch protection rules

### Performance Optimization

Monitor workflow execution times and optimize:

1. **Parallel execution**: Split jobs where possible
2. **Smart caching**: Cache dependencies and build artifacts
3. **Conditional execution**: Skip unnecessary steps
4. **Matrix optimization**: Balance coverage vs. cost

---

**Next Steps**: Once workflows are implemented, monitor their performance and adjust as needed. Consider implementing additional workflows for specific use cases like nightly builds or deployment to staging environments.