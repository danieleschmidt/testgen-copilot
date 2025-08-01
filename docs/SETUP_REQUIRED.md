# Manual Setup Requirements

This document outlines manual setup steps required after the automated SDLC implementation due to GitHub App permission limitations.

## Overview

The checkpointed SDLC implementation has successfully created all necessary configuration, documentation, and templates. However, certain operations require manual intervention by repository maintainers due to GitHub permissions.

## Required Manual Steps

### 1. GitHub Actions Workflows ⚠️ REQUIRED

**Issue**: GitHub App lacks permissions to create workflow files
**Impact**: CI/CD pipelines not active until manually implemented

**Solution**: Use the provided workflow setup script and templates

```bash
# Run the automated setup script
./scripts/setup_workflows.sh

# Or manually copy templates
mkdir -p .github/workflows
cp docs/workflow-templates/comprehensive-ci.yml .github/workflows/ci.yml
cp docs/workflow-templates/security-scan.yml .github/workflows/security.yml
cp docs/workflow-templates/dependency-update.yml .github/workflows/dependency-update.yml
cp docs/workflow-templates/release.yml .github/workflows/release.yml
```

**Files to Create**:
- `.github/workflows/ci.yml` - Main CI/CD pipeline
- `.github/workflows/security.yml` - Security scanning
- `.github/workflows/dependency-update.yml` - Automated dependency updates
- `.github/workflows/release.yml` - Release automation

**Templates Available**:
- `docs/workflow-templates/` - Complete workflow templates
- `docs/workflows/IMPLEMENTATION_GUIDE.md` - Detailed setup instructions

### 2. Repository Secrets Configuration 🔐 REQUIRED

**Location**: Settings → Secrets and variables → Actions

**Required Secrets**:
```
OPENAI_API_KEY          # OpenAI API key for testing
ANTHROPIC_API_KEY       # Anthropic API key for testing  
CODECOV_TOKEN          # Code coverage reporting
DOCKER_USERNAME        # Docker Hub credentials
DOCKER_PASSWORD        # Docker Hub token
PYPI_API_TOKEN         # PyPI publishing
```

**Optional Secrets**:
```
SLACK_WEBHOOK_URL      # Slack notifications
SONAR_TOKEN           # SonarQube analysis
SNYK_TOKEN            # Security scanning
```

### 3. Branch Protection Rules 🛡️ RECOMMENDED

**Location**: Settings → Branches → Add rule

**Branch Pattern**: `main`

**Required Settings**:
- [x] Require a pull request before merging
  - [x] Require approvals: 1
  - [x] Dismiss stale PR approvals when new commits are pushed
  - [x] Require review from code owners
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - **Status checks**: `test`, `lint`, `type-check`, `security-scan`, `docker-build`
- [x] Require conversation resolution before merging
- [x] Include administrators

### 4. Repository Settings Updates 📋 RECOMMENDED

**General Settings** (Settings → General):
- **Description**: "CLI tool and VS Code extension that uses LLMs to automatically generate comprehensive unit tests and highlight potential security vulnerabilities"
- **Website**: Your project website or documentation URL
- **Topics**: `testing`, `python`, `cli`, `automation`, `security`, `llm`, `vscode-extension`

**Features** (Settings → General → Features):
- [x] Issues
- [x] Projects  
- [x] Wiki (if desired)
- [x] Discussions (recommended for community)

**Actions** (Settings → Actions → General):
- **Actions permissions**: Allow all actions and reusable workflows
- **Artifact and log retention**: 90 days (or per policy)
- **Fork pull request workflows**: Require approval for first-time contributors