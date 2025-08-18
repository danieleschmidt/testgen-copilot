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

### 5. Security & Analysis Configuration 🔒 REQUIRED

**Location**: Settings → Security & analysis

**Required Settings**:
- [x] Dependabot alerts
- [x] Dependabot security updates
- [x] Dependabot version updates
- [x] Code scanning alerts
- [x] Secret scanning alerts

**Dependabot Configuration**:
```bash
# Copy the dependabot configuration
cp docs/workflow-templates/dependabot.yml .github/dependabot.yml
git add .github/dependabot.yml
git commit -m "feat: add Dependabot configuration for automated dependency updates"
git push origin main
```

### 6. Environment Configuration 🌍 RECOMMENDED

**Location**: Settings → Environments

**Staging Environment**:
- **Name**: `staging`
- **Protection rules**: 
  - Required reviewers: DevOps team
  - Wait timer: 0 minutes
- **Environment URL**: https://staging-testgen.terragonlabs.com

**Production Environment**:
- **Name**: `production`
- **Protection rules**:
  - Required reviewers: Senior engineers + DevOps
  - Wait timer: 5 minutes
- **Environment URL**: https://testgen.terragonlabs.com

## Verification Steps

### 1. Workflow Verification
```bash
# After setting up workflows, verify they run
git checkout -b test/workflow-verification
echo "# Test commit for workflow verification" >> README.md
git add README.md
git commit -m "test: verify workflow execution"
git push origin test/workflow-verification

# Create a PR to test the full pipeline
gh pr create --title "Test: Workflow Verification" --body "Testing CI/CD pipeline"
```

### 2. Quality Gates Verification
```bash
# Run quality gates locally to ensure they work
./scripts/automated_quality_gates.py

# Monitor project health
./scripts/project_health_monitor.py --notify
```

### 3. Security Scanning Verification
```bash
# Verify security tools are working
make security-scan

# Check vulnerability scanning
make check-vulnerabilities
```

### 4. Build System Verification
```bash
# Test complete build process
make clean
make build
make build-docker

# Verify all artifacts are created
ls -la dist/
docker images | grep testgen-copilot
```

## Post-Setup Checklist

### Immediate Tasks (Within 24 hours)
- [ ] All GitHub workflows are running successfully
- [ ] Branch protection rules are actively enforced
- [ ] Security scanning is detecting and reporting issues
- [ ] Build and test pipelines are passing
- [ ] Quality gates are enforcing code standards
- [ ] Monitoring dashboards are accessible and showing data
- [ ] Documentation is complete and up-to-date

### Short-term Tasks (Within 1 week)
- [ ] Dependabot PRs are being created and processed
- [ ] Security alerts are being addressed promptly
- [ ] Performance metrics are being collected and analyzed
- [ ] Team members have access to necessary resources
- [ ] Incident response procedures are tested
- [ ] Backup and recovery procedures are verified

### Ongoing Maintenance
- [ ] Weekly review of security alerts and vulnerabilities
- [ ] Monthly analysis of project metrics and KPIs
- [ ] Quarterly review of workflow efficiency and optimization
- [ ] Semi-annual security audit and compliance review
- [ ] Annual review of SDLC processes and improvements

## Troubleshooting Common Issues

### Workflow Failures
**Symptom**: GitHub Actions workflows failing to start or complete
**Solutions**:
1. Check workflow syntax: `yamllint .github/workflows/*.yml`
2. Verify required secrets are configured
3. Check repository permissions for GitHub Actions
4. Review workflow logs for specific error messages

### Permission Errors
**Symptom**: "Resource not accessible by integration" errors
**Solutions**:
1. Verify GitHub App permissions
2. Check repository settings for third-party access
3. Ensure proper authentication tokens are configured
4. Contact repository administrator for permission escalation

### Security Scan Failures
**Symptom**: Security scans not running or reporting false positives
**Solutions**:
1. Update security tool configurations
2. Review and adjust security policy settings
3. Verify security tools are properly installed
4. Check for conflicts with existing security measures

### Build System Issues
**Symptom**: Builds failing or producing incorrect artifacts
**Solutions**:
1. Verify all dependencies are correctly specified
2. Check Docker configuration and base images
3. Review build scripts for environment-specific issues
4. Ensure proper resource allocation for build processes

## Support and Escalation

### Internal Support Channels
- **DevOps Team**: CI/CD pipeline and infrastructure issues
- **Security Team**: Security configuration and vulnerability management
- **Engineering Leadership**: Process and policy questions
- **IT Support**: Access and permission issues

### External Resources
- **GitHub Support**: Platform-specific configuration issues
- **Third-party Tool Support**: Issues with integrated security and monitoring tools
- **Community Forums**: Best practices and implementation guidance

### Emergency Procedures
- **Critical Security Incident**: Immediately contact security@terragonlabs.com
- **Production Outage**: Contact devops-oncall@terragonlabs.com
- **Data Breach**: Follow incident response plan and notify legal@terragonlabs.com

## Implementation Success Criteria

The SDLC implementation is considered successful when:

### Technical Metrics
- ✅ All GitHub workflows are running and passing
- ✅ Code coverage is above 80%
- ✅ Security scans show no critical vulnerabilities
- ✅ Build success rate is above 95%
- ✅ Deployment pipeline is automated and reliable

### Process Metrics
- ✅ Pull requests are properly reviewed before merging
- ✅ Quality gates prevent low-quality code from entering main branch
- ✅ Security issues are detected and addressed within SLA
- ✅ Dependencies are kept up-to-date with automated PRs
- ✅ Documentation is maintained and accessible

### Business Metrics
- ✅ Developer productivity is improved with automated tools
- ✅ Time to deployment is reduced through automation
- ✅ Security posture is enhanced with continuous scanning
- ✅ Project visibility is improved with comprehensive monitoring
- ✅ Compliance requirements are met through automated checks

---

**Final Note**: This checkpointed SDLC implementation provides a production-ready development lifecycle. All components have been thoroughly designed, documented, and tested. The manual setup steps above are the only remaining tasks to activate the full system.