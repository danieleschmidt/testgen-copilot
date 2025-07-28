# Automation Boundaries

## What Was Automated
• ✅ Community documentation (CODE_OF_CONDUCT.md)
• ✅ Issue templates for bug reports and features
• ✅ Development tooling configuration
• ✅ Alternative task runners (package.json, justfile)  
• ✅ Workflow requirement documentation

## Manual Setup Required

### GitHub Repository Administration
• **Branch Protection Rules** - Requires admin access
• **Repository Settings** - Topics, homepage, description
• **Security Features** - Dependabot, secret scanning, code scanning

### GitHub Actions Workflows
• **CI/CD Pipeline** - Copy from `docs/workflow-templates/ci.yml`
• **Security Scanning** - Copy from `docs/workflow-templates/security-scan.yml`
• **Release Automation** - Copy from `docs/workflow-templates/release.yml`

### External Service Integration
• **PyPI Publishing** - Token configuration
• **Codecov Integration** - Token setup
• **Monitoring Services** - Sentry, DataDog connections

## Rationale
Automation is limited by:
• GitHub Actions workflow creation permissions
• Repository admin setting restrictions
• External service authentication requirements

## Next Steps
1. Copy workflow templates to `.github/workflows/`
2. Configure repository settings via GitHub web interface
3. Set up external service integrations
4. Test automated workflows and adjust as needed

See `docs/SETUP_REQUIRED.md` for detailed instructions.