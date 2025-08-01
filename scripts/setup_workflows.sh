#!/bin/bash

# TestGen-Copilot Workflow Setup Script
# This script helps repository maintainers set up GitHub Actions workflows

set -e

echo "🚀 TestGen-Copilot Workflow Setup"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/testgen_copilot" ]; then
    echo -e "${RED}❌ Error: This script must be run from the TestGen-Copilot repository root${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Current directory: $(pwd)${NC}"
echo

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "${GREEN}✅ Created directory: $1${NC}"
    else
        echo -e "${YELLOW}📁 Directory already exists: $1${NC}"
    fi
}

# Function to copy workflow file
copy_workflow() {
    local source="$1"
    local target="$2"
    local description="$3"
    
    if [ -f "$source" ]; then
        if [ -f "$target" ]; then
            echo -e "${YELLOW}⚠️  Workflow already exists: $target${NC}"
            read -p "   Overwrite? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}   Skipped: $target${NC}"
                return
            fi
        fi
        
        cp "$source" "$target"
        echo -e "${GREEN}✅ Copied $description: $target${NC}"
    else
        echo -e "${RED}❌ Source file not found: $source${NC}"
    fi
}

# Main setup process
echo -e "${BLUE}🔧 Setting up GitHub Actions workflows...${NC}"
echo

# Step 1: Create .github/workflows directory
echo -e "${BLUE}Step 1: Creating .github/workflows directory${NC}"
create_dir ".github"
create_dir ".github/workflows"
echo

# Step 2: Copy workflow templates
echo -e "${BLUE}Step 2: Copying workflow templates${NC}"

# Main CI workflow
copy_workflow "docs/workflow-templates/comprehensive-ci.yml" ".github/workflows/ci.yml" "Main CI/CD workflow"

# Security scanning
copy_workflow "docs/workflow-templates/security-scan.yml" ".github/workflows/security.yml" "Security scanning workflow"

# Dependency updates
copy_workflow "docs/workflow-templates/dependency-update.yml" ".github/workflows/dependency-update.yml" "Dependency update workflow"

# Release workflow
copy_workflow "docs/workflow-templates/release.yml" ".github/workflows/release.yml" "Release workflow"

echo

# Step 3: Check for required files
echo -e "${BLUE}Step 3: Checking for required configuration files${NC}"

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅ Found: $1${NC}"
    else
        echo -e "${YELLOW}⚠️  Missing: $1${NC}"
        echo -e "   ${BLUE}Consider creating this file for optimal workflow functionality${NC}"
    fi
}

check_file ".env.example"
check_file "pyproject.toml"
check_file "pytest.ini"
check_file "Dockerfile"
check_file ".dockerignore"
check_file ".pre-commit-config.yaml"

echo

# Step 4: Validation
echo -e "${BLUE}Step 4: Validating workflow syntax${NC}"

validate_workflow() {
    local workflow="$1"
    if [ -f "$workflow" ]; then
        # Basic YAML syntax check using python
        if python -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
            echo -e "${GREEN}✅ Valid YAML syntax: $workflow${NC}"
        else
            echo -e "${RED}❌ Invalid YAML syntax: $workflow${NC}"
        fi
    fi
}

if command -v python3 >/dev/null 2>&1; then
    python3 -c "import yaml" 2>/dev/null || pip install PyYAML
    
    validate_workflow ".github/workflows/ci.yml"
    validate_workflow ".github/workflows/security.yml"
    validate_workflow ".github/workflows/dependency-update.yml"
    validate_workflow ".github/workflows/release.yml"
else
    echo -e "${YELLOW}⚠️  Python not found - skipping YAML validation${NC}"
fi

echo

# Step 5: Next steps and requirements
echo -e "${BLUE}🔐 Step 5: Required Secrets and Configuration${NC}"
echo
echo -e "${YELLOW}Repository Secrets (Settings → Secrets and variables → Actions):${NC}"
echo "   • OPENAI_API_KEY - OpenAI API key for testing"
echo "   • ANTHROPIC_API_KEY - Anthropic API key for testing"
echo "   • CODECOV_TOKEN - Code coverage reporting token"
echo "   • DOCKER_USERNAME - Docker Hub username"
echo "   • DOCKER_PASSWORD - Docker Hub password/token"
echo "   • PYPI_API_TOKEN - PyPI publishing token"
echo "   • SLACK_WEBHOOK_URL - Slack notifications (optional)"
echo
echo -e "${YELLOW}Repository Variables:${NC}"
echo "   • DOCKER_REGISTRY - Docker registry URL (default: docker.io)"
echo "   • IMAGE_NAME - Docker image name (default: testgen-copilot)"
echo
echo -e "${YELLOW}Branch Protection Rules (Settings → Branches):${NC}"
echo "   • Protect 'main' branch"
echo "   • Require pull request reviews"
echo "   • Require status checks to pass"
echo "   • Require up-to-date branches"
echo

# Step 6: Testing workflows
echo -e "${BLUE}🧪 Step 6: Testing Workflows${NC}"
echo
echo "To test the workflows:"
echo "1. Commit and push the workflow files"
echo "2. Create a test pull request"
echo "3. Monitor the Actions tab for workflow execution"
echo "4. Check for any failures and adjust configuration"
echo

# Summary
echo -e "${GREEN}🎉 Workflow Setup Complete!${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review and customize the workflow files in .github/workflows/"
echo "2. Add required secrets to your GitHub repository"
echo "3. Configure branch protection rules"
echo "4. Test workflows with a pull request"
echo "5. Monitor workflow performance and adjust as needed"
echo
echo -e "${BLUE}Documentation:${NC}"
echo "• Implementation Guide: docs/workflows/IMPLEMENTATION_GUIDE.md"
echo "• Workflow Templates: docs/workflow-templates/"
echo "• GitHub Actions: https://docs.github.com/en/actions"
echo

# Optional: Git operations
if command -v git >/dev/null 2>&1; then
    echo -e "${BLUE}Git Status:${NC}"
    git status --porcelain .github/
    echo
    
    if [ -n "$(git status --porcelain .github/)" ]; then
        echo -e "${YELLOW}New workflow files detected. Consider committing them:${NC}"
        echo "  git add .github/workflows/"
        echo "  git commit -m 'ci: add GitHub Actions workflows'"
        echo "  git push"
        echo
    fi
fi

echo -e "${GREEN}✨ Happy coding!${NC}"