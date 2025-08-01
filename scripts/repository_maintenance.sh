#!/bin/bash

# TestGen-Copilot Repository Maintenance Script
# Performs automated maintenance tasks to keep the repository healthy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 TestGen-Copilot Repository Maintenance${NC}"
echo "=========================================="
echo

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/reports"
METRICS_FILE="$REPORTS_DIR/maintenance_report_$(date +%Y%m%d_%H%M%S).json"

# Create reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# Initialize maintenance report
MAINTENANCE_REPORT="{
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"tasks\": {},
  \"metrics\": {},
  \"issues_found\": [],
  \"recommendations\": []
}"

# Function to update maintenance report
update_report() {
    local task="$1"
    local status="$2"
    local details="$3"
    
    # Create a temporary report update (simplified for bash)
    echo "Task: $task, Status: $status, Details: $details" >> "$REPORTS_DIR/maintenance.log"
}

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

# Change to project root
cd "$PROJECT_ROOT"

# Task 1: Clean up build artifacts and cache
log "🧹 Cleaning build artifacts and cache files"
cleanup_files() {
    local cleaned=0
    
    # Python cache files
    if find . -name "__pycache__" -type d | head -1 | grep -q .; then
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        cleaned=$((cleaned + 1))
    fi
    
    # Coverage files
    [ -f ".coverage" ] && rm -f .coverage && cleaned=$((cleaned + 1))
    [ -d "htmlcov" ] && rm -rf htmlcov && cleaned=$((cleaned + 1))
    [ -f "coverage.xml" ] && rm -f coverage.xml && cleaned=$((cleaned + 1))
    
    # Build artifacts
    [ -d "build" ] && rm -rf build && cleaned=$((cleaned + 1))
    [ -d "dist" ] && rm -rf dist && cleaned=$((cleaned + 1))
    find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Test artifacts
    [ -d ".pytest_cache" ] && rm -rf .pytest_cache && cleaned=$((cleaned + 1))
    [ -f "benchmark.json" ] && rm -f benchmark.json && cleaned=$((cleaned + 1))
    
    # Linting cache
    [ -d ".ruff_cache" ] && rm -rf .ruff_cache && cleaned=$((cleaned + 1))
    [ -d ".mypy_cache" ] && rm -rf .mypy_cache && cleaned=$((cleaned + 1))
    
    echo -e "   ${GREEN}✅ Cleaned $cleaned cache/build directories${NC}"
    update_report "cleanup" "success" "Cleaned $cleaned directories"
}

cleanup_files

# Task 2: Update and validate dependencies
log "📦 Checking and updating dependencies"
check_dependencies() {
    local issues=0
    
    # Check for outdated packages
    if command -v pip >/dev/null 2>&1; then
        echo "   Checking for outdated packages..."
        outdated=$(pip list --outdated --format=json 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")
        if [ "$outdated" -gt 0 ]; then
            echo -e "   ${YELLOW}⚠️  $outdated outdated packages found${NC}"
            issues=$((issues + 1))
        else
            echo -e "   ${GREEN}✅ All packages are up to date${NC}"
        fi
    fi
    
    # Check for security vulnerabilities
    if command -v safety >/dev/null 2>&1; then
        echo "   Checking for security vulnerabilities..."
        if safety check --json > /tmp/safety_report.json 2>/dev/null; then
            vulns=$(jq '. | length' /tmp/safety_report.json 2>/dev/null || echo "0")
            if [ "$vulns" -gt 0 ]; then
                echo -e "   ${RED}❌ $vulns security vulnerabilities found${NC}"
                issues=$((issues + 1))
            else
                echo -e "   ${GREEN}✅ No security vulnerabilities found${NC}"
            fi
        else
            echo -e "   ${YELLOW}⚠️  Could not check for security vulnerabilities${NC}"
        fi
    fi
    
    update_report "dependencies" "checked" "$issues issues found"
    return $issues
}

check_dependencies
dep_issues=$?

# Task 3: Run code quality checks
log "🔍 Running code quality checks"
quality_checks() {
    local issues=0
    
    # Linting with ruff
    if command -v ruff >/dev/null 2>&1; then
        echo "   Running linting checks..."
        if ruff check . --quiet; then
            echo -e "   ${GREEN}✅ Linting passed${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Linting issues found${NC}"
            issues=$((issues + 1))
        fi
    fi
    
    # Type checking with mypy
    if command -v mypy >/dev/null 2>&1; then
        echo "   Running type checks..."
        if mypy src/testgen_copilot --quiet; then
            echo -e "   ${GREEN}✅ Type checking passed${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Type checking issues found${NC}"
            issues=$((issues + 1))
        fi
    fi
    
    # Security scanning with bandit
    if command -v bandit >/dev/null 2>&1; then
        echo "   Running security scan..."
        if bandit -r src/testgen_copilot -q; then
            echo -e "   ${GREEN}✅ Security scan passed${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Security issues found${NC}"
            issues=$((issues + 1))
        fi
    fi
    
    update_report "quality_checks" "completed" "$issues issues found"
    return $issues
}

quality_checks
quality_issues=$?

# Task 4: Validate configuration files
log "⚙️  Validating configuration files"
validate_configs() {
    local issues=0
    
    # Validate JSON files
    for json_file in $(find . -name "*.json" -not -path "./node_modules/*" -not -path "./.git/*"); do
        if ! python -m json.tool "$json_file" >/dev/null 2>&1; then
            echo -e "   ${RED}❌ Invalid JSON: $json_file${NC}"
            issues=$((issues + 1))
        fi
    done
    
    # Validate YAML files
    for yaml_file in $(find . -name "*.yml" -o -name "*.yaml" | grep -v ".git"); do
        if command -v python >/dev/null 2>&1; then
            if ! python -c "import yaml; yaml.safe_load(open('$yaml_file'))" 2>/dev/null; then
                echo -e "   ${RED}❌ Invalid YAML: $yaml_file${NC}"
                issues=$((issues + 1))
            fi
        fi
    done
    
    # Validate Python syntax
    for py_file in $(find src/ tests/ scripts/ -name "*.py" 2>/dev/null); do
        if ! python -m py_compile "$py_file" 2>/dev/null; then
            echo -e "   ${RED}❌ Python syntax error: $py_file${NC}"
            issues=$((issues + 1))
        fi
    done
    
    if [ $issues -eq 0 ]; then
        echo -e "   ${GREEN}✅ All configuration files are valid${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $issues configuration issues found${NC}"
    fi
    
    update_report "config_validation" "completed" "$issues issues found"
    return $issues
}

validate_configs
config_issues=$?

# Task 5: Check Docker configuration
log "🐳 Validating Docker configuration"
validate_docker() {
    local issues=0
    
    if [ -f "Dockerfile" ]; then
        # Basic Dockerfile validation
        if ! grep -q "FROM" Dockerfile; then
            echo -e "   ${RED}❌ Dockerfile missing FROM instruction${NC}"
            issues=$((issues + 1))
        fi
        
        # Check for best practices
        if ! grep -q "USER" Dockerfile; then
            echo -e "   ${YELLOW}⚠️  Dockerfile doesn't specify non-root user${NC}"
            issues=$((issues + 1))
        fi
        
        if grep -q "COPY \. \." Dockerfile; then
            echo -e "   ${YELLOW}⚠️  Dockerfile copies entire context (consider .dockerignore)${NC}"
            issues=$((issues + 1))
        fi
    fi
    
    # Validate docker-compose files
    for compose_file in docker-compose*.yml; do
        if [ -f "$compose_file" ]; then
            if command -v docker-compose >/dev/null 2>&1; then
                if ! docker-compose -f "$compose_file" config >/dev/null 2>&1; then
                    echo -e "   ${RED}❌ Invalid docker-compose file: $compose_file${NC}"
                    issues=$((issues + 1))
                fi
            fi
        fi
    done
    
    if [ $issues -eq 0 ]; then
        echo -e "   ${GREEN}✅ Docker configuration is valid${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $issues Docker issues found${NC}"
    fi
    
    update_report "docker_validation" "completed" "$issues issues found"
    return $issues
}

validate_docker
docker_issues=$?

# Task 6: Check repository health
log "📊 Checking repository health"
check_repo_health() {
    local score=100
    local recommendations=()
    
    # Check for required files
    required_files=("README.md" "LICENSE" "CONTRIBUTING.md" "SECURITY.md" ".gitignore")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo -e "   ${YELLOW}⚠️  Missing required file: $file${NC}"
            score=$((score - 10))
            recommendations+=("Add $file to repository")
        fi
    done
    
    # Check for empty directories
    empty_dirs=$(find . -type d -empty -not -path "./.git/*" | wc -l)
    if [ "$empty_dirs" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠️  $empty_dirs empty directories found${NC}"
        score=$((score - 5))
        recommendations+=("Remove or populate empty directories")
    fi
    
    # Check for large files
    large_files=$(find . -type f -size +10M -not -path "./.git/*" | wc -l)
    if [ "$large_files" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠️  $large_files large files (>10MB) found${NC}"
        score=$((score - 10))
        recommendations+=("Consider using Git LFS for large files")
    fi
    
    # Check commit message format (last 10 commits)
    if command -v git >/dev/null 2>&1; then
        bad_commits=$(git log --oneline -10 | grep -v -E "^[a-f0-9]+ (feat|fix|docs|style|refactor|test|chore|ci|build|perf)(\(.+\))?: " | wc -l)
        if [ "$bad_commits" -gt 0 ]; then
            echo -e "   ${YELLOW}⚠️  $bad_commits commits don't follow conventional format${NC}"
            score=$((score - 5))
            recommendations+=("Follow conventional commit format")
        fi
    fi
    
    echo -e "   ${GREEN}📊 Repository health score: $score/100${NC}"
    
    if [ ${#recommendations[@]} -gt 0 ]; then
        echo -e "   ${BLUE}💡 Recommendations:${NC}"
        for rec in "${recommendations[@]}"; do
            echo -e "      • $rec"
        done
    fi
    
    update_report "repo_health" "completed" "Score: $score/100"
}

check_repo_health

# Task 7: Collect and update metrics
log "📈 Collecting project metrics"
collect_metrics() {
    if [ -f "$SCRIPT_DIR/collect_metrics.py" ]; then
        if python "$SCRIPT_DIR/collect_metrics.py" --output "$REPORTS_DIR/metrics_$(date +%Y%m%d).json"; then
            echo -e "   ${GREEN}✅ Metrics collected successfully${NC}"
            update_report "metrics_collection" "success" "Metrics saved"
        else
            echo -e "   ${YELLOW}⚠️  Metrics collection completed with warnings${NC}"
            update_report "metrics_collection" "partial" "Some metrics failed"
        fi
    else
        echo -e "   ${YELLOW}⚠️  Metrics collection script not found${NC}"
        update_report "metrics_collection" "skipped" "Script not found"
    fi
}

collect_metrics

# Task 8: Generate maintenance summary
log "📋 Generating maintenance summary"
total_issues=$((dep_issues + quality_issues + config_issues + docker_issues))

echo
echo -e "${BLUE}🏁 Maintenance Summary${NC}"
echo "======================"
echo -e "Dependency issues:     ${dep_issues}"
echo -e "Code quality issues:   ${quality_issues}"
echo -e "Configuration issues:  ${config_issues}"
echo -e "Docker issues:         ${docker_issues}"
echo -e "Total issues found:    ${total_issues}"
echo

# Generate recommendations based on issues found
if [ $total_issues -gt 0 ]; then
    echo -e "${YELLOW}🔧 Recommended Actions:${NC}"
    
    if [ $dep_issues -gt 0 ]; then
        echo "   • Update outdated dependencies"
        echo "   • Address security vulnerabilities"
    fi
    
    if [ $quality_issues -gt 0 ]; then
        echo "   • Fix linting and type checking issues"
        echo "   • Review security scan results"
    fi
    
    if [ $config_issues -gt 0 ]; then
        echo "   • Fix configuration file syntax errors"
        echo "   • Validate all JSON/YAML files"
    fi
    
    if [ $docker_issues -gt 0 ]; then
        echo "   • Review Docker configuration"
        echo "   • Apply Docker best practices"
    fi
    
    echo
    echo -e "${BLUE}💡 Consider running:${NC}"
    echo "   • make format (to fix formatting issues)"
    echo "   • make lint (to see detailed linting results)"
    echo "   • make test (to ensure tests still pass)"
    echo "   • make security-scan (for detailed security analysis)"
else
    echo -e "${GREEN}✅ Repository is in excellent condition!${NC}"
fi

# Create final maintenance report
FINAL_REPORT="{
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"summary\": {
    \"total_issues\": $total_issues,
    \"dependency_issues\": $dep_issues,
    \"quality_issues\": $quality_issues,
    \"config_issues\": $config_issues,
    \"docker_issues\": $docker_issues
  },
  \"status\": \"$([ $total_issues -eq 0 ] && echo "healthy" || echo "needs_attention")\",
  \"next_maintenance\": \"$(date -u -d '+1 week' +%Y-%m-%dT%H:%M:%SZ)\"
}"

echo "$FINAL_REPORT" > "$METRICS_FILE"
echo -e "${GREEN}📄 Maintenance report saved: $METRICS_FILE${NC}"

# Exit with appropriate code
if [ $total_issues -eq 0 ]; then
    exit 0
else
    exit 1
fi