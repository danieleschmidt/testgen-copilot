#!/bin/bash
set -euo pipefail

# TestGen Copilot Security Scanner Script
# Comprehensive security scanning with multiple tools

WORKSPACE="${WORKSPACE:-/workspace}"
REPORTS_DIR="${REPORTS_DIR:-$WORKSPACE/reports}"
SCAN_TARGET="${SCAN_TARGET:-$WORKSPACE}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-json}"
FAIL_ON_SEVERITY="${FAIL_ON_SEVERITY:-CRITICAL}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Initialize exit code
OVERALL_EXIT_CODE=0

log_info "Starting comprehensive security scan of $SCAN_TARGET"
log_info "Reports will be saved to $REPORTS_DIR"

# ===============================================================================
# 1. Trivy Vulnerability Scanning
# ===============================================================================
log_info "Running Trivy vulnerability scan..."

if command -v trivy >/dev/null 2>&1; then
    trivy fs \
        --format json \
        --output "$REPORTS_DIR/trivy-vulnerabilities.json" \
        --severity HIGH,CRITICAL \
        "$SCAN_TARGET" || {
        log_warn "Trivy scan completed with issues"
        OVERALL_EXIT_CODE=1
    }
    
    # Generate human-readable report
    trivy fs \
        --format table \
        --output "$REPORTS_DIR/trivy-vulnerabilities.txt" \
        --severity HIGH,CRITICAL \
        "$SCAN_TARGET" || true
        
    log_success "Trivy scan completed"
else
    log_warn "Trivy not available, skipping vulnerability scan"
fi

# ===============================================================================
# 2. Semgrep Static Analysis
# ===============================================================================
log_info "Running Semgrep static analysis..."

if command -v semgrep >/dev/null 2>&1; then
    cd "$SCAN_TARGET"
    
    # Run Semgrep with multiple rulesets
    semgrep \
        --config=auto \
        --json \
        --output="$REPORTS_DIR/semgrep-findings.json" \
        . || {
        log_warn "Semgrep scan found security issues"
        OVERALL_EXIT_CODE=1
    }
    
    # Generate summary report
    semgrep \
        --config=auto \
        --output="$REPORTS_DIR/semgrep-summary.txt" \
        . || true
        
    log_success "Semgrep analysis completed"
else
    log_warn "Semgrep not available, skipping static analysis"
fi

# ===============================================================================
# 3. Bandit Security Analysis (Python-specific)
# ===============================================================================
log_info "Running Bandit security analysis..."

if command -v bandit >/dev/null 2>&1; then
    cd "$SCAN_TARGET"
    
    # Find Python files to scan
    if find . -name "*.py" -type f | grep -q .; then
        bandit \
            -r . \
            -f json \
            -o "$REPORTS_DIR/bandit-security.json" \
            --skip B101,B601 || {
            log_warn "Bandit found security issues"
            OVERALL_EXIT_CODE=1
        }
        
        # Generate readable report
        bandit \
            -r . \
            -f txt \
            -o "$REPORTS_DIR/bandit-security.txt" \
            --skip B101,B601 || true
            
        log_success "Bandit analysis completed"
    else
        log_info "No Python files found, skipping Bandit scan"
    fi
else
    log_warn "Bandit not available, skipping Python security analysis"
fi

# ===============================================================================
# 4. Safety Dependency Vulnerability Check
# ===============================================================================
log_info "Running Safety dependency vulnerability check..."

if command -v safety >/dev/null 2>&1; then
    cd "$SCAN_TARGET"
    
    # Check for requirements files
    for req_file in requirements.txt requirements-dev.txt pyproject.toml; do
        if [[ -f "$req_file" ]]; then
            log_info "Checking dependencies in $req_file"
            
            safety check \
                --json \
                --output "$REPORTS_DIR/safety-${req_file%.*}.json" \
                --file "$req_file" || {
                log_warn "Safety found vulnerabilities in $req_file"
                OVERALL_EXIT_CODE=1
            }
        fi
    done
    
    log_success "Safety check completed"
else
    log_warn "Safety not available, skipping dependency vulnerability check"
fi

# ===============================================================================
# 5. pip-audit for Python Package Vulnerabilities
# ===============================================================================
log_info "Running pip-audit for Python package vulnerabilities..."

if command -v pip-audit >/dev/null 2>&1; then
    cd "$SCAN_TARGET"
    
    pip-audit \
        --format=json \
        --output="$REPORTS_DIR/pip-audit-vulnerabilities.json" \
        --requirement=requirements.txt || {
        log_warn "pip-audit found vulnerabilities"
        OVERALL_EXIT_CODE=1
    }
    
    log_success "pip-audit completed"
else
    log_warn "pip-audit not available, skipping Python package vulnerability check"
fi

# ===============================================================================
# 6. TestGen Copilot Native Security Scan
# ===============================================================================
log_info "Running TestGen Copilot native security scan..."

if command -v testgen >/dev/null 2>&1; then
    cd "$SCAN_TARGET"
    
    testgen analyze \
        --project . \
        --security-scan \
        --output-format json \
        --output "$REPORTS_DIR/testgen-security.json" || {
        log_warn "TestGen security scan found issues"
        OVERALL_EXIT_CODE=1
    }
    
    log_success "TestGen security scan completed"
else
    log_warn "TestGen not available, skipping native security scan"
fi

# ===============================================================================
# 7. Generate SBOM (Software Bill of Materials)
# ===============================================================================
log_info "Generating Software Bill of Materials (SBOM)..."

if command -v cyclonedx-py >/dev/null 2>&1; then
    cd "$SCAN_TARGET"
    
    cyclonedx-py \
        --output-format json \
        --output-file "$REPORTS_DIR/sbom.json" \
        . || {
        log_warn "SBOM generation completed with warnings"
    }
    
    log_success "SBOM generated"
else
    log_warn "CycloneDX not available, skipping SBOM generation"
fi

# ===============================================================================
# 8. Aggregate Results and Generate Summary
# ===============================================================================
log_info "Generating security scan summary..."

SUMMARY_FILE="$REPORTS_DIR/security-summary.json"
SUMMARY_TEXT="$REPORTS_DIR/security-summary.txt"

{
    echo "{"
    echo "  \"scan_timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
    echo "  \"scan_target\": \"$SCAN_TARGET\","
    echo "  \"reports_directory\": \"$REPORTS_DIR\","
    echo "  \"overall_status\": $([ $OVERALL_EXIT_CODE -eq 0 ] && echo '"PASS"' || echo '"FAIL"'),"
    echo "  \"exit_code\": $OVERALL_EXIT_CODE,"
    echo "  \"tools_executed\": ["
    
    TOOLS=()
    [[ -f "$REPORTS_DIR/trivy-vulnerabilities.json" ]] && TOOLS+=("\"trivy\"")
    [[ -f "$REPORTS_DIR/semgrep-findings.json" ]] && TOOLS+=("\"semgrep\"")
    [[ -f "$REPORTS_DIR/bandit-security.json" ]] && TOOLS+=("\"bandit\"")
    [[ -f "$REPORTS_DIR/safety-requirements.json" ]] && TOOLS+=("\"safety\"")
    [[ -f "$REPORTS_DIR/pip-audit-vulnerabilities.json" ]] && TOOLS+=("\"pip-audit\"")
    [[ -f "$REPORTS_DIR/testgen-security.json" ]] && TOOLS+=("\"testgen\"")
    [[ -f "$REPORTS_DIR/sbom.json" ]] && TOOLS+=("\"cyclonedx\"")
    
    IFS=','
    echo "    ${TOOLS[*]}"
    echo "  ],"
    echo "  \"reports\": {"
    echo "    \"trivy\": \"trivy-vulnerabilities.json\","
    echo "    \"semgrep\": \"semgrep-findings.json\","
    echo "    \"bandit\": \"bandit-security.json\","
    echo "    \"safety\": \"safety-requirements.json\","
    echo "    \"pip_audit\": \"pip-audit-vulnerabilities.json\","
    echo "    \"testgen\": \"testgen-security.json\","
    echo "    \"sbom\": \"sbom.json\""
    echo "  }"
    echo "}"
} > "$SUMMARY_FILE"

# Generate human-readable summary
{
    echo "==============================================================================="
    echo "                    TESTGEN COPILOT SECURITY SCAN SUMMARY"
    echo "==============================================================================="
    echo
    echo "Scan Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "Scan Target: $SCAN_TARGET"
    echo "Reports Directory: $REPORTS_DIR"
    echo "Overall Status: $([ $OVERALL_EXIT_CODE -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
    echo
    echo "Tools Executed:"
    for tool in "${TOOLS[@]}"; do
        echo "  - ${tool//\"/}"
    done
    echo
    echo "Report Files Generated:"
    find "$REPORTS_DIR" -name "*.json" -o -name "*.txt" | sort | while read -r file; do
        echo "  - $(basename "$file")"
    done
    echo
    echo "==============================================================================="
    
    if [[ $OVERALL_EXIT_CODE -ne 0 ]]; then
        echo "⚠️  Security issues found. Review the detailed reports above."
    else
        echo "✅ No critical security issues detected."
    fi
    echo "==============================================================================="
} > "$SUMMARY_TEXT"

# Display summary
cat "$SUMMARY_TEXT"

# Exit with appropriate code
if [[ $OVERALL_EXIT_CODE -eq 0 ]]; then
    log_success "Security scan completed successfully - no critical issues found"
else
    log_error "Security scan completed with issues - review reports for details"
fi

exit $OVERALL_EXIT_CODE