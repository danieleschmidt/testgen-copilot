#!/bin/bash
# Container Security Scanning Script
# Comprehensive security scanning for Docker containers

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="testgen-copilot:latest"
REPORT_DIR="./security-reports"
DOCKERFILE_PATH="./Dockerfile"
SCAN_RESULTS=""

# Help function
show_help() {
    cat << EOF
Container Security Scanner

Usage: $0 [OPTIONS]

OPTIONS:
    -i, --image NAME        Docker image name (default: $IMAGE_NAME)
    -r, --report-dir DIR    Report output directory (default: $REPORT_DIR)
    -f, --dockerfile PATH   Dockerfile path (default: $DOCKERFILE_PATH)
    -h, --help             Show this help message

EXAMPLES:
    $0 --image myapp:latest
    $0 --report-dir /tmp/reports
    $0 --dockerfile ./docker/security-scan.dockerfile

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -r|--report-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create report directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}🔍 Starting Container Security Scan${NC}"
echo -e "Image: ${YELLOW}$IMAGE_NAME${NC}"
echo -e "Report Directory: ${YELLOW}$REPORT_DIR${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run scan and capture results
run_scan() {
    local scanner="$1"
    local description="$2"
    local command="$3"
    local output_file="$4"
    
    echo -e "${BLUE}Running $description...${NC}"
    
    if command_exists "$scanner"; then
        if eval "$command" > "$output_file" 2>&1; then
            echo -e "${GREEN}✅ $description completed${NC}"
            SCAN_RESULTS="$SCAN_RESULTS\n✅ $description: $(wc -l < "$output_file") lines"
        else
            echo -e "${RED}❌ $description failed${NC}"
            SCAN_RESULTS="$SCAN_RESULTS\n❌ $description: Failed"
        fi
    else
        echo -e "${YELLOW}⚠️  $scanner not found, skipping $description${NC}"
        SCAN_RESULTS="$SCAN_RESULTS\n⚠️  $description: Skipped (tool not available)"
    fi
    echo ""
}

# Dockerfile security scan with hadolint
run_scan "hadolint" \
    "Dockerfile Security Lint" \
    "hadolint '$DOCKERFILE_PATH' --format json" \
    "$REPORT_DIR/hadolint_report.json"

# Container image vulnerability scan with trivy
run_scan "trivy" \
    "Container Vulnerability Scan" \
    "trivy image --format json --output '$REPORT_DIR/trivy_report.json' '$IMAGE_NAME'" \
    "$REPORT_DIR/trivy_scan.log"

# Container image scan with grype
run_scan "grype" \
    "Container Image Analysis" \
    "grype '$IMAGE_NAME' -o json" \
    "$REPORT_DIR/grype_report.json"

# Container configuration scan with docker-bench-security
if command_exists "docker-bench-security"; then
    echo -e "${BLUE}Running Docker Bench Security...${NC}"
    if docker run --rm --net host --pid host --userns host --cap-add audit_control \
        -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
        -v /etc:/etc:ro \
        -v /lib/systemd/system:/lib/systemd/system:ro \
        -v /usr/bin/containerd:/usr/bin/containerd:ro \
        -v /usr/bin/runc:/usr/bin/runc:ro \
        -v /usr/lib/systemd:/usr/lib/systemd:ro \
        -v /var/lib:/var/lib:ro \
        -v /var/run/docker.sock:/var/run/docker.sock:ro \
        --label docker_bench_security \
        docker/docker-bench-security > "$REPORT_DIR/docker_bench_security.log" 2>&1; then
        echo -e "${GREEN}✅ Docker Bench Security completed${NC}"
        SCAN_RESULTS="$SCAN_RESULTS\n✅ Docker Bench Security: Completed"
    else
        echo -e "${RED}❌ Docker Bench Security failed${NC}"
        SCAN_RESULTS="$SCAN_RESULTS\n❌ Docker Bench Security: Failed"
    fi
    echo ""
fi

# Container runtime security scan with clair-scanner (if available)
run_scan "clair-scanner" \
    "Container Runtime Security" \
    "clair-scanner --ip localhost --report '$REPORT_DIR/clair_report.json' '$IMAGE_NAME'" \
    "$REPORT_DIR/clair_scan.log"

# SBOM generation for container
if command_exists "syft"; then
    echo -e "${BLUE}Generating Container SBOM...${NC}"
    if syft "$IMAGE_NAME" -o json > "$REPORT_DIR/container_sbom.json" 2>&1; then
        echo -e "${GREEN}✅ Container SBOM generated${NC}"
        SCAN_RESULTS="$SCAN_RESULTS\n✅ Container SBOM: Generated"
    else
        echo -e "${RED}❌ Container SBOM generation failed${NC}"
        SCAN_RESULTS="$SCAN_RESULTS\n❌ Container SBOM: Failed"
    fi
    echo ""
fi

# Generate summary report
echo -e "${BLUE}📊 Generating Security Scan Summary...${NC}"

cat > "$REPORT_DIR/security_scan_summary.md" << EOF
# Container Security Scan Summary

**Image:** $IMAGE_NAME  
**Scan Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")  
**Report Directory:** $REPORT_DIR

## Scan Results

$(echo -e "$SCAN_RESULTS")

## Files Generated

$(ls -la "$REPORT_DIR" | awk 'NR>1 {print "- " $9 " (" $5 " bytes)"}')

## Security Recommendations

1. **Review Vulnerability Reports**: Check trivy_report.json and grype_report.json for critical vulnerabilities
2. **Dockerfile Issues**: Review hadolint_report.json for Dockerfile best practices
3. **Runtime Security**: Check docker_bench_security.log for configuration issues
4. **Supply Chain**: Validate container_sbom.json for dependency security

## Next Steps

- Address critical and high severity vulnerabilities
- Implement recommended Dockerfile improvements
- Review and apply Docker security benchmarks
- Update base images and dependencies as needed

---
*Generated by Container Security Scanner*
EOF

echo -e "${GREEN}✅ Security scan completed${NC}"
echo -e "📁 Reports saved to: ${YELLOW}$REPORT_DIR${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo -e "$SCAN_RESULTS"
echo ""
echo -e "${BLUE}📖 Full report: ${YELLOW}$REPORT_DIR/security_scan_summary.md${NC}"

exit 0