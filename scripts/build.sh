#!/bin/bash
set -euo pipefail

# TestGen-Copilot Build Script
# Comprehensive build automation with quality gates and security checks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="development"
SKIP_TESTS="false"
SKIP_SECURITY="false"
DOCKER_BUILD="false"
PUSH_IMAGE="false"
CLEANUP="true"
VERBOSE="false"

# Project variables
PROJECT_NAME="testgen-copilot"
VERSION=$(python3 -c "import src.testgen_copilot.version; print(src.testgen_copilot.version.__version__)" 2>/dev/null || echo "0.0.1")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Docker variables
DOCKER_REGISTRY="docker.io"
DOCKER_NAMESPACE="terragonlabs"
DOCKER_IMAGE="${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${PROJECT_NAME}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for TestGen-Copilot with comprehensive quality gates.

OPTIONS:
    -t, --type TYPE         Build type: development|staging|production (default: development)
    -s, --skip-tests        Skip test execution
    -S, --skip-security     Skip security scans
    -d, --docker            Build Docker image
    -p, --push              Push Docker image to registry
    -c, --no-cleanup        Skip cleanup of build artifacts
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                                          # Development build
    $0 --type production --docker --push       # Production build with Docker
    $0 --skip-tests --docker                   # Quick Docker build without tests

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        -S|--skip-security)
            SKIP_SECURITY="true"
            shift
            ;;
        -d|--docker)
            DOCKER_BUILD="true"
            shift
            ;;
        -p|--push)
            PUSH_IMAGE="true"
            DOCKER_BUILD="true"
            shift
            ;;
        -c|--no-cleanup)
            CLEANUP="false"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Enable verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

print_status "Starting build for TestGen-Copilot"
print_status "Version: $VERSION"
print_status "Build Type: $BUILD_TYPE"
print_status "Git Commit: $GIT_COMMIT"
print_status "Git Branch: $GIT_BRANCH"

# Validate build type
if [[ ! "$BUILD_TYPE" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid build type: $BUILD_TYPE"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check for required tools
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v pip >/dev/null 2>&1 || missing_tools+=("pip")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    
    if [[ "$DOCKER_BUILD" == "true" ]]; then
        command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    fi
    
    if [[ ${#missing_tools[@]} -ne 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Clean previous build artifacts
cleanup_artifacts() {
    if [[ "$CLEANUP" == "true" ]]; then
        print_status "Cleaning previous build artifacts..."
        rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        print_success "Cleanup completed"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    python3 -m pip install --upgrade pip setuptools wheel
    
    case "$BUILD_TYPE" in
        "development")
            python3 -m pip install -e ".[dev,ai,security,all]"
            ;;
        "staging"|"production")
            python3 -m pip install -e ".[ai,security,api,monitoring,database]"
            ;;
    esac
    
    print_success "Dependencies installed"
}

# Run code formatting
format_code() {
    print_status "Formatting code..."
    python3 -m black src/ tests/ --line-length 100
    python3 -m isort src/ tests/
    python3 -m ruff check --fix src/ tests/
    print_success "Code formatting completed"
}

# Run linting
run_linting() {
    print_status "Running linting..."
    python3 -m ruff check src/ tests/
    python3 -m black --check src/ tests/ --line-length 100
    print_success "Linting passed"
}

# Run type checking
run_type_checking() {
    print_status "Running type checking..."
    python3 -m mypy src/testgen_copilot --config-file pyproject.toml
    print_success "Type checking passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_warning "Skipping tests"
        return 0
    fi
    
    print_status "Running tests..."
    
    case "$BUILD_TYPE" in
        "development")
            python3 -m pytest tests/ -v --tb=short -m "not slow"
            ;;
        "staging")
            python3 -m pytest tests/ -v --tb=short --cov=src/testgen_copilot --cov-report=xml
            ;;
        "production")
            python3 -m pytest tests/ -v --tb=short --cov=src/testgen_copilot --cov-report=xml --cov-fail-under=80
            ;;
    esac
    
    print_success "Tests passed"
}

# Run security scans
run_security_scans() {
    if [[ "$SKIP_SECURITY" == "true" ]]; then
        print_warning "Skipping security scans"
        return 0
    fi
    
    print_status "Running security scans..."
    
    # Bandit security linting
    python3 -m bandit -r src/ -f json -o bandit-report.json || true
    
    # Safety vulnerability check
    python3 -m safety check --json --output safety-report.json || true
    
    # Dependency audit (if available)
    if command -v pip-audit >/dev/null 2>&1; then
        python3 -m pip-audit --output=json --desc > pip-audit-report.json || true
    fi
    
    print_success "Security scans completed"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    print_status "Generating SBOM..."
    
    if command -v syft >/dev/null 2>&1; then
        syft packages dir:. -o spdx-json=sbom.spdx.json
    else
        # Fallback: generate simple dependency list
        python3 -m pip list --format=json > dependencies.json
        cat > sbom.json << EOF
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "version": 1,
  "metadata": {
    "timestamp": "${BUILD_DATE}",
    "component": {
      "type": "application",
      "name": "${PROJECT_NAME}",
      "version": "${VERSION}"
    }
  },
  "components": []
}
EOF
    fi
    
    print_success "SBOM generated"
}

# Build Python package
build_package() {
    print_status "Building Python package..."
    python3 -m build
    print_success "Package built successfully"
}

# Build Docker image
build_docker_image() {
    if [[ "$DOCKER_BUILD" != "true" ]]; then
        return 0
    fi
    
    print_status "Building Docker image..."
    
    # Determine Docker target based on build type
    local docker_target=""
    case "$BUILD_TYPE" in
        "development")
            docker_target="development"
            ;;
        "staging"|"production")
            docker_target="runtime"
            ;;
    esac
    
    # Build Docker image
    docker build \
        --target "$docker_target" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$GIT_COMMIT" \
        -t "${DOCKER_IMAGE}:${VERSION}" \
        -t "${DOCKER_IMAGE}:latest" \
        .
    
    print_success "Docker image built: ${DOCKER_IMAGE}:${VERSION}"
}

# Push Docker image
push_docker_image() {
    if [[ "$PUSH_IMAGE" != "true" ]]; then
        return 0
    fi
    
    print_status "Pushing Docker image..."
    docker push "${DOCKER_IMAGE}:${VERSION}"
    docker push "${DOCKER_IMAGE}:latest"
    print_success "Docker image pushed successfully"
}

# Generate build report
generate_build_report() {
    print_status "Generating build report..."
    
    cat > build-report.json << EOF
{
  "build": {
    "project": "${PROJECT_NAME}",
    "version": "${VERSION}",
    "type": "${BUILD_TYPE}",
    "timestamp": "${BUILD_DATE}",
    "git": {
      "commit": "${GIT_COMMIT}",
      "branch": "${GIT_BRANCH}"
    },
    "docker": {
      "built": ${DOCKER_BUILD},
      "image": "${DOCKER_IMAGE}:${VERSION}",
      "pushed": ${PUSH_IMAGE}
    },
    "quality_gates": {
      "tests_run": $([ "$SKIP_TESTS" == "false" ] && echo "true" || echo "false"),
      "security_scanned": $([ "$SKIP_SECURITY" == "false" ] && echo "true" || echo "false"),
      "linting_passed": true,
      "type_checking_passed": true
    }
  }
}
EOF
    
    print_success "Build report generated: build-report.json"
}

# Main build process
main() {
    local start_time=$(date +%s)
    
    check_prerequisites
    cleanup_artifacts
    install_dependencies
    format_code
    run_linting
    run_type_checking
    run_tests
    run_security_scans
    generate_sbom
    build_package
    build_docker_image
    push_docker_image
    generate_build_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_success "Build completed successfully in ${duration} seconds"
    print_status "Artifacts:"
    print_status "  - Python package: dist/"
    if [[ "$DOCKER_BUILD" == "true" ]]; then
        print_status "  - Docker image: ${DOCKER_IMAGE}:${VERSION}"
    fi
    print_status "  - Build report: build-report.json"
    print_status "  - SBOM: sbom.json"
    if [[ "$SKIP_SECURITY" == "false" ]]; then
        print_status "  - Security reports: *-report.json"
    fi
}

# Handle script interruption
trap 'print_error "Build interrupted"; exit 1' INT TERM

# Run main function
main "$@"