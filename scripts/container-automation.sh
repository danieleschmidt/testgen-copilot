#!/bin/bash
set -euo pipefail

# TestGen Copilot Container Automation Script
# Handles multi-registry publishing, security scanning, and cleanup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REGISTRY_CONFIG="$PROJECT_ROOT/.github/registry-config.yml"

# Default values
ACTION="${1:-help}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$PROJECT_ROOT}"
PUSH_TO_REGISTRY="${PUSH_TO_REGISTRY:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Check dependencies
check_dependencies() {
    local deps=("docker" "jq" "yq")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            log_error "Required dependency '$dep' not found"
            exit 1
        fi
    done
}

# Load registry configuration
load_config() {
    if [[ ! -f "$REGISTRY_CONFIG" ]]; then
        log_error "Registry config not found: $REGISTRY_CONFIG"
        exit 1
    fi
    
    log_info "Loading registry configuration from $REGISTRY_CONFIG"
}

# Build multi-platform images
build_images() {
    local dockerfile="${1:-Dockerfile}"
    local target="${2:-production}"
    local context="${3:-$BUILD_CONTEXT}"
    
    log_info "Building multi-platform images..."
    log_info "Dockerfile: $dockerfile"
    log_info "Target: $target"
    log_info "Context: $context"
    log_info "Platforms: $PLATFORMS"
    
    # Create builder if it doesn't exist
    if ! docker buildx inspect testgen-builder >/dev/null 2>&1; then
        log_info "Creating buildx builder..."
        docker buildx create --name testgen-builder --use
    fi
    
    # Build for each registry
    local registries=(
        "ghcr.io/terragonlabs/testgen-copilot"
        "docker.io/terragonlabs/testgen-copilot"
    )
    
    for registry in "${registries[@]}"; do
        local full_tag="$registry:$IMAGE_TAG"
        log_info "Building $full_tag..."
        
        docker buildx build \
            --platform "$PLATFORMS" \
            --target "$target" \
            --file "$dockerfile" \
            --tag "$full_tag" \
            --cache-from "type=gha" \
            --cache-to "type=gha,mode=max" \
            ${PUSH_TO_REGISTRY:+--push} \
            "$context"
    done
    
    log_success "Multi-platform build completed"
}

# Security scan images
scan_images() {
    log_info "Running security scans on built images..."
    
    local images=(
        "ghcr.io/terragonlabs/testgen-copilot:$IMAGE_TAG"
        "docker.io/terragonlabs/testgen-copilot:$IMAGE_TAG"
    )
    
    for image in "${images[@]}"; do
        log_info "Scanning $image with Trivy..."
        
        # Run Trivy scan
        docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v "$PROJECT_ROOT/reports:/reports" \
            aquasec/trivy:latest image \
            --format json \
            --output "/reports/trivy-$(basename "$image" | tr ':' '-').json" \
            --severity HIGH,CRITICAL \
            "$image" || {
            log_warn "Security vulnerabilities found in $image"
        }
        
        # Run Grype scan if available
        if command -v grype >/dev/null 2>&1; then
            log_info "Scanning $image with Grype..."
            grype "$image" \
                -o json \
                --file "$PROJECT_ROOT/reports/grype-$(basename "$image" | tr ':' '-').json" || {
                log_warn "Grype scan found issues in $image"
            }
        fi
    done
    
    log_success "Security scanning completed"
}

# Generate SBOM for images
generate_sbom() {
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    local images=(
        "ghcr.io/terragonlabs/testgen-copilot:$IMAGE_TAG"
    )
    
    for image in "${images[@]}"; do
        local sbom_file="$PROJECT_ROOT/reports/sbom-$(basename "$image" | tr ':' '-').json"
        
        # Generate SBOM using Syft
        if command -v syft >/dev/null 2>&1; then
            log_info "Generating SBOM for $image using Syft..."
            syft "$image" -o spdx-json > "$sbom_file"
        else
            # Fallback to Docker SBOM
            log_info "Generating SBOM for $image using Docker..."
            docker sbom "$image" --format spdx-json > "$sbom_file" || {
                log_warn "SBOM generation failed for $image"
            }
        fi
    done
    
    log_success "SBOM generation completed"
}

# Sign images with cosign
sign_images() {
    if ! command -v cosign >/dev/null 2>&1; then
        log_warn "Cosign not available, skipping image signing"
        return
    fi
    
    log_info "Signing images with cosign..."
    
    local images=(
        "ghcr.io/terragonlabs/testgen-copilot:$IMAGE_TAG"
        "docker.io/terragonlabs/testgen-copilot:$IMAGE_TAG"
    )
    
    for image in "${images[@]}"; do
        log_info "Signing $image..."
        
        # Sign with keyless mode
        cosign sign --yes "$image" || {
            log_warn "Failed to sign $image"
        }
        
        # Attach SBOM attestation
        local sbom_file="$PROJECT_ROOT/reports/sbom-$(basename "$image" | tr ':' '-').json"
        if [[ -f "$sbom_file" ]]; then
            cosign attest --yes --predicate "$sbom_file" "$image" || {
                log_warn "Failed to attach SBOM attestation to $image"
            }
        fi
    done
    
    log_success "Image signing completed"
}

# Clean up old images and tags
cleanup_registry() {
    log_info "Cleaning up old container images and tags..."
    
    # This would typically be done via registry APIs or GitHub Actions
    log_info "Registry cleanup should be configured via GitHub Actions or registry policies"
    log_info "See .github/registry-config.yml for retention settings"
}

# Verify image functionality
verify_images() {
    log_info "Verifying image functionality..."
    
    local images=(
        "ghcr.io/terragonlabs/testgen-copilot:$IMAGE_TAG"
    )
    
    for image in "${images[@]}"; do
        log_info "Testing $image..."
        
        # Test basic functionality
        docker run --rm "$image" --version || {
            log_error "Image $image failed basic functionality test"
            return 1
        }
        
        # Test help command
        docker run --rm "$image" --help >/dev/null || {
            log_error "Image $image failed help command test"
            return 1
        }
        
        log_success "$image passed verification tests"
    done
}

# Main automation pipeline
run_pipeline() {
    log_info "Starting container automation pipeline..."
    
    check_dependencies
    load_config
    
    # Create reports directory
    mkdir -p "$PROJECT_ROOT/reports"
    
    # Build images
    build_images "Dockerfile" "production"
    build_images "docker/debian.dockerfile" "production"
    build_images "docker/security-scan.dockerfile" "scanner"
    
    # Security scanning
    scan_images
    
    # Generate SBOM
    generate_sbom
    
    # Sign images (if enabled)
    if [[ "${SIGN_IMAGES:-true}" == "true" ]]; then
        sign_images
    fi
    
    # Verify images
    verify_images
    
    # Cleanup (if enabled)
    if [[ "${CLEANUP_ENABLED:-false}" == "true" ]]; then
        cleanup_registry
    fi
    
    log_success "Container automation pipeline completed successfully"
}

# Show help
show_help() {
    cat << EOF
TestGen Copilot Container Automation Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build         Build multi-platform container images
    scan          Run security scans on images
    sbom          Generate Software Bill of Materials
    sign          Sign images with cosign
    verify        Verify image functionality
    cleanup       Clean up old images and tags
    pipeline      Run complete automation pipeline
    help          Show this help message

Environment Variables:
    IMAGE_TAG           Tag for built images (default: latest)
    PLATFORMS           Target platforms (default: linux/amd64,linux/arm64)
    BUILD_CONTEXT       Build context directory (default: project root)
    PUSH_TO_REGISTRY    Push images to registry (default: false)
    SIGN_IMAGES         Enable image signing (default: true)
    CLEANUP_ENABLED     Enable registry cleanup (default: false)

Examples:
    # Build images locally
    $0 build

    # Build and push to registry
    PUSH_TO_REGISTRY=true $0 build

    # Run complete pipeline
    IMAGE_TAG=v1.0.0 PUSH_TO_REGISTRY=true $0 pipeline

    # Security scan existing images
    IMAGE_TAG=latest $0 scan

EOF
}

# Main script logic
case "$ACTION" in
    build)
        check_dependencies
        load_config
        build_images
        ;;
    scan)
        check_dependencies
        scan_images
        ;;
    sbom)
        check_dependencies
        generate_sbom
        ;;
    sign)
        check_dependencies
        sign_images
        ;;
    verify)
        check_dependencies
        verify_images
        ;;
    cleanup)
        check_dependencies
        cleanup_registry
        ;;
    pipeline)
        run_pipeline
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $ACTION"
        show_help
        exit 1
        ;;
esac