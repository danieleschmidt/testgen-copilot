#!/bin/bash
set -euo pipefail

# Quantum-Inspired Task Planner - Production Deployment Script
# Version: 1.0.0

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"production"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"quantumplanner"}
IMAGE_NAME=${IMAGE_NAME:-"quantum-task-planner"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
COMPOSE_FILE=${COMPOSE_FILE:-"docker-compose.quantum-production.yml"}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check if running as root (not recommended for production)
check_user() {
    if [ "$EUID" -eq 0 ]; then
        log_warning "Running as root. Consider using a dedicated deployment user."
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed or not in PATH"
    fi
    
    # Check disk space (minimum 5GB)
    available_space=$(df / | awk 'NR==2{print $4}')
    required_space=5242880  # 5GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        error_exit "Insufficient disk space. Required: 5GB, Available: $(($available_space/1024/1024))GB"
    fi
    
    # Check if ports are available
    if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "Port 80 is already in use"
    fi
    
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "Port 8000 is already in use"
    fi
    
    log_success "Prerequisites check completed"
}

# Create backup of current deployment
create_backup() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_info "No existing deployment found, skipping backup"
        return 0
    fi
    
    log_info "Creating backup of current deployment..."
    
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Docker Compose configuration
    cp "$COMPOSE_FILE" "$backup_dir/"
    
    # Backup database if running
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log_info "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U quantum quantum_planner > "$backup_dir/database_backup.sql"
    fi
    
    # Backup application data
    if [ -d "data" ]; then
        cp -r data "$backup_dir/"
    fi
    
    log_success "Backup created: $backup_dir"
}

# Build Docker images
build_images() {
    log_info "Building Quantum Task Planner Docker image..."
    
    # Build with build args for caching
    docker build \
        -f Dockerfile.quantum \
        -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
        -t "${DOCKER_REGISTRY}/${IMAGE_NAME}:$(date +%Y%m%d_%H%M%S)" \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
        --build-arg VERSION=${IMAGE_TAG} \
        .
    
    log_success "Docker image built successfully"
}

# Run security scan
run_security_scan() {
    log_info "Running security scan on Docker image..."
    
    # Basic security check - scan for known vulnerabilities
    if command -v docker scan &> /dev/null; then
        docker scan "${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" || log_warning "Docker scan found potential issues"
    else
        log_warning "Docker scan not available, skipping vulnerability check"
    fi
    
    # Run our custom security scanner
    if [ -f "scripts/quantum_security_scan.py" ]; then
        python3 scripts/quantum_security_scan.py || log_warning "Custom security scan found issues"
    fi
    
    log_success "Security scan completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying Quantum Task Planner services..."
    
    # Create necessary directories
    mkdir -p logs data monitoring/{prometheus,grafana,alertmanager}
    
    # Set proper permissions
    chmod 755 logs data
    
    # Pull latest images for dependencies
    docker-compose -f "$COMPOSE_FILE" pull postgres redis nginx prometheus grafana alertmanager
    
    # Deploy services
    docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans
    
    log_success "Services deployed successfully"
}

# Health check
perform_health_check() {
    log_info "Performing health checks..."
    
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log_success "Application health check passed"
            break
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -eq $max_attempts ]; then
            error_exit "Application failed health check after $max_attempts attempts"
        fi
        
        log_info "Waiting for application to be ready (attempt $attempt/$max_attempts)..."
        sleep 5
    done
    
    # Check database connection
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U quantum > /dev/null; then
        log_success "Database health check passed"
    else
        log_error "Database health check failed"
    fi
    
    # Check Redis
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q PONG; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    if [ -d "backups" ]; then
        log_info "Cleaning up old backups (retention: $BACKUP_RETENTION_DAYS days)..."
        find backups -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
        log_success "Backup cleanup completed"
    fi
}

# Display deployment information
show_deployment_info() {
    log_success "🚀 Quantum Task Planner Deployment Complete!"
    echo
    echo "📋 Deployment Information:"
    echo "  Environment: $DEPLOYMENT_ENV"
    echo "  Image: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  Compose File: $COMPOSE_FILE"
    echo
    echo "🌐 Service Endpoints:"
    echo "  Application: http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Monitoring: http://localhost:3000 (Grafana)"
    echo "  Metrics: http://localhost:9090 (Prometheus)"
    echo
    echo "📊 Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    echo "🔧 Useful Commands:"
    echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  Restart: docker-compose -f $COMPOSE_FILE restart"
    echo "  Scale API: docker-compose -f $COMPOSE_FILE up -d --scale quantum-api=3"
}

# Rollback function
rollback() {
    log_warning "Rolling back to previous deployment..."
    
    latest_backup=$(ls -1t backups/ | head -n1)
    if [ -z "$latest_backup" ]; then
        error_exit "No backup found for rollback"
    fi
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from backup
    cp "backups/$latest_backup/$COMPOSE_FILE" ./
    
    # Restore database if backup exists
    if [ -f "backups/$latest_backup/database_backup.sql" ]; then
        docker-compose -f "$COMPOSE_FILE" up -d postgres
        sleep 10
        docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U quantum -d quantum_planner < "backups/$latest_backup/database_backup.sql"
    fi
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_success "Rollback completed using backup: $latest_backup"
}

# Main deployment function
deploy() {
    log_info "🚀 Starting Quantum Task Planner Deployment"
    
    check_user
    check_prerequisites
    create_backup
    build_images
    run_security_scan
    deploy_services
    perform_health_check
    cleanup_old_backups
    show_deployment_info
    
    log_success "✅ Deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    build)
        build_images
        ;;
    health-check)
        perform_health_check
        ;;
    rollback)
        rollback
        ;;
    security-scan)
        run_security_scan
        ;;
    backup)
        create_backup
        ;;
    cleanup)
        cleanup_old_backups
        ;;
    *)
        echo "Usage: $0 {deploy|build|health-check|rollback|security-scan|backup|cleanup}"
        echo
        echo "Commands:"
        echo "  deploy        - Full deployment (default)"
        echo "  build         - Build Docker images only"
        echo "  health-check  - Run health checks"
        echo "  rollback      - Rollback to previous deployment"
        echo "  security-scan - Run security scans"
        echo "  backup        - Create backup only"
        echo "  cleanup       - Clean up old backups"
        exit 1
        ;;
esac