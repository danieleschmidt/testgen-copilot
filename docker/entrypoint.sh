#!/bin/bash
set -e

# Graceful shutdown handler for Quantum Task Planner API
# Ensures proper cleanup of resources and ongoing tasks

# Global variables
PID=0
SHUTDOWN_TIMEOUT=30
HEALTH_CHECK_INTERVAL=5

# Signal handlers
shutdown_handler() {
    echo "$(date): Received shutdown signal, initiating graceful shutdown..."
    
    if [ $PID -ne 0 ]; then
        echo "$(date): Sending SIGTERM to process $PID"
        kill -TERM "$PID"
        
        # Wait for graceful shutdown
        local wait_time=0
        while kill -0 "$PID" 2>/dev/null && [ $wait_time -lt $SHUTDOWN_TIMEOUT ]; do
            echo "$(date): Waiting for graceful shutdown... ($wait_time/$SHUTDOWN_TIMEOUT seconds)"
            sleep 1
            wait_time=$((wait_time + 1))
        done
        
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "$(date): Graceful shutdown timed out, sending SIGKILL"
            kill -KILL "$PID"
        else
            echo "$(date): Process shutdown gracefully"
        fi
        
        wait "$PID"
    fi
    
    echo "$(date): Entrypoint shutdown complete"
    exit 0
}

# Health check function
health_check() {
    local url="http://localhost:8000/health"
    
    if command -v curl >/dev/null 2>&1; then
        curl -f -s "$url" >/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget -q --spider "$url"
    else
        # Fallback using python
        python3 -c "
import urllib.request
import sys
try:
    urllib.request.urlopen('$url', timeout=5)
    sys.exit(0)
except:
    sys.exit(1)
"
    fi
}

# Pre-flight checks
preflight_checks() {
    echo "$(date): Running pre-flight checks..."
    
    # Check required environment variables
    required_vars=("PYTHONUNBUFFERED" "PYTHONDONTWRITEBYTECODE")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "$(date): Warning: Required environment variable $var is not set"
        fi
    done
    
    # Check Python installation
    if ! command -v python3 >/dev/null 2>&1; then
        echo "$(date): Error: Python3 is not installed"
        exit 1
    fi
    
    # Check if we can import required modules
    python3 -c "
import sys
try:
    import uvicorn
    import testgen_copilot.quantum_api
    print('✅ Required modules available')
except ImportError as e:
    print(f'❌ Missing required module: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "$(date): Error: Required Python modules not available"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/temp
    
    # Set proper permissions
    chmod 750 /app/logs /app/data /app/temp
    
    echo "$(date): Pre-flight checks completed successfully"
}

# Main application startup
start_application() {
    echo "$(date): Starting Quantum Task Planner API..."
    
    # Set up signal handlers
    trap 'shutdown_handler' SIGTERM SIGINT SIGQUIT
    
    # Start the application in the background
    exec "$@" &
    PID=$!
    
    echo "$(date): Application started with PID $PID"
    
    # Wait for application to be ready
    local ready=false
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ] && [ "$ready" = false ]; do
        if health_check; then
            echo "$(date): Application is ready and healthy"
            ready=true
        else
            echo "$(date): Waiting for application to be ready... (attempt $((attempts + 1))/$max_attempts)"
            sleep 2
            attempts=$((attempts + 1))
        fi
    done
    
    if [ "$ready" = false ]; then
        echo "$(date): Error: Application failed to become ready within timeout"
        kill -TERM $PID 2>/dev/null || true
        exit 1
    fi
    
    # Monitor application health
    monitor_application
}

# Application monitoring loop
monitor_application() {
    echo "$(date): Starting application monitoring..."
    
    local failed_checks=0
    local max_failed_checks=3
    
    while kill -0 "$PID" 2>/dev/null; do
        if health_check; then
            failed_checks=0
        else
            failed_checks=$((failed_checks + 1))
            echo "$(date): Health check failed ($failed_checks/$max_failed_checks)"
            
            if [ $failed_checks -ge $max_failed_checks ]; then
                echo "$(date): Maximum health check failures reached, initiating shutdown"
                shutdown_handler
                break
            fi
        fi
        
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    # Wait for the process to finish
    wait $PID
    local exit_code=$?
    
    echo "$(date): Application process exited with code $exit_code"
    exit $exit_code
}

# Resource monitoring (optional background task)
resource_monitor() {
    while true; do
        # Memory usage check
        local mem_usage=$(ps -o pid,ppid,user,%mem,comm -p $PID 2>/dev/null | tail -1 | awk '{print $4}')
        if [ -n "$mem_usage" ] && (( $(echo "$mem_usage > 80" | bc -l) )); then
            echo "$(date): Warning: High memory usage detected: ${mem_usage}%"
        fi
        
        # CPU usage could be monitored here as well
        
        sleep 60
    done
}

# Main execution
main() {
    echo "$(date): Quantum Task Planner entrypoint started"
    echo "$(date): Arguments: $*"
    
    # Run pre-flight checks
    preflight_checks
    
    # Start resource monitoring in background (optional)
    # resource_monitor &
    
    # Start the main application
    start_application "$@"
}

# Execute main function with all arguments
main "$@"