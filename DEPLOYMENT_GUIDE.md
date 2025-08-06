# Quantum-Inspired Task Planner - Deployment Guide

## Quick Start

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 10GB+ free disk space
- Python 3.11+ (for development)

### One-Command Deployment
```bash
./scripts/deploy-quantum.sh
```

## Production Deployment

### 1. System Requirements

#### Minimum Requirements
- **CPU**: 4 cores (2.0 GHz+)
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 100 Mbps
- **OS**: Ubuntu 20.04+, CentOS 8+, or RHEL 8+

#### Recommended for Production
- **CPU**: 8+ cores (3.0 GHz+)
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD with backup
- **Network**: 1 Gbps
- **Load Balancer**: Nginx/HAProxy
- **Monitoring**: Prometheus + Grafana

### 2. Pre-Deployment Setup

#### Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

#### System Optimization
```bash
# Increase file limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
echo "net.core.somaxconn=65535" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 3. Configuration

#### Environment Variables
Create `.env` file:
```bash
# Database Configuration
QUANTUM_DATABASE_URL=postgresql://quantum:quantum_secure_2024@postgres:5432/quantum_planner
POSTGRES_DB=quantum_planner
POSTGRES_USER=quantum
POSTGRES_PASSWORD=quantum_secure_2024

# Redis Configuration  
QUANTUM_REDIS_URL=redis://redis:6379/0

# Application Configuration
QUANTUM_ENV=production
QUANTUM_LOG_LEVEL=INFO
QUANTUM_WORKERS=4
QUANTUM_MAX_ITERATIONS=2000
QUANTUM_COHERENCE_TIME=60
QUANTUM_ENABLE_MONITORING=true

# Security Configuration
QUANTUM_SECRET_KEY=your-secret-key-here
QUANTUM_JWT_SECRET=your-jwt-secret-here
QUANTUM_ENCRYPTION_KEY=your-encryption-key-here

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=quantum_admin_2024
PROMETHEUS_RETENTION=30d

# SSL Configuration (optional)
SSL_CERT_PATH=/etc/ssl/certs/quantum-planner.crt
SSL_KEY_PATH=/etc/ssl/private/quantum-planner.key
```

#### Production Docker Compose
The production configuration in `docker-compose.quantum-production.yml` includes:

- **Multi-replica API services** (3 instances)
- **Background workers** (2 instances)
- **Database with persistence** (PostgreSQL)
- **Cache layer** (Redis)
- **Load balancer** (Nginx)
- **Monitoring stack** (Prometheus, Grafana, AlertManager)
- **Health checks and restart policies**
- **Resource limits and reservations**

### 4. SSL/TLS Setup

#### Generate Self-Signed Certificate (Development)
```bash
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/quantum-planner.key \
  -out ssl/quantum-planner.crt \
  -subj "/C=US/ST=State/L=City/O=Quantum/CN=quantum-planner.local"
```

#### Use Let's Encrypt (Production)
```bash
# Install certbot
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/quantum-planner.crt
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/quantum-planner.key
```

### 5. Deployment Process

#### Automated Deployment
```bash
# Full deployment with all checks
./scripts/deploy-quantum.sh deploy

# Build images only
./scripts/deploy-quantum.sh build

# Health check only
./scripts/deploy-quantum.sh health-check

# Security scan
./scripts/deploy-quantum.sh security-scan
```

#### Manual Deployment Steps
```bash
# 1. Clone repository
git clone https://github.com/your-org/quantum-task-planner.git
cd quantum-task-planner

# 2. Build images
docker build -f Dockerfile.quantum -t quantum-task-planner:latest .

# 3. Run security scan
python3 scripts/quantum_security_scan.py

# 4. Start services
docker-compose -f docker-compose.quantum-production.yml up -d

# 5. Verify deployment
curl -f http://localhost:8000/health
```

### 6. Service Verification

#### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready -U quantum

# Redis health  
docker-compose exec redis redis-cli ping

# All services status
docker-compose ps
```

#### Performance Validation
```bash
# Load test (requires curl and apache bench)
ab -n 1000 -c 10 http://localhost:8000/health

# Quantum metrics
curl http://localhost:8000/quantum-metrics

# System metrics
curl http://localhost:9090/metrics
```

### 7. Monitoring Setup

#### Access Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/quantum_admin_2024)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

#### Import Grafana Dashboards
1. Login to Grafana
2. Go to **+** → **Import**
3. Upload `monitoring/grafana/dashboards/quantum-overview.json`
4. Configure data source: Prometheus (http://prometheus:9090)

#### Configure Alerts
```bash
# Test alert rules
curl -X POST http://localhost:9090/-/reload

# Check alert status
curl http://localhost:9090/api/v1/rules

# Test AlertManager
curl -XPOST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels":{"alertname":"TestAlert","severity":"warning"}}]'
```

### 8. Backup Strategy

#### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U quantum quantum_planner > backup_$(date +%Y%m%d).sql

# Automated backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
docker-compose exec -T postgres pg_dump -U quantum quantum_planner > $BACKUP_DIR/db_$DATE.sql
find $BACKUP_DIR -name "db_*.sql" -mtime +7 -delete
EOF

chmod +x backup.sh

# Schedule with cron
echo "0 2 * * * /path/to/backup.sh" | crontab -
```

#### Application Data Backup
```bash
# Backup volumes
docker run --rm -v quantum_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup_$(date +%Y%m%d).tar.gz -C /data .
docker run --rm -v quantum_redis_data:/data -v $(pwd):/backup ubuntu tar czf /backup/redis_backup_$(date +%Y%m%d).tar.gz -C /data .
```

### 9. Scaling

#### Horizontal Scaling
```bash
# Scale API instances
docker-compose -f docker-compose.quantum-production.yml up -d --scale quantum-api=5

# Scale workers
docker-compose -f docker-compose.quantum-production.yml up -d --scale quantum-worker=4

# Check scaling status
docker-compose ps
```

#### Vertical Scaling
Edit `docker-compose.quantum-production.yml`:
```yaml
quantum-api:
  deploy:
    resources:
      limits:
        cpus: '4'      # Increased from 2
        memory: 8G     # Increased from 4G
      reservations:
        cpus: '1'
        memory: 1G
```

### 10. Security Hardening

#### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable

# Block direct access to internal ports
sudo ufw deny 5432      # PostgreSQL
sudo ufw deny 6379      # Redis
sudo ufw deny 9090      # Prometheus
```

#### Regular Security Updates
```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Docker security scan
docker scan quantum-task-planner:latest

# Application security scan  
python3 scripts/quantum_security_scan.py

# Check for vulnerabilities
docker-compose exec quantum-api python -m safety check
```

### 11. Maintenance

#### Log Management
```bash
# View logs
docker-compose logs -f quantum-api
docker-compose logs --tail=100 quantum-worker

# Log rotation
docker-compose exec quantum-api logrotate /etc/logrotate.d/quantum

# Clean old logs
docker system prune -a --filter "until=24h"
```

#### Performance Monitoring
```bash
# Resource usage
docker stats

# Quantum performance benchmark
docker-compose exec quantum-api python -c "
from src.testgen_copilot.quantum_performance import benchmark_quantum_performance
import asyncio
asyncio.run(benchmark_quantum_performance())
"
```

### 12. Troubleshooting

#### Common Issues

**Service Won't Start**
```bash
# Check logs
docker-compose logs quantum-api

# Check resource usage
docker system df
free -h

# Restart specific service
docker-compose restart quantum-api
```

**Database Connection Issues**
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection
docker-compose exec postgres psql -U quantum -d quantum_planner -c "SELECT version();"

# Reset database
docker-compose down
docker volume rm quantum_postgres_data
docker-compose up -d postgres
```

**Performance Issues**
```bash
# Check system resources
htop
iotop

# Quantum metrics
curl http://localhost:8000/quantum-metrics | jq

# Benchmark performance
./scripts/deploy-quantum.sh performance-test
```

### 13. Rollback Procedure

#### Quick Rollback
```bash
# Using deployment script
./scripts/deploy-quantum.sh rollback

# Manual rollback
latest_backup=$(ls -1t backups/ | head -n1)
docker-compose down
cp backups/$latest_backup/docker-compose.quantum-production.yml ./
docker-compose up -d
```

### 14. Production Checklist

- [ ] System requirements met
- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Firewall configured
- [ ] Monitoring dashboards working
- [ ] Backup strategy implemented
- [ ] Log rotation configured
- [ ] Health checks passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Team trained on operations

## Support

For deployment issues:
1. Check logs: `docker-compose logs`
2. Run health checks: `./scripts/deploy-quantum.sh health-check`
3. Review security scan: `python3 scripts/quantum_security_scan.py`
4. Check system resources: `docker system df && free -h`

For production support, contact the quantum team at quantum-support@yourcompany.com