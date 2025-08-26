# 🚀 Production Deployment Guide - Terragon Autonomous SDLC

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+/RHEL 8+), macOS (10.15+), Windows (Server 2019+)
- **Python**: 3.8+ (3.11+ recommended for optimal performance)
- **Memory**: 4GB minimum, 8GB+ recommended for production
- **CPU**: 2 cores minimum, 4+ cores recommended
- **Disk Space**: 10GB minimum, 50GB+ recommended for logs and data

### Infrastructure Requirements
- **Container Runtime**: Docker 20.10+ or Podman 3.0+
- **Orchestration** (Optional): Kubernetes 1.21+ with Helm 3.0+
- **Database**: PostgreSQL 13+ (for enterprise features)
- **Redis**: 6.0+ (for caching and session management)
- **Load Balancer**: Nginx, HAProxy, or cloud load balancer

## 🐳 Docker Deployment (Recommended)

### Single Container Deployment

```bash
# Pull the latest image
docker pull terragonlabs/testgen-copilot:latest

# Run with basic configuration
docker run -d \
  --name testgen-copilot \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=postgresql://user:pass@localhost/testgen \
  -v $(pwd)/data:/app/data \
  terragonlabs/testgen-copilot:latest

# Check status
docker logs testgen-copilot
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  testgen-api:
    image: terragonlabs/testgen-copilot:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://testgen:${DB_PASSWORD}@postgres:5432/testgen
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - QUANTUM_ENABLED=true
      - MONITORING_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=testgen
      - POSTGRES_USER=testgen
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - testgen-api

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Deploy with Docker Compose

```bash
# Set environment variables
export DB_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(openssl rand -base64 32)
export GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Save environment variables
cat << EOF > .env
DB_PASSWORD=${DB_PASSWORD}
SECRET_KEY=${SECRET_KEY}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
EOF

# Deploy the stack
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
```

## ☸️ Kubernetes Deployment

### Namespace and Configuration

```yaml
# k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: testgen-copilot
  labels:
    app: testgen-copilot

---
apiVersion: v1
kind: Secret
metadata:
  name: testgen-secrets
  namespace: testgen-copilot
type: Opaque
data:
  db-password: <base64-encoded-password>
  secret-key: <base64-encoded-secret>
  
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: testgen-config
  namespace: testgen-copilot
data:
  ENVIRONMENT: "production"
  QUANTUM_ENABLED: "true"
  MONITORING_ENABLED: "true"
  DATABASE_URL: "postgresql://testgen:$(DB_PASSWORD)@postgres:5432/testgen"
  REDIS_URL: "redis://redis:6379/0"
```

### Application Deployment

```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: testgen-api
  namespace: testgen-copilot
  labels:
    app: testgen-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: testgen-api
  template:
    metadata:
      labels:
        app: testgen-api
    spec:
      containers:
      - name: testgen-api
        image: terragonlabs/testgen-copilot:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: testgen-secrets
              key: db-password
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: testgen-secrets
              key: secret-key
        envFrom:
        - configMapRef:
            name: testgen-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: testgen-data

---
apiVersion: v1
kind: Service
metadata:
  name: testgen-api-service
  namespace: testgen-copilot
spec:
  selector:
    app: testgen-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: testgen-data
  namespace: testgen-copilot
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### Ingress Configuration

```yaml
# k8s/ingress.yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: testgen-ingress
  namespace: testgen-copilot
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: testgen-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: testgen-api-service
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Create secrets
kubectl create secret generic testgen-secrets \
  --from-literal=db-password="$(openssl rand -base64 32)" \
  --from-literal=secret-key="$(openssl rand -base64 32)" \
  -n testgen-copilot

# Apply configurations
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/ingress.yml

# Verify deployment
kubectl get pods -n testgen-copilot
kubectl get services -n testgen-copilot
kubectl get ingress -n testgen-copilot
```

## 🔧 Configuration Management

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production                    # deployment environment
SECRET_KEY=your-secret-key               # JWT signing key
DATABASE_URL=postgresql://...            # database connection
REDIS_URL=redis://localhost:6379/0       # Redis connection

# Feature Flags
QUANTUM_ENABLED=true                     # enable quantum optimization
MONITORING_ENABLED=true                  # enable monitoring
AUTO_SCALING_ENABLED=true               # enable auto-scaling
COMPLIANCE_ENABLED=true                 # enable compliance engine

# Performance Tuning
MAX_WORKERS=4                           # worker process count
CACHE_SIZE=1000                         # cache size limit
BATCH_SIZE=100                          # processing batch size
TIMEOUT_SECONDS=300                     # operation timeout

# Security Configuration
CORS_ORIGINS=https://yourdomain.com     # CORS allowed origins
RATE_LIMIT_ENABLED=true                 # enable rate limiting
SESSION_TIMEOUT=3600                    # session timeout in seconds

# Compliance Configuration  
COMPLIANCE_PROFILE=global               # compliance profile
DATA_RESIDENCY_REGION=us-east-1        # data residency region
AUDIT_LOGGING=true                      # enable audit logging

# Monitoring Configuration
PROMETHEUS_ENABLED=true                 # enable Prometheus metrics
GRAFANA_ENABLED=true                    # enable Grafana dashboards
ALERT_WEBHOOK_URL=https://...           # alert webhook URL
```

### Configuration Files

#### application.yml
```yaml
# config/application.yml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20
  pool_pre_ping: true

cache:
  redis_url: ${REDIS_URL}
  default_ttl: 3600
  max_size: 1000

security:
  secret_key: ${SECRET_KEY}
  cors_origins: 
    - "https://yourdomain.com"
  rate_limiting:
    enabled: true
    requests_per_minute: 100

features:
  quantum_optimization: true
  auto_scaling: true
  compliance_engine: true
  monitoring: true

compliance:
  profile: "global"
  data_residency: "us-east-1"
  audit_logging: true
  encryption_at_rest: true

monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3000
  alerts:
    webhook_url: ${ALERT_WEBHOOK_URL}
```

## 🔐 Security Configuration

### SSL/TLS Setup

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -newkey rsa:4096 -keyout ssl/private.key \
  -out ssl/certificate.crt -days 365 -nodes \
  -subj "/C=US/ST=CA/L=SF/O=YourOrg/CN=yourdomain.com"

# Or use Let's Encrypt with Certbot
certbot certonly --standalone -d yourdomain.com
```

### Nginx SSL Configuration

```nginx
# nginx/nginx.conf
upstream testgen_backend {
    server testgen-api:8000;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://testgen_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://testgen_backend/health;
        access_log off;
    }
}
```

### Security Headers

```python
# Add to FastAPI application
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'testgen-api'
    static_configs:
      - targets: ['testgen-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: testgen_alerts
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: High error rate detected
        
    - alert: HighResponseTime
      expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: High response time detected
        
    - alert: DatabaseConnectionFailure
      expr: pg_up == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: Database connection failed
```

## 🚀 Deployment Process

### Pre-deployment Checklist

```bash
# 1. Environment validation
./scripts/validate-environment.sh

# 2. Configuration validation
./scripts/validate-config.sh

# 3. Security scan
./scripts/security-scan.sh

# 4. Performance baseline
./scripts/performance-baseline.sh

# 5. Database migration
./scripts/migrate-database.sh

# 6. Backup verification
./scripts/verify-backups.sh
```

### Blue-Green Deployment Script

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "Starting blue-green deployment..."
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"

# Deploy green environment
echo "Deploying green environment..."
docker-compose -f docker-compose.green.yml up -d

# Wait for health checks
echo "Waiting for health checks..."
for i in {1..30}; do
  if curl -f http://localhost:8001/health; then
    echo "Green environment is healthy"
    break
  fi
  sleep 10
done

# Run smoke tests
echo "Running smoke tests..."
./scripts/smoke-tests.sh http://localhost:8001

# Switch traffic
echo "Switching traffic to green environment..."
# Update load balancer configuration
./scripts/switch-traffic.sh green

# Monitor for 5 minutes
echo "Monitoring new deployment..."
sleep 300

# Verify deployment success
if ./scripts/verify-deployment.sh; then
  echo "Deployment successful - cleaning up blue environment"
  docker-compose -f docker-compose.blue.yml down
else
  echo "Deployment failed - rolling back"
  ./scripts/switch-traffic.sh blue
  docker-compose -f docker-compose.green.yml down
  exit 1
fi

echo "Blue-green deployment completed successfully"
```

### Health Checks

```bash
#!/bin/bash
# scripts/health-checks.sh

API_URL=${1:-http://localhost:8000}

echo "Running health checks against $API_URL"

# Basic health check
if ! curl -f "$API_URL/health"; then
  echo "❌ Health check failed"
  exit 1
fi

# Database connectivity
if ! curl -f "$API_URL/health/database"; then
  echo "❌ Database health check failed"
  exit 1
fi

# Cache connectivity
if ! curl -f "$API_URL/health/cache"; then
  echo "❌ Cache health check failed"
  exit 1
fi

# Performance check
response_time=$(curl -o /dev/null -s -w '%{time_total}' "$API_URL/api/v1/status")
if (( $(echo "$response_time > 2.0" | bc -l) )); then
  echo "❌ Response time too high: ${response_time}s"
  exit 1
fi

echo "✅ All health checks passed"
```

## 🔧 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats testgen-copilot

# Analyze memory profile
curl http://localhost:8000/debug/memory

# Adjust memory limits
docker update --memory=2g testgen-copilot
```

#### Database Connection Issues
```bash
# Check database connectivity
psql -h localhost -U testgen -d testgen -c "SELECT 1;"

# Check connection pool
curl http://localhost:8000/debug/db-pool

# Restart database
docker-compose restart postgres
```

#### Performance Issues
```bash
# Check performance metrics
curl http://localhost:8000/metrics

# Analyze slow queries
docker-compose exec postgres psql -U testgen -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC 
  LIMIT 10;"

# Check system resources
htop
iotop -o
```

### Log Analysis

```bash
# View application logs
docker-compose logs -f testgen-api

# Search for errors
docker-compose logs testgen-api | grep -i error

# Monitor performance logs
tail -f logs/performance.log | grep -v "GET /health"

# Analyze access patterns
cat logs/access.log | awk '{print $7}' | sort | uniq -c | sort -nr
```

## 📈 Scaling Guidelines

### Horizontal Scaling

```bash
# Scale API instances
docker-compose up --scale testgen-api=3

# Kubernetes scaling
kubectl scale deployment testgen-api --replicas=5 -n testgen-copilot

# Auto-scaling configuration
kubectl autoscale deployment testgen-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n testgen-copilot
```

### Vertical Scaling

```bash
# Increase resource limits
docker update --memory=4g --cpus=2 testgen-copilot

# Kubernetes resource adjustment
kubectl patch deployment testgen-api -n testgen-copilot -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "testgen-api",
          "resources": {
            "requests": {"memory": "1Gi", "cpu": "500m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

## 🔒 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup-database.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${BACKUP_DATE}.sql"

# Create backup
docker-compose exec postgres pg_dump -U testgen testgen > "backups/$BACKUP_FILE"

# Compress backup
gzip "backups/$BACKUP_FILE"

# Upload to cloud storage (optional)
# aws s3 cp "backups/${BACKUP_FILE}.gz" s3://your-backup-bucket/

echo "Backup completed: ${BACKUP_FILE}.gz"
```

### Restore Process

```bash
#!/bin/bash
# scripts/restore-database.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup-file>"
  exit 1
fi

# Stop application
docker-compose stop testgen-api

# Restore database
gunzip -c "$BACKUP_FILE" | docker-compose exec -T postgres psql -U testgen testgen

# Start application
docker-compose start testgen-api

echo "Restore completed from: $BACKUP_FILE"
```

## 📝 Maintenance

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash
# scripts/weekly-maintenance.sh

echo "Starting weekly maintenance..."

# Update system packages
apt update && apt upgrade -y

# Clean Docker
docker system prune -f
docker volume prune -f

# Rotate logs
logrotate /etc/logrotate.conf

# Database maintenance
docker-compose exec postgres psql -U testgen -c "VACUUM ANALYZE;"

# Check disk space
df -h

# Update certificates (if using Let's Encrypt)
certbot renew --quiet

echo "Weekly maintenance completed"
```

### Monitoring Checklist

```bash
# Daily checks
curl -f http://localhost:8000/health
docker-compose ps
df -h
free -h

# Weekly checks
docker-compose logs --since=1w | grep -i error
psql -U testgen -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Monthly checks
analyze security scan results
review performance metrics
update documentation
review backup retention policy
```

---

## 🎯 Production Readiness Verification

### Final Checklist

- [ ] All environment variables configured
- [ ] SSL/TLS certificates installed and valid
- [ ] Database connections tested
- [ ] Redis cache operational
- [ ] Load balancer configured
- [ ] Monitoring dashboards accessible
- [ ] Alerting rules configured
- [ ] Backup procedures tested
- [ ] Security scans passed
- [ ] Performance baselines established
- [ ] Documentation updated
- [ ] Team training completed

### Go-Live Steps

1. **Final smoke tests** in staging environment
2. **Backup current production** (if upgrading)
3. **Deploy using blue-green strategy**
4. **Verify all health checks pass**
5. **Monitor for 30 minutes minimum**
6. **Update DNS/load balancer** to route traffic
7. **Monitor metrics and logs**
8. **Notify stakeholders** of successful deployment

---

**🚀 Your Terragon Autonomous SDLC system is now ready for production deployment!**