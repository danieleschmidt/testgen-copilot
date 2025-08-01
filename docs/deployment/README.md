# Deployment Guide

This document provides comprehensive deployment instructions for TestGen-Copilot across different environments.

## Quick Start

### Local Development
```bash
# Using Docker Compose (Recommended)
docker-compose up -d

# Or traditional setup
make setup-dev
make test
```

### Production Deployment
```bash
# Build production image
docker build --target runtime -t testgen-copilot:production .

# Run with minimal dependencies
docker run -d --name testgen-copilot testgen-copilot:production
```

## Deployment Environments

### Development Environment

The development environment includes all tools for coding, testing, and debugging.

```bash
# Start full development stack
docker-compose up -d

# Services included:
# - testgen: Main application with development tools
# - redis: Caching and session management
# - ollama: Local LLM service
# - prometheus: Metrics collection
# - grafana: Metrics visualization
# - postgres: Analytics database
```

#### Environment Variables (.env)
```bash
# LLM API Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Local LLM (Optional)
LOCAL_LLM_ENABLED=true

# Database
POSTGRES_DB=testgen
POSTGRES_USER=testgen
POSTGRES_PASSWORD=secure_password_here

# Monitoring
GRAFANA_PASSWORD=admin_password_here
REDIS_PASSWORD=redis_password_here

# Build metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=0.1.0
VCS_REF=$(git rev-parse HEAD)
```

### Production Environment

#### Container Deployment

**Single Container (Minimal)**
```bash
# Build production image
docker build --target runtime -t testgen-copilot:latest .

# Run with basic configuration
docker run -d \
  --name testgen-copilot \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e TESTGEN_ENV=production \
  -e TESTGEN_LOG_LEVEL=INFO \
  -v testgen-data:/home/testgen/.testgen \
  testgen-copilot:latest
```

**Production Stack with Services**
```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes Deployment

**Basic Deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: testgen-copilot
  namespace: testgen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: testgen-copilot
  template:
    metadata:
      labels:
        app: testgen-copilot
    spec:
      containers:
      - name: testgen-copilot
        image: testgen-copilot:latest
        ports:
        - containerPort: 8000
        env:
        - name: TESTGEN_ENV
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: testgen-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - testgen
            - --version
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - testgen
            - --version
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Service Configuration**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: testgen-copilot-service
  namespace: testgen
spec:
  selector:
    app: testgen-copilot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Cloud Deployments

**AWS ECS**
```json
{
  "family": "testgen-copilot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "testgen-copilot",
      "image": "your-registry/testgen-copilot:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "TESTGEN_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:testgen/openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/testgen-copilot",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Google Cloud Run**
```bash
# Deploy to Cloud Run
gcloud run deploy testgen-copilot \
  --image gcr.io/your-project/testgen-copilot:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TESTGEN_ENV=production \
  --set-secrets OPENAI_API_KEY=openai-key:latest \
  --memory 512Mi \
  --cpu 1 \
  --concurrency 10 \
  --max-instances 100
```

**Azure Container Instances**
```bash
# Deploy to ACI
az container create \
  --resource-group testgen-rg \
  --name testgen-copilot \
  --image your-registry.azurecr.io/testgen-copilot:latest \
  --cpu 1 \
  --memory 1 \
  --ports 8000 \
  --environment-variables TESTGEN_ENV=production \
  --secure-environment-variables OPENAI_API_KEY=$OPENAI_API_KEY \
  --restart-policy Always
```

### CI/CD Deployment

#### GitHub Actions
```yaml
name: Deploy to Production
on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build and push Docker image
      run: |
        docker build -t testgen-copilot:${{ github.ref_name }} .
        docker tag testgen-copilot:${{ github.ref_name }} your-registry/testgen-copilot:${{ github.ref_name }}
        docker push your-registry/testgen-copilot:${{ github.ref_name }}
    
    - name: Deploy to production
      run: |
        # Your deployment script here
        kubectl set image deployment/testgen-copilot testgen-copilot=your-registry/testgen-copilot:${{ github.ref_name }}
```

## Configuration Management

### Environment-Specific Configuration

**Development (.env.development)**
```bash
TESTGEN_ENV=development
TESTGEN_LOG_LEVEL=DEBUG
TESTGEN_DEBUG=true
LOCAL_LLM_ENABLED=true
CACHE_TTL=300
MAX_WORKERS=2
```

**Production (.env.production)**
```bash
TESTGEN_ENV=production
TESTGEN_LOG_LEVEL=INFO
TESTGEN_DEBUG=false
LOCAL_LLM_ENABLED=false
CACHE_TTL=3600
MAX_WORKERS=8
RATE_LIMIT=100
```

### Secrets Management

**Local Development**
- Use `.env` files (never commit to git)
- Use Docker secrets for docker-compose

**Production**
- Kubernetes: Use Secrets and ConfigMaps
- AWS: Use Systems Manager Parameter Store or Secrets Manager
- Azure: Use Key Vault
- GCP: Use Secret Manager

### Configuration Validation

The application validates configuration on startup:
```python
# Example configuration validation
from pydantic import BaseSettings

class Settings(BaseSettings):
    testgen_env: str = "development"
    openai_api_key: Optional[str] = None
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
```

## Monitoring and Observability

### Health Checks

The application provides health check endpoints:
```bash
# Check application health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready

# Check liveness
curl http://localhost:8000/alive
```

### Metrics Collection

Prometheus metrics are available at `/metrics`:
```bash
curl http://localhost:8000/metrics
```

### Logging

Structured logging is configured based on environment:
- Development: Console output with DEBUG level
- Production: JSON format with INFO level
- Log aggregation: Compatible with ELK, Fluentd, etc.

### Alerting

Key alerts to configure:
- Application down/unhealthy
- High error rate (>5%)
- High memory usage (>80%)
- API rate limit exceeded
- LLM API errors

## Security Considerations

### Container Security
- Non-root user (testgen:testgen)
- Read-only root filesystem where possible
- Minimal attack surface (slim base image)
- Regular security scanning

### Network Security
- Use TLS/HTTPS in production
- Network policies in Kubernetes
- Firewall rules for cloud deployments
- VPC/subnet isolation

### Secrets Security
- Never include secrets in images
- Use platform-specific secret management
- Rotate secrets regularly
- Monitor for secret leaks

### API Security
- Rate limiting
- Input validation
- Authentication/authorization
- CORS configuration

## Backup and Recovery

### Data Backup
```bash
# Backup configuration and cache
docker run --rm -v testgen-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/testgen-backup-$(date +%Y%m%d).tar.gz /data
```

### Database Backup (if using)
```bash
# PostgreSQL backup
pg_dump -h localhost -U testgen -d testgen > backup.sql

# Restore
psql -h localhost -U testgen -d testgen < backup.sql
```

### Disaster Recovery
1. Regular automated backups
2. Test restore procedures
3. Document recovery time objectives (RTO)
4. Cross-region backup storage

## Performance Optimization

### Resource Allocation
- CPU: 1-2 cores for typical workloads
- Memory: 512MB-1GB base + cache requirements
- Storage: SSD preferred for cache performance

### Scaling Strategies
- Horizontal scaling: Multiple container instances
- Vertical scaling: Increase container resources
- Auto-scaling based on CPU/memory usage
- Queue-based processing for batch operations

### Caching Strategy
- Redis for session and result caching
- Local filesystem cache for models
- CDN for static assets (if applicable)

## Troubleshooting

### Common Issues

**Container Won't Start**
```bash
# Check logs
docker logs testgen-copilot

# Check configuration
docker exec -it testgen-copilot env

# Test configuration
docker run --rm testgen-copilot:latest testgen --version
```

**High Memory Usage**
```bash
# Check memory usage
docker stats testgen-copilot

# Analyze memory usage
docker exec -it testgen-copilot python -c "import psutil; print(psutil.virtual_memory())"
```

**API Errors**
```bash
# Check API connectivity
curl -v http://localhost:8000/api/health

# Check logs for errors
docker logs testgen-copilot | grep ERROR
```

### Debug Mode

Enable debug mode for troubleshooting:
```bash
docker run -e TESTGEN_DEBUG=true -e TESTGEN_LOG_LEVEL=DEBUG testgen-copilot:latest
```

## Maintenance

### Updates
1. Test in development environment
2. Run integration tests
3. Backup current version
4. Deploy with blue-green or rolling update
5. Monitor for issues
6. Rollback if necessary

### Monitoring
- Regular health checks
- Performance metrics review
- Log analysis
- Security scanning
- Dependency updates

---

For additional deployment questions, see the [CONTRIBUTING.md](../CONTRIBUTING.md) guide or contact the development team.