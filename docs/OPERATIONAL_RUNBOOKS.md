# Operational Runbooks

This document contains operational procedures for managing TestGen Copilot in production environments.

## Table of Contents

- [System Health Monitoring](#system-health-monitoring)
- [Performance Monitoring](#performance-monitoring)
- [Security Incident Response](#security-incident-response)
- [Deployment Procedures](#deployment-procedures)
- [Backup and Recovery](#backup-and-recovery)
- [Scaling Operations](#scaling-operations)
- [Troubleshooting](#troubleshooting)

## System Health Monitoring

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/admin/status

# Metrics endpoint
curl http://localhost:8000/metrics
```

### Key Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| Response Time | > 2s | Investigate performance |
| Error Rate | > 5% | Check logs and dependencies |
| Memory Usage | > 80% | Scale up or optimize |
| CPU Usage | > 70% | Scale horizontally |
| Disk Space | > 85% | Clean logs or scale storage |

### Alerting Setup

Configure alerts in your monitoring system:

```yaml
# Prometheus alerting rules example
groups:
  - name: testgen-copilot
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High response time detected"
```

## Performance Monitoring

### Database Performance

```bash
# Check database connections
psql -h localhost -U testgen -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor slow queries
psql -h localhost -U testgen -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### Application Performance

```bash
# Check memory usage
ps aux | grep testgen

# Monitor file descriptors
lsof -p $(pgrep testgen)

# Check network connections
netstat -an | grep :8000
```

### Performance Tuning

1. **Database Optimization**:
   ```sql
   -- Update table statistics
   ANALYZE;
   
   -- Check index usage
   SELECT schemaname, tablename, attname, n_distinct, correlation 
   FROM pg_stats WHERE tablename = 'your_table';
   ```

2. **Application Tuning**:
   ```python
   # Adjust worker processes
   gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker
   
   # Configure connection pooling
   DATABASE_POOL_SIZE=10
   DATABASE_MAX_OVERFLOW=20
   ```

## Security Incident Response

### Immediate Response Checklist

1. **Assess the Situation**
   - [ ] Identify the type of security incident
   - [ ] Determine the scope and impact
   - [ ] Document initial findings

2. **Contain the Incident**
   - [ ] Isolate affected systems
   - [ ] Disable compromised accounts
   - [ ] Block malicious traffic

3. **Evidence Collection**
   - [ ] Preserve logs before rotation
   - [ ] Take system snapshots
   - [ ] Document all actions taken

### Security Commands

```bash
# Check for suspicious processes
ps aux | grep -E '(python|node|java)' | grep -v testgen

# Monitor network connections
netstat -an | grep ESTABLISHED

# Check authentication logs
tail -f /var/log/auth.log

# Review application logs for security events
grep -E '(WARN|ERROR|CRITICAL)' /var/log/testgen/security.log
```

### Incident Response Contacts

- **Security Team**: security@testgen.dev
- **On-call Engineer**: +1-XXX-XXX-XXXX
- **Legal/Compliance**: legal@testgen.dev

## Deployment Procedures

### Blue-Green Deployment

```bash
# 1. Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# 2. Health check green environment
curl http://green.testgen.internal/health

# 3. Run smoke tests
pytest tests/smoke/ --target=green

# 4. Switch traffic to green
# Update load balancer configuration

# 5. Monitor for 15 minutes
# Check metrics and logs

# 6. If successful, shutdown blue
docker-compose -f docker-compose.blue.yml down
```

### Rollback Procedure

```bash
# 1. Switch traffic back to blue
# Update load balancer configuration

# 2. Verify blue environment health
curl http://blue.testgen.internal/health

# 3. Shutdown green environment
docker-compose -f docker-compose.green.yml down

# 4. Investigate and fix issues
```

### Database Migration

```bash
# 1. Backup database
pg_dump -h localhost -U testgen testgen_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Run migrations
alembic upgrade head

# 3. Verify migration
alembic current

# 4. Test application connectivity
python -c "from testgen_copilot.database import test_connection; test_connection()"
```

## Backup and Recovery

### Database Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups/testgen
mkdir -p $BACKUP_DIR

# Create backup
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > $BACKUP_DIR/testgen_$DATE.sql.gz

# Verify backup
gunzip -t $BACKUP_DIR/testgen_$DATE.sql.gz

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "testgen_*.sql.gz" -mtime +30 -delete
```

### Application Data Backup

```bash
# Backup configuration and user data
tar -czf /backups/testgen_config_$DATE.tar.gz \
    /etc/testgen/ \
    /var/lib/testgen/user_data/ \
    /var/lib/testgen/templates/
```

### Recovery Procedures

```bash
# Database recovery
gunzip -c /backups/testgen_20240129_120000.sql.gz | psql -h localhost -U testgen testgen_db

# Configuration recovery
tar -xzf /backups/testgen_config_20240129_120000.tar.gz -C /

# Restart services
systemctl restart testgen-copilot
```

## Scaling Operations

### Horizontal Scaling

```bash
# Scale up application instances
docker-compose up --scale app=3

# Or with Kubernetes
kubectl scale deployment testgen-copilot --replicas=3
```

### Vertical Scaling

```yaml
# Update resource limits
resources:
  limits:
    memory: "2Gi"
    cpu: "1000m"
  requests:
    memory: "1Gi"
    cpu: "500m"
```

### Database Scaling

```bash
# Read replica setup
psql -h primary -U testgen -c "SELECT pg_start_backup('replica_backup');"
rsync -av postgres@primary:/var/lib/postgresql/data/ /var/lib/postgresql/data/
psql -h primary -U testgen -c "SELECT pg_stop_backup();"
```

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Identify memory-hungry processes
top -o %MEM

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python -m testgen_copilot.cli

# Restart application
systemctl restart testgen-copilot
```

#### Database Connection Issues

```bash
# Check database connectivity
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# Check connection pool
psql -h $DB_HOST -U $DB_USER -c "SELECT count(*) FROM pg_stat_activity WHERE datname='$DB_NAME';"

# Reset connections
sudo service postgresql restart
```

#### Performance Degradation

```bash
# Check system resources
top
iostat 1 5
sar -u 1 5

# Application profiling
python -m cProfile -o profile.stats -m testgen_copilot.cli

# Database performance
psql -h $DB_HOST -U $DB_USER -c "SELECT query, total_time, calls, mean_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
```

### Log Analysis

```bash
# Search for errors in the last hour
find /var/log/testgen/ -name "*.log" -mmin -60 -exec grep -l "ERROR\|CRITICAL" {} \;

# Analyze error patterns
grep "ERROR" /var/log/testgen/app.log | awk '{print $4}' | sort | uniq -c | sort -nr

# Monitor logs in real-time
tail -f /var/log/testgen/app.log | grep -E "(ERROR|WARN|CRITICAL)"
```

### Emergency Procedures

#### Service Outage

1. **Immediate Response**
   ```bash
   # Check service status
   systemctl status testgen-copilot
   
   # Check logs for errors
   journalctl -u testgen-copilot -f
   
   # Restart service
   systemctl restart testgen-copilot
   ```

2. **If restart fails**
   ```bash
   # Start in debug mode
   systemctl stop testgen-copilot
   sudo -u testgen python -m testgen_copilot.cli --debug
   ```

#### Data Corruption

1. **Stop all services**
2. **Restore from latest backup**
3. **Verify data integrity**
4. **Restart services**
5. **Monitor for issues**

---

## Emergency Contacts

- **Primary On-call**: +1-XXX-XXX-XXXX
- **Secondary On-call**: +1-XXX-XXX-XXXX
- **DevOps Team**: devops@testgen.dev
- **Security Team**: security@testgen.dev

For critical issues, contact multiple channels simultaneously.