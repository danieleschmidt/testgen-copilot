# Runbook: Service Down

## Overview

This runbook provides step-by-step instructions for diagnosing and resolving TestGen Copilot Assistant service outages.

## Alert Information

- **Alert Name**: TestGenServiceDown
- **Severity**: Critical
- **Threshold**: Service unreachable for > 1 minute
- **Impact**: Users cannot generate tests or run security scans

## Initial Response (0-5 minutes)

### 1. Acknowledge Alert
- [ ] Acknowledge the alert in monitoring system
- [ ] Check if this is a planned maintenance
- [ ] Notify team in incident channel

### 2. Quick Assessment
```bash
# Check service status
curl -f http://testgen-service:8000/health

# Check container status
docker ps | grep testgen

# Check recent logs
docker logs testgen-copilot --tail 50
```

### 3. Immediate Actions
If service is completely down:
- [ ] Restart service container
- [ ] Check resource availability
- [ ] Verify network connectivity

## Detailed Diagnosis (5-15 minutes)

### System Resources

```bash
# Check CPU usage
top -n 1 | head -20

# Check memory usage
free -h

# Check disk space
df -h

# Check load average
uptime
```

### Application Logs

```bash
# Check application logs for errors
docker logs testgen-copilot --since=30m | grep -i error

# Check startup logs
docker logs testgen-copilot --since=30m | head -50

# Check for specific error patterns
docker logs testgen-copilot --since=30m | grep -E "(failed|exception|timeout|connection)"
```

### Network Connectivity

```bash
# Test internal network
ping redis
ping postgres

# Test external dependencies
curl -I https://api.openai.com
curl -I https://api.anthropic.com

# Check port availability
netstat -tlnp | grep :8000
```

### Database Connectivity

```bash
# Test Redis connection
redis-cli -h redis ping

# Test PostgreSQL connection (if used)
pg_isready -h postgres -p 5432 -U testgen
```

## Common Issues and Solutions

### Issue 1: Out of Memory

**Symptoms**:
- High memory usage (>90%)
- OOMKilled in container logs
- Slow response times

**Solution**:
```bash
# Check memory usage
docker stats testgen-copilot

# Restart with more memory
docker-compose down
docker-compose up -d --scale testgen=1 --memory=2g

# Monitor memory usage
watch -n 5 'docker stats testgen-copilot --no-stream'
```

### Issue 2: Database Connection Lost

**Symptoms**:
- "Connection refused" errors
- Database timeout messages
- Unable to save/load data

**Solution**:
```bash
# Check database container
docker ps | grep postgres
docker logs postgres-container --tail 20

# Restart database if needed
docker-compose restart postgres

# Verify connection
docker exec testgen-copilot python -c "
import psycopg2
conn = psycopg2.connect('postgresql://user:pass@postgres:5432/db')
print('Connection successful')
"
```

### Issue 3: API Rate Limits

**Symptoms**:
- "Rate limit exceeded" errors
- 429 HTTP status codes
- Slow test generation

**Solution**:
```bash
# Check API usage metrics
curl http://testgen-service:8000/metrics | grep api_requests

# Check API key configuration
docker exec testgen-copilot env | grep API_KEY

# Implement backoff strategy
# (Restart with updated configuration)
```

### Issue 4: Disk Space Full

**Symptoms**:
- "No space left on device" errors
- Unable to write logs or cache

**Solution**:
```bash
# Check disk usage
df -h

# Clean up Docker images
docker system prune -f

# Clean up logs
docker logs testgen-copilot --since=0 > /dev/null

# Clean up application cache
docker exec testgen-copilot rm -rf /tmp/testgen-cache/*
```

## Resolution Steps

### Service Restart

```bash
# Graceful restart
docker-compose restart testgen

# Force restart if needed
docker-compose down
docker-compose up -d

# Verify service is healthy
curl http://testgen-service:8000/health
```

### Configuration Update

```bash
# Update environment variables
vim .env

# Apply configuration changes
docker-compose down
docker-compose up -d

# Verify new configuration
docker exec testgen-copilot env | grep TESTGEN_
```

### Emergency Rollback

```bash
# Rollback to previous version
docker pull testgen-copilot:previous-tag
docker-compose down
docker-compose up -d

# Verify rollback successful
curl http://testgen-service:8000/version
```

## Monitoring and Verification

### Health Checks

```bash
# Basic health check
curl -f http://testgen-service:8000/health

# Detailed health check
curl http://testgen-service:8000/health | jq .

# Test key functionality
curl -X POST http://testgen-service:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"code": "def add(a, b): return a + b"}'
```

### Performance Verification

```bash
# Check response times
time curl http://testgen-service:8000/health

# Monitor metrics
curl http://testgen-service:8000/metrics

# Load test (if needed)
ab -n 100 -c 10 http://testgen-service:8000/health
```

## Escalation

### When to Escalate

Escalate if:
- [ ] Service cannot be restored within 30 minutes
- [ ] Data corruption is suspected
- [ ] Security incident is suspected
- [ ] Multiple services are affected

### Escalation Contacts

1. **Primary**: On-call engineer (PagerDuty)
2. **Secondary**: Team lead
3. **Management**: Engineering manager (for major incidents)

### Escalation Actions

- [ ] Page on-call engineer
- [ ] Create incident ticket
- [ ] Start incident bridge
- [ ] Notify stakeholders

## Post-Incident Actions

### Immediate (within 1 hour)

- [ ] Document timeline of events
- [ ] Collect relevant logs and metrics
- [ ] Verify service is fully operational
- [ ] Update monitoring if needed

### Follow-up (within 24 hours)

- [ ] Conduct post-incident review
- [ ] Create action items for improvements
- [ ] Update runbooks based on learnings
- [ ] Communicate resolution to stakeholders

### Long-term (within 1 week)

- [ ] Implement preventive measures
- [ ] Improve monitoring and alerting
- [ ] Update documentation
- [ ] Training for team members

## Prevention

### Monitoring Improvements

- Add synthetic transaction monitoring
- Implement predictive alerting
- Monitor business metrics
- Set up capacity planning alerts

### Infrastructure Improvements

- Implement auto-scaling
- Add redundancy for critical components
- Improve deployment automation
- Regular disaster recovery testing

### Process Improvements

- Regular health checks
- Automated failover procedures
- Improved incident response training
- Better communication processes

## Related Documentation

- [Health Check Guide](../guides/health-checks.md)
- [Monitoring Setup](../guides/monitoring.md)
- [Incident Response Plan](incident-response.md)
- [Service Architecture](../ARCHITECTURE.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2024-01-01 | DevOps Team | Initial version |

---

**Remember**: Stay calm, follow the steps systematically, and don't hesitate to escalate if needed.