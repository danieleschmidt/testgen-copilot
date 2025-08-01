# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in TestGen-Copilot.

## Incident Severity Levels

### SEV-1: Critical (Service Down)
- **Definition**: Complete service outage affecting all users
- **Response Time**: Immediate (< 5 minutes)
- **Escalation**: Page on-call engineer immediately

### SEV-2: High (Significant Impact)
- **Definition**: Major functionality impaired, affecting >50% of users
- **Response Time**: 15 minutes
- **Escalation**: Notify on-call engineer via Slack

### SEV-3: Medium (Minor Impact)
- **Definition**: Limited functionality affected, <50% of users
- **Response Time**: 1 hour
- **Escalation**: Create JIRA ticket, notify team

### SEV-4: Low (Minimal Impact)
- **Definition**: Minor issues, no user impact
- **Response Time**: Next business day
- **Escalation**: Log issue for planning

## Incident Response Process

### 1. Detection and Alert
```bash
# Alert channels:
# - PagerDuty for SEV-1/SEV-2
# - Slack #alerts for all severities
# - Email notifications for SEV-3/SEV-4

# Check alert source
kubectl get pods -n testgen
docker ps | grep testgen
curl -f http://localhost:8000/health
```

### 2. Initial Assessment
```bash
# Quick health check
make health-check

# Check service status
systemctl status testgen-copilot
docker logs testgen-copilot --tail 50

# Review metrics dashboard
# Navigate to: http://grafana:3000/d/testgen-overview
```

### 3. Immediate Response Actions

#### Service Down (SEV-1)
```bash
# 1. Check if containers are running
docker ps | grep testgen

# 2. Restart service if needed
docker-compose restart testgen

# 3. Check logs for errors
docker logs testgen-copilot --since 5m

# 4. Verify health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready

# 5. Check dependencies
curl http://localhost:6379/  # Redis
ping prometheus  # Prometheus
```

#### High Error Rate (SEV-2)
```bash
# 1. Identify error patterns
tail -100 /var/log/testgen/app.log | grep ERROR

# 2. Check LLM API status
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# 3. Review recent deployments
git log --oneline -10
docker images | grep testgen

# 4. Check resource usage
docker stats testgen-copilot
free -h
df -h
```

#### Performance Issues (SEV-2/SEV-3)
```bash
# 1. Check response times
curl -w "%{time_total}\n" -s http://localhost:8000/health

# 2. Monitor active requests
curl http://localhost:8000/metrics | grep active_requests

# 3. Check system resources
top -p $(pgrep -f testgen)
iotop -a

# 4. Review cache performance
redis-cli info stats
```

### 4. Communication

#### Incident Declaration
```markdown
# Slack template for #incidents channel
🚨 **INCIDENT DECLARED** - SEV-{LEVEL}

**Service**: TestGen-Copilot
**Impact**: {Brief description of user impact}
**Started**: {Timestamp}
**Incident Commander**: {Your name}

**Current Status**: Investigating
**Next Update**: {Timestamp + 30 minutes}

#incident #{incident-id}
```

#### Status Updates
```markdown
# Update template (every 30 minutes for SEV-1/SEV-2)
📊 **INCIDENT UPDATE** - SEV-{LEVEL} - #{incident-id}

**Progress**: {What has been done}
**Current Focus**: {What you're working on now}
**ETA**: {Expected resolution time or next milestone}
**Next Update**: {Timestamp + 30 minutes}
```

#### Resolution Notice
```markdown
✅ **INCIDENT RESOLVED** - SEV-{LEVEL} - #{incident-id}

**Root Cause**: {Brief explanation}
**Resolution**: {What fixed the issue}
**Duration**: {Total incident time}
**Impact**: {Number of users/requests affected}

**Follow-up**: {Post-incident review scheduled for X}
```

## Common Incident Scenarios

### Scenario 1: Service Won't Start

**Symptoms**: Health checks failing, containers exiting

**Investigation Steps**:
```bash
# 1. Check container status
docker ps -a | grep testgen

# 2. Review startup logs
docker logs testgen-copilot

# 3. Check configuration
docker run --rm testgen-copilot:latest env | grep TESTGEN

# 4. Verify dependencies
docker-compose ps
```

**Common Causes**:
- Missing environment variables
- Database connectivity issues
- Port conflicts
- Configuration errors

**Resolution Steps**:
```bash
# Fix missing environment variables
cp .env.example .env
vim .env  # Add required values

# Fix port conflicts
docker-compose down
docker-compose up -d

# Reset configuration
docker-compose down -v
docker-compose up -d
```

### Scenario 2: High Memory Usage

**Symptoms**: Memory alerts, slow response times, OOM kills

**Investigation Steps**:
```bash
# 1. Check current memory usage
docker stats testgen-copilot

# 2. Review memory trends
# Check Grafana memory dashboard

# 3. Look for memory leaks
docker exec testgen-copilot python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"

# 4. Check for large objects in cache
redis-cli info memory
```

**Resolution Steps**:
```bash
# 1. Restart service (immediate relief)
docker-compose restart testgen

# 2. Clear cache if needed
redis-cli FLUSHALL

# 3. Scale resources if needed
docker-compose up -d --scale testgen=2

# 4. Investigate root cause
# Look for memory-intensive operations in logs
```

### Scenario 3: LLM API Issues

**Symptoms**: Generation failures, timeout errors, quota exceeded

**Investigation Steps**:
```bash
# 1. Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# 2. Check usage limits
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/usage

# 3. Review error patterns
grep "llm_error" /var/log/testgen/*.log | tail -20

# 4. Check rate limiting
curl http://localhost:8000/metrics | grep llm_requests_total
```

**Resolution Steps**:
```bash
# 1. Switch to backup provider
export PRIMARY_LLM_PROVIDER=anthropic

# 2. Implement rate limiting
export LLM_RATE_LIMIT=10  # requests per minute

# 3. Enable local LLM fallback
export LOCAL_LLM_ENABLED=true
docker-compose up -d ollama

# 4. Notify users of degraded service
# Post in #announcements channel
```

### Scenario 4: Database Connection Issues

**Symptoms**: 500 errors, database connection failures

**Investigation Steps**:
```bash
# 1. Check database status
docker-compose ps postgres

# 2. Test connectivity
docker exec testgen-copilot pg_isready -h postgres -p 5432

# 3. Check connection pool
docker logs testgen-copilot | grep "database"

# 4. Review database metrics
# Check Grafana database dashboard
```

**Resolution Steps**:
```bash
# 1. Restart database
docker-compose restart postgres

# 2. Check database logs
docker logs testgen-postgres

# 3. Reset connections
docker-compose restart testgen

# 4. Verify data integrity
docker exec testgen-postgres psql -U testgen -c "\dt"
```

## Post-Incident Procedures

### 1. Immediate Post-Resolution (< 1 hour)
- [ ] Verify service is fully operational
- [ ] Update monitoring dashboards
- [ ] Notify stakeholders of resolution
- [ ] Document timeline and actions taken

### 2. Post-Incident Review (Within 24 hours)
```markdown
# Template: docs/post-incident-reports/YYYY-MM-DD-{incident-id}.md

# Post-Incident Review: {Incident Title}

## Summary
- **Date**: {Date}
- **Duration**: {Start time} - {End time} ({Total duration})
- **Severity**: SEV-{Level}
- **Incident Commander**: {Name}

## Impact
- **Users Affected**: {Number}
- **Requests Failed**: {Number}
- **Revenue Impact**: {Amount if applicable}

## Timeline
| Time | Event | Action Taken |
|------|-------|--------------|
| 10:00 | Alert fired | Investigation started |
| 10:05 | Root cause identified | Applied fix |
| 10:15 | Service restored | Monitoring |

## Root Cause
{Detailed explanation of what went wrong}

## Resolution
{What was done to fix the issue}

## Lessons Learned
### What Went Well
- Quick detection through monitoring
- Effective communication during incident

### What Could Be Improved
- Faster root cause identification
- Better runbook documentation

## Action Items
- [ ] {Action 1} - Assigned to {Person} - Due {Date}
- [ ] {Action 2} - Assigned to {Person} - Due {Date}

## Follow-up
- Monitoring improvements: {Details}
- Process improvements: {Details}
- Technical debt: {Details}
```

### 3. Long-term Improvements
- [ ] Update runbooks with new learnings
- [ ] Implement preventive measures
- [ ] Improve monitoring and alerting
- [ ] Conduct chaos engineering exercises
- [ ] Update incident response procedures

## Emergency Contacts

### On-Call Rotation
- **Primary**: {Name} - {Phone} - {Slack: @username}
- **Secondary**: {Name} - {Phone} - {Slack: @username}
- **Manager**: {Name} - {Phone} - {Slack: @username}

### Escalation Path
1. **L1**: On-call Engineer (respond within 5 minutes)
2. **L2**: Lead Engineer (escalate after 30 minutes)
3. **L3**: Engineering Manager (escalate after 1 hour)
4. **L4**: Director/VP Engineering (for major incidents)

### External Contacts
- **LLM Provider Support**: {Contact information}
- **Cloud Provider Support**: {Contact information}
- **Monitoring Vendor**: {Contact information}

## Tools and Resources

### Monitoring Dashboards
- **Service Overview**: http://grafana:3000/d/testgen-overview
- **Infrastructure**: http://grafana:3000/d/testgen-infra
- **Security**: http://grafana:3000/d/testgen-security

### Log Analysis
- **Application Logs**: `/var/log/testgen/app.log`
- **Error Logs**: `/var/log/testgen/error.log`
- **Access Logs**: `/var/log/nginx/access.log`

### Useful Commands
```bash
# Quick health check
make health-check

# Service restart
docker-compose restart testgen

# View recent logs
docker logs testgen-copilot --tail 100 --follow

# Check metrics
curl http://localhost:8000/metrics

# Database status
docker exec testgen-postgres pg_isready

# Redis status
docker exec testgen-redis redis-cli ping
```

---

**Remember**: Stay calm, communicate clearly, and follow the runbook. When in doubt, escalate to the next level.