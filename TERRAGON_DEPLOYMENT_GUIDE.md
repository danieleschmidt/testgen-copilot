# TestGen-Copilot TERRAGON SDLC Deployment Guide

## Quick Start - Global Production Deployment

This guide covers deployment of TestGen-Copilot with the new TERRAGON SDLC enhancements including robustness, scaling, and global-first features.

### Prerequisites
- Docker & Docker Compose
- Python 3.8+
- PostgreSQL 12+ (for production)
- Redis (for caching and sessions)

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository-url>
cd testgen-copilot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# Install additional dependencies for global features
pip install psutil redis asyncpg
```

### 2. Configuration
```bash
# Configure for your deployment
export TESTGEN_REGION="us-east-1"
export TESTGEN_LOCALE="en_US"
export TESTGEN_COMPLIANCE_FRAMEWORKS="GDPR,CCPA"
export TESTGEN_DATABASE_URL="postgresql://user:pass@localhost:5432/testgen"
export TESTGEN_REDIS_URL="redis://localhost:6379/0"
```

### 3. Initialize TERRAGON Features
```python
from testgen_copilot import (
    set_global_locale, get_compliance_engine, get_multi_region_manager,
    get_health_monitor, get_performance_optimizer, get_auto_scaler
)

# Set locale
set_global_locale("en_US")  # or your target locale

# Enable compliance
engine = get_compliance_engine()
engine.enable_compliance_framework("GDPR")  # as needed

# Start monitoring
monitor = get_health_monitor()
monitor.start_monitoring()

# Configure region
manager = get_multi_region_manager()
manager.start_monitoring()
```

## TERRAGON Enhancement Deployment

### Generation 2: Robustness Features

#### Health Monitoring
```python
from testgen_copilot.monitoring import get_health_monitor

# Start comprehensive monitoring
monitor = get_health_monitor()
monitor.start_monitoring()

# Configure alert thresholds
monitor.configure_alerts(
    cpu_threshold=80.0,
    memory_threshold=85.0,
    disk_threshold=90.0,
    alert_email="ops@example.com"
)

# Check system health
health = monitor.check_system_health()
print(f"System status: {health['status']}")
```

#### Fault Tolerance
```python
from testgen_copilot.resilience import (
    get_resilience_manager, circuit_breaker, retry
)

# Configure circuit breakers
@circuit_breaker(failure_threshold=5, timeout=60)
def external_api_call():
    # Your external API call here
    pass

# Configure retries
@retry(max_attempts=3, backoff_strategy="exponential")
def database_operation():
    # Your database operation here
    pass
```

#### Security Monitoring
```python
from testgen_copilot.security_monitoring import get_security_scanner

# Enable enhanced security scanning
scanner = get_security_scanner()

# Scan code for vulnerabilities
threats = scanner.scan_code("path/to/code.py")
for threat in threats:
    print(f"Security threat: {threat.description} (Level: {threat.level})")
```

### Generation 3: Performance & Scaling

#### Performance Optimization
```python
from testgen_copilot.performance_optimizer import get_performance_optimizer

# Configure performance optimization
optimizer = get_performance_optimizer()

# Use performance context for operations
with optimizer.optimize_operation("test_generation", cacheable=True):
    # Your test generation code here
    pass

# Get performance report
report = optimizer.get_comprehensive_report()
```

#### Auto-Scaling
```python
from testgen_copilot.auto_scaling import get_auto_scaler, ScalingPolicy

# Configure auto-scaling
scaler = get_auto_scaler(ScalingPolicy.BALANCED)

# Register workers
scaler.load_balancer.register_worker("worker1", capacity=100)
scaler.load_balancer.register_worker("worker2", capacity=150)

# Get scaling recommendations
metrics = ScalingMetrics(
    cpu_utilization=75.0,
    memory_utilization=60.0,
    response_time_ms=200.0
)
decision = scaler.should_scale(metrics)
```

### Generation 4: Global-First Features

#### Multi-Region Setup
```python
from testgen_copilot.multi_region import RegionConfig, Region, DataResidencyRequirement
from testgen_copilot.compliance import ComplianceFramework

# Configure EU region with GDPR compliance
eu_config = RegionConfig(
    region=Region.EU_WEST_1,
    name="Europe (Ireland)",
    country_code="IE",
    jurisdiction="European Union",
    data_residency=DataResidencyRequirement.GDPR_COMPLIANT,
    compliance_frameworks={ComplianceFramework.GDPR},
    endpoint_url="https://eu-west-1.testgen.example.com"
)

manager = get_multi_region_manager()
manager.configure_region(eu_config)
```

#### Compliance Configuration
```python
from testgen_copilot.compliance import ComplianceFramework, DataClassification

# Enable multiple compliance frameworks
engine = get_compliance_engine()
engine.enable_compliance_framework(ComplianceFramework.GDPR)
engine.enable_compliance_framework(ComplianceFramework.CCPA)
engine.enable_compliance_framework(ComplianceFramework.HIPAA)

# Set geographic jurisdiction
engine.set_geographic_jurisdiction({"US", "EU", "CA"})

# Log data processing for compliance
from testgen_copilot.compliance import log_data_processing, ProcessingPurpose

log_data_processing(
    DataClassification.PERSONAL,
    ProcessingPurpose.TESTING,
    data_subject_id="user123"
)
```

#### Internationalization
```python
from testgen_copilot.internationalization import SupportedLocales

# Configure multiple locales for global deployment
from testgen_copilot import t, set_global_locale

# Set German locale
set_global_locale(SupportedLocales.DE_DE)

# Use translated messages
message = t("cli.generating_tests", filename="beispiel.py")
error_msg = t("cli.error", error="Datei nicht gefunden")

# Format dates and currency for locale
from datetime import datetime
now = datetime.now()
localized_date = format_localized_datetime(now)
price = format_localized_currency(29.99)
```

## Docker Deployment

### Basic Docker Setup
```bash
# Build production image with TERRAGON enhancements
docker build -t testgen-copilot:terragon .

# Run with environment variables
docker run -d \
  -e TESTGEN_REGION=us-east-1 \
  -e TESTGEN_LOCALE=en_US \
  -e TESTGEN_COMPLIANCE_FRAMEWORKS=GDPR,CCPA \
  -p 8000:8000 \
  testgen-copilot:terragon
```

### Docker Compose Production
```yaml
version: '3.8'
services:
  testgen-copilot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TESTGEN_REGION=us-east-1
      - TESTGEN_LOCALE=en_US
      - TESTGEN_COMPLIANCE_FRAMEWORKS=GDPR,CCPA
      - TESTGEN_DATABASE_URL=postgresql://user:pass@postgres:5432/testgen
      - TESTGEN_REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: testgen
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Production Checklist

### Pre-Deployment
- [ ] **Generation 2 Features**
  - [ ] Health monitoring configured and tested
  - [ ] Circuit breakers configured for external services
  - [ ] Retry mechanisms tested with exponential backoff
  - [ ] Security monitoring enabled and alerts configured
  - [ ] Bulkhead isolation implemented for critical resources

- [ ] **Generation 3 Features**
  - [ ] Performance optimizer cache sizes configured
  - [ ] Auto-scaling policies defined and tested
  - [ ] Load balancer health checks configured
  - [ ] Worker registration and failover tested
  - [ ] Performance benchmarks validated

- [ ] **Generation 4 Features**
  - [ ] Target locales configured and translations verified
  - [ ] Compliance frameworks enabled for target jurisdictions
  - [ ] Data residency requirements configured per region
  - [ ] Multi-region failover tested
  - [ ] GDPR/CCPA data handling verified

### Security & Compliance
- [ ] HTTPS/TLS encryption enabled
- [ ] Security scanning pipeline active
- [ ] Compliance audit logging enabled
- [ ] Data retention policies configured
- [ ] Incident response procedures documented

### Monitoring & Alerting
- [ ] Health monitoring dashboards configured
- [ ] Performance metrics collection verified
- [ ] Auto-scaling alerts configured
- [ ] Compliance violation alerts setup
- [ ] Multi-region monitoring active

### Performance & Scaling
- [ ] Cache warming strategies implemented
- [ ] Database connection pooling configured
- [ ] Load balancing tested under load
- [ ] Auto-scaling triggers verified
- [ ] Regional failover performance validated

## API Usage Examples

### Complete TERRAGON-Enhanced Workflow
```python
import asyncio
from testgen_copilot import (
    TestGenOrchestrator, set_global_locale, log_data_processing,
    DataClassification, ProcessingPurpose, get_performance_optimizer
)

async def generate_tests_with_terragon():
    # Set locale for German market
    set_global_locale("de_DE")
    
    # Log data processing for GDPR compliance
    log_data_processing(
        DataClassification.INTERNAL,
        ProcessingPurpose.TESTING,
        data_subject_id="de_user_123"
    )
    
    # Use performance optimization
    optimizer = get_performance_optimizer()
    
    with optimizer.optimize_operation("test_generation", cacheable=True):
        # Generate tests with all enhancements active
        orchestrator = TestGenOrchestrator()
        result = await orchestrator.process_file(
            "beispiel.py", 
            "/output",
            user_location="DE"
        )
    
    # Get comprehensive performance report
    perf_report = optimizer.get_comprehensive_report()
    
    return result, perf_report

# Run with all TERRAGON enhancements
result, report = asyncio.run(generate_tests_with_terragon())
```

### Multi-Region Data Storage with Compliance
```python
from testgen_copilot.multi_region import store_data_globally
from testgen_copilot.compliance import DataClassification

# Store user data in GDPR-compliant region
location = store_data_globally(
    data_id="user_preferences_123",
    content={
        "theme": "dark", 
        "language": "de",
        "privacy_settings": {"analytics": False}
    },
    data_classification=DataClassification.PERSONAL,
    user_location="DE"  # Will automatically store in EU region
)

print(f"Data stored in region: {location.primary_region.value}")
print(f"Replicated to: {[r.value for r in location.replicated_regions]}")
```

### Health Monitoring Integration
```python
from testgen_copilot.monitoring import get_health_monitor

# Get comprehensive system health
monitor = get_health_monitor()
health = monitor.get_comprehensive_health()

print(f"System Health: {health['overall_status']}")
print(f"CPU Usage: {health['system_metrics']['cpu_percent']}%")
print(f"Memory Usage: {health['system_metrics']['memory_percent']}%")
print(f"Active Operations: {health['application_metrics']['active_operations']}")

# Check for active alerts
alerts = monitor.get_active_alerts()
if alerts:
    print(f"Active alerts: {len(alerts)}")
    for alert in alerts:
        print(f"- {alert.message} (Severity: {alert.severity})")
```

## Troubleshooting

### Common Issues

1. **Import Errors for New Components**
   ```bash
   # Ensure all TERRAGON dependencies are installed
   pip install -e ".[dev,api,monitoring,compliance,i18n,scaling]"
   ```

2. **Monitoring Not Starting**
   ```python
   # Check if monitoring is properly initialized
   from testgen_copilot.monitoring import get_health_monitor
   
   monitor = get_health_monitor()
   print(f"Monitoring status: {monitor.is_monitoring_active()}")
   ```

3. **Auto-Scaling Issues**
   ```python
   # Verify auto-scaler configuration
   from testgen_copilot.auto_scaling import get_auto_scaler, ScalingPolicy
   
   scaler = get_auto_scaler(ScalingPolicy.BALANCED)
   status = scaler.get_scaling_report()
   print(f"Workers: {len(scaler.load_balancer.workers)}")
   ```

4. **Compliance Framework Errors**
   ```python
   # Check compliance engine status
   from testgen_copilot.compliance import get_compliance_engine
   
   engine = get_compliance_engine()
   report = engine.generate_compliance_report()
   print(f"Active frameworks: {report['active_frameworks']}")
   ```

5. **Multi-Region Connection Issues**
   ```python
   # Verify region health
   from testgen_copilot.multi_region import get_multi_region_manager
   
   manager = get_multi_region_manager()
   status = manager.get_region_status()
   
   for region in status['regions']:
       print(f"Region {region['region']}: {region.get('status', 'unknown')}")
   ```

### Performance Tuning

1. **Memory Optimization**
   ```python
   # Adjust cache sizes based on available memory
   from testgen_copilot.performance_optimizer import get_performance_optimizer
   
   optimizer = get_performance_optimizer()
   optimizer.cache.max_memory_mb = 200  # Reduce for smaller instances
   optimizer.cache.max_disk_mb = 1000   # Adjust disk cache
   ```

2. **Auto-Scaling Sensitivity**
   ```python
   # Tune scaling thresholds
   from testgen_copilot.auto_scaling import get_auto_scaler, ScalingPolicy
   
   scaler = get_auto_scaler(ScalingPolicy.CONSERVATIVE)  # Less aggressive scaling
   ```

3. **Monitoring Frequency**
   ```python
   # Adjust monitoring intervals for better performance
   from testgen_copilot.monitoring import get_health_monitor
   
   monitor = get_health_monitor()
   monitor.check_interval_seconds = 60.0  # Check every minute instead of 30 seconds
   ```

## Migration from Previous Versions

### Upgrading Existing Installations
```bash
# Backup existing configuration
cp .env .env.backup

# Update dependencies
pip install -e ".[all]" --upgrade

# Run database migrations if needed
python -c "from testgen_copilot.database import run_migrations; run_migrations()"
```

### Configuration Migration
```python
# Update existing configuration to include TERRAGON features
from testgen_copilot import (
    get_health_monitor, get_compliance_engine, 
    get_multi_region_manager, set_global_locale
)

# Initialize new components with existing settings
set_global_locale("en_US")  # or your current locale

# Enable health monitoring
monitor = get_health_monitor()
monitor.start_monitoring()

# Configure compliance based on your jurisdiction
engine = get_compliance_engine()
if "EU" in your_jurisdictions:
    engine.enable_compliance_framework("GDPR")
if "US-CA" in your_jurisdictions:
    engine.enable_compliance_framework("CCPA")
```

## Support and Maintenance

### Regular Maintenance Tasks
- Monitor system health and performance metrics
- Review compliance reports and audit logs
- Test auto-scaling behavior under different loads
- Verify multi-region failover procedures
- Update translations for new locales
- Review and update security configurations

### Getting Help
- **Documentation**: See `TERRAGON_IMPLEMENTATION.md` for technical details
- **GitHub Issues**: Report bugs and feature requests
- **Performance Issues**: Check monitoring dashboards and performance reports
- **Compliance Questions**: Review compliance engine reports and logs

---

**Status**: Production ready with comprehensive TERRAGON SDLC enhancements including robustness, performance optimization, auto-scaling, global compliance, and multi-region deployment capabilities.