"""Production Deployment - Demonstration

This script demonstrates the comprehensive production deployment capabilities
including multi-strategy deployments, blue-green, canary, rolling updates,
and production readiness validation.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from testgen_copilot.production_deployment_system import (
    ProductionDeploymentSystem,
    DeploymentConfiguration,
    DeploymentTarget,
    DeploymentEnvironment,
    DeploymentStrategy,
    DeploymentStatus
)

async def demonstrate_production_deployment():
    """Demonstrate production deployment capabilities."""
    
    print("🚀 Production Deployment System - COMPREHENSIVE DEMONSTRATION")
    print("=" * 75)
    
    # 1. Initialize production deployment system
    print("\n1. Initializing Production Deployment System...")
    system = ProductionDeploymentSystem(
        dry_run=True,  # Safe demonstration mode
        max_parallel_deployments=3
    )
    
    print(f"✅ Production Deployment System initialized:")
    print(f"   - Dry run mode: {system.dry_run}")
    print(f"   - Max parallel deployments: {system.max_parallel_deployments}")
    print(f"   - Infrastructure components enabled:")
    for component, config in system.infrastructure_config.items():
        if isinstance(config, dict) and config.get("enabled", False):
            print(f"      ✅ {component.title()}")
    
    print(f"   - Global regions configured: {len(system.global_regions)}")
    for region in system.global_regions:
        primary_indicator = "🌟 PRIMARY" if region["primary"] else f"⚖️  Weight: {region['weight']}"
        print(f"      📍 {region['name']} - {primary_indicator}")
    
    # 2. Demonstrate Blue-Green Deployment
    print("\n2. Demonstrating Blue-Green Deployment Strategy...")
    
    blue_green_config = DeploymentConfiguration(
        application_name="testgen-copilot-web",
        version="2.1.0",
        image="gcr.io/terragon-labs/testgen-copilot:2.1.0",
        strategy=DeploymentStrategy.BLUE_GREEN,
        targets=[
            DeploymentTarget(
                environment=DeploymentEnvironment.STAGING,
                region="us-east-1",
                cluster="staging-cluster",
                namespace="testgen-copilot",
                replicas=3,
                environment_variables={
                    "NODE_ENV": "staging",
                    "LOG_LEVEL": "info",
                    "DATABASE_URL": "postgres://staging-db:5432/testgen"
                }
            )
        ],
        success_criteria={
            "error_rate_threshold": 0.01,
            "response_time_p95_ms": 800,
            "availability_percentage": 99.9
        },
        pre_deployment_hooks=[
            "run_pre_deployment_tests.sh",
            "backup_database.sh",
            "notify_team_deployment_start.sh"
        ],
        post_deployment_hooks=[
            "run_smoke_tests.sh",
            "update_monitoring_dashboards.sh",
            "notify_team_deployment_complete.sh"
        ]
    )
    
    print(f"   🔵🟢 Deploying {blue_green_config.application_name} v{blue_green_config.version}")
    print(f"   📋 Strategy: {blue_green_config.strategy.value}")
    print(f"   🎯 Targets: {len(blue_green_config.targets)} environments")
    print(f"   📊 Success criteria: {len(blue_green_config.success_criteria)} metrics")
    
    start_time = time.time()
    blue_green_result = await system.deploy_application(blue_green_config, auto_approve=True)
    blue_green_duration = time.time() - start_time
    
    print(f"   ✅ Blue-green deployment completed in {blue_green_duration:.2f} seconds")
    print(f"   📊 Status: {blue_green_result.status.value}")
    print(f"   🏥 Health checks: {len(blue_green_result.health_checks)} performed")
    print(f"   📈 Metrics collected: {len(blue_green_result.metrics)} data points")
    
    # 3. Demonstrate Canary Deployment
    print("\n3. Demonstrating Canary Deployment Strategy...")
    
    canary_config = DeploymentConfiguration(
        application_name="testgen-copilot-api",
        version="2.2.0-beta",
        image="gcr.io/terragon-labs/testgen-copilot-api:2.2.0-beta",
        strategy=DeploymentStrategy.CANARY,
        targets=[
            DeploymentTarget(
                environment=DeploymentEnvironment.PRODUCTION,
                region="us-west-2",
                cluster="prod-cluster-west",
                namespace="testgen-copilot",
                replicas=5,
                environment_variables={
                    "NODE_ENV": "production",
                    "LOG_LEVEL": "warn",
                    "DATABASE_URL": "postgres://prod-db:5432/testgen",
                    "REDIS_URL": "redis://prod-cache:6379"
                }
            )
        ],
        canary_percentage=5.0,  # Start with 5% traffic
        success_criteria={
            "error_rate_threshold": 0.005,  # Stricter for production
            "response_time_p95_ms": 500,
            "availability_percentage": 99.95
        },
        timeout_seconds=1200  # 20 minutes for canary
    )
    
    print(f"   🐤 Deploying {canary_config.application_name} v{canary_config.version}")
    print(f"   📋 Strategy: {canary_config.strategy.value}")
    print(f"   📊 Initial canary traffic: {canary_config.canary_percentage}%")
    print(f"   ⏱️  Timeout: {canary_config.timeout_seconds} seconds")
    
    start_time = time.time()
    canary_result = await system.deploy_application(canary_config, auto_approve=True)
    canary_duration = time.time() - start_time
    
    print(f"   ✅ Canary deployment completed in {canary_duration:.2f} seconds")
    print(f"   📊 Status: {canary_result.status.value}")
    print(f"   📈 Metrics monitored at each traffic level:")
    for metric, value in canary_result.metrics.items():
        if "canary" in metric:
            print(f"      📊 {metric}: {value:.3f}")
    
    # 4. Demonstrate Rolling Update Deployment
    print("\n4. Demonstrating Rolling Update Deployment Strategy...")
    
    rolling_config = DeploymentConfiguration(
        application_name="testgen-copilot-worker",
        version="2.1.1",
        image="gcr.io/terragon-labs/testgen-copilot-worker:2.1.1",
        strategy=DeploymentStrategy.ROLLING_UPDATE,
        targets=[
            DeploymentTarget(
                environment=DeploymentEnvironment.PRODUCTION,
                region="eu-west-1",
                cluster="prod-cluster-eu",
                namespace="testgen-copilot",
                replicas=6,  # Will update in batches of 2
                resource_limits={
                    "cpu": "500m",
                    "memory": "1Gi"
                },
                environment_variables={
                    "WORKER_THREADS": "4",
                    "QUEUE_URL": "redis://prod-queue:6379",
                    "LOG_LEVEL": "info"
                }
            )
        ]
    )
    
    print(f"   🔄 Deploying {rolling_config.application_name} v{rolling_config.version}")
    print(f"   📋 Strategy: {rolling_config.strategy.value}")
    print(f"   🎯 Replicas: {rolling_config.targets[0].replicas} (batched updates)")
    print(f"   💾 Resource limits: {rolling_config.targets[0].resource_limits}")
    
    start_time = time.time()
    rolling_result = await system.deploy_application(rolling_config, auto_approve=True)
    rolling_duration = time.time() - start_time
    
    print(f"   ✅ Rolling update completed in {rolling_duration:.2f} seconds")
    print(f"   📊 Status: {rolling_result.status.value}")
    
    # Show batch update details
    for target_key, target_result in rolling_result.target_results.items():
        if "batches_updated" in target_result:
            print(f"   📦 Batches updated: {target_result['batches_updated']}")
    
    # 5. Demonstrate Multi-Region Deployment
    print("\n5. Demonstrating Multi-Region Global Deployment...")
    
    global_config = DeploymentConfiguration(
        application_name="testgen-copilot-global",
        version="2.0.0",
        image="gcr.io/terragon-labs/testgen-copilot:2.0.0",
        strategy=DeploymentStrategy.BLUE_GREEN,
        targets=[
            DeploymentTarget(
                environment=DeploymentEnvironment.PRODUCTION,
                region="us-east-1",
                cluster="prod-us-east",
                namespace="testgen-copilot",
                replicas=4,
                environment_variables={"REGION": "us-east-1", "PRIMARY_REGION": "true"}
            ),
            DeploymentTarget(
                environment=DeploymentEnvironment.PRODUCTION,
                region="us-west-2",
                cluster="prod-us-west",
                namespace="testgen-copilot",
                replicas=3,
                environment_variables={"REGION": "us-west-2", "PRIMARY_REGION": "false"}
            ),
            DeploymentTarget(
                environment=DeploymentEnvironment.PRODUCTION,
                region="eu-west-1",
                cluster="prod-eu-west",
                namespace="testgen-copilot",
                replicas=2,
                environment_variables={"REGION": "eu-west-1", "PRIMARY_REGION": "false"}
            )
        ]
    )
    
    print(f"   🌍 Global deployment: {global_config.application_name} v{global_config.version}")
    print(f"   📋 Strategy: {global_config.strategy.value}")
    print(f"   🎯 Regions: {len(global_config.targets)} global regions")
    
    total_replicas = sum(target.replicas for target in global_config.targets)
    print(f"   📊 Total replicas across all regions: {total_replicas}")
    
    for target in global_config.targets:
        primary_indicator = "🌟" if target.environment_variables.get("PRIMARY_REGION") == "true" else "🌐"
        print(f"      {primary_indicator} {target.region}: {target.replicas} replicas")
    
    start_time = time.time()
    global_result = await system.deploy_application(global_config, auto_approve=True)
    global_duration = time.time() - start_time
    
    print(f"   ✅ Global deployment completed in {global_duration:.2f} seconds")
    print(f"   📊 Status: {global_result.status.value}")
    print(f"   🌍 Regions deployed: {len(global_result.target_results)}")
    
    # 6. Generate Kubernetes Manifests
    print("\n6. Generating Kubernetes Deployment Manifests...")
    
    manifests = await system.generate_deployment_manifest(global_config)
    
    print(f"   📄 Generated manifests for {len(manifests)} deployment targets:")
    for target_key, manifest_set in manifests.items():
        print(f"      📦 {target_key}:")
        print(f"         - Deployment manifest: {manifest_set['deployment']['metadata']['name']}")
        print(f"         - Service manifest: {manifest_set['service']['metadata']['name']}")
        print(f"         - Namespace: {manifest_set['deployment']['metadata']['namespace']}")
        print(f"         - Replicas: {manifest_set['deployment']['spec']['replicas']}")
    
    # 7. Deployment Summary and Reports
    print("\n7. Deployment Summary and Reporting...")
    
    all_deployments = [blue_green_result, canary_result, rolling_result, global_result]
    
    print(f"   📊 Deployment Session Summary:")
    print(f"      - Total deployments: {len(all_deployments)}")
    print(f"      - Successful deployments: {sum(1 for d in all_deployments if d.status == DeploymentStatus.COMPLETED)}")
    print(f"      - Failed deployments: {sum(1 for d in all_deployments if d.status == DeploymentStatus.FAILED)}")
    print(f"      - Total targets deployed: {sum(len(d.target_results) for d in all_deployments)}")
    print(f"      - Total health checks: {sum(len(d.health_checks) for d in all_deployments)}")
    
    # Generate deployment reports
    print(f"\n   📄 Generating deployment reports...")
    for i, deployment in enumerate(all_deployments, 1):
        report_path = await system.export_deployment_report(
            deployment, 
            Path(f"deployment_report_{i}_{deployment.deployment_id}.json")
        )
        print(f"      📄 Report {i}: {report_path}")
    
    # 8. Deployment Strategies Comparison
    print("\n8. Deployment Strategies Performance Comparison...")
    
    strategy_performance = {}
    for deployment in all_deployments:
        strategy = deployment.configuration.strategy.value
        duration = (deployment.end_time - deployment.start_time).total_seconds()
        
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(duration)
    
    print(f"   ⚡ Strategy Performance Analysis:")
    for strategy, durations in strategy_performance.items():
        avg_duration = sum(durations) / len(durations)
        print(f"      📊 {strategy.replace('_', ' ').title()}:")
        print(f"         - Average duration: {avg_duration:.2f}s")
        print(f"         - Deployments: {len(durations)}")
        print(f"         - Use case: {_get_strategy_use_case(strategy)}")
    
    # 9. Production Readiness Assessment
    print("\n9. Production Readiness Assessment...")
    
    readiness_score = 0
    readiness_factors = []
    
    # Check deployment success rate
    success_rate = sum(1 for d in all_deployments if d.status == DeploymentStatus.COMPLETED) / len(all_deployments)
    if success_rate >= 0.9:
        readiness_score += 25
        readiness_factors.append(f"✅ High deployment success rate: {success_rate:.1%}")
    else:
        readiness_factors.append(f"⚠️ Low deployment success rate: {success_rate:.1%}")
    
    # Check multi-region deployment
    has_multi_region = any(len(d.target_results) > 1 for d in all_deployments)
    if has_multi_region:
        readiness_score += 25
        readiness_factors.append("✅ Multi-region deployment capability demonstrated")
    else:
        readiness_factors.append("⚠️ Single-region deployments only")
    
    # Check advanced strategies
    advanced_strategies = [DeploymentStrategy.BLUE_GREEN, DeploymentStrategy.CANARY]
    has_advanced = any(d.configuration.strategy in advanced_strategies for d in all_deployments)
    if has_advanced:
        readiness_score += 25
        readiness_factors.append("✅ Advanced deployment strategies (Blue-Green, Canary)")
    else:
        readiness_factors.append("⚠️ Only basic deployment strategies")
    
    # Check health monitoring
    total_health_checks = sum(len(d.health_checks) for d in all_deployments)
    if total_health_checks > 0:
        readiness_score += 25
        readiness_factors.append(f"✅ Comprehensive health monitoring: {total_health_checks} checks")
    else:
        readiness_factors.append("⚠️ No health monitoring configured")
    
    print(f"   🎯 Production Readiness Score: {readiness_score}/100")
    print(f"   📋 Readiness Factors:")
    for factor in readiness_factors:
        print(f"      {factor}")
    
    if readiness_score >= 80:
        readiness_status = "🎉 PRODUCTION READY"
    elif readiness_score >= 60:
        readiness_status = "⚠️ NEEDS IMPROVEMENT"
    else:
        readiness_status = "❌ NOT READY"
    
    print(f"   🏆 Overall Assessment: {readiness_status}")
    
    # 10. Recommendations
    print(f"\n10. Production Deployment Recommendations...")
    
    recommendations = []
    
    if success_rate < 1.0:
        recommendations.append("🔧 Investigate and fix deployment failures")
    
    if not has_multi_region:
        recommendations.append("🌍 Implement multi-region deployment for high availability")
    
    recommendations.extend([
        "📊 Set up comprehensive monitoring and alerting",
        "🔄 Implement automated rollback procedures",
        "🧪 Add comprehensive smoke tests and health checks",
        "📝 Document deployment procedures and runbooks",
        "🔒 Implement security scanning in deployment pipeline",
        "⚡ Optimize deployment performance for faster releases"
    ])
    
    print(f"   💡 Top Recommendations:")
    for i, rec in enumerate(recommendations[:6], 1):
        print(f"      {i}. {rec}")
    
    # Clean up
    system.cleanup()
    
    print("\n🚀 Production Deployment System - DEMONSTRATION COMPLETE!")
    print("   🔵🟢 Blue-Green deployments: Zero-downtime production releases")
    print("   🐤 Canary deployments: Risk-free progressive rollouts")
    print("   🔄 Rolling updates: Seamless application updates")
    print("   🌍 Multi-region deployments: Global high-availability")
    print("   📊 Comprehensive monitoring: Real-time deployment health")
    print("   ⚡ Auto-rollback: Automatic failure recovery")
    print("   📄 Production manifests: Infrastructure-as-code ready")


def _get_strategy_use_case(strategy: str) -> str:
    """Get use case description for deployment strategy"""
    use_cases = {
        "blue_green": "Zero-downtime releases with instant rollback",
        "canary": "Risk mitigation for new features with gradual rollout",
        "rolling_update": "Standard updates with minimal resource usage",
        "recreate": "Simple deployments with acceptable downtime"
    }
    return use_cases.get(strategy, "General purpose deployment")


if __name__ == "__main__":
    asyncio.run(demonstrate_production_deployment())