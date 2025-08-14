#!/usr/bin/env python3
"""
Performance Report Generator

Generates comprehensive performance reports from benchmark results
for the Autonomous SDLC implementation.
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


def generate_performance_report(benchmark_file: Path) -> str:
    """Generate performance report from benchmark JSON file"""
    
    if not benchmark_file.exists():
        return "# Performance Report\n\n❌ Benchmark file not found\n"
    
    try:
        with open(benchmark_file) as f:
            benchmark_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return f"# Performance Report\n\n❌ Error reading benchmark file: {e}\n"
    
    report_lines = [
        "# 🚀 Autonomous SDLC Performance Report",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 📊 Executive Summary",
        ""
    ]
    
    # Extract benchmark results
    benchmarks = benchmark_data.get("benchmarks", [])
    
    if not benchmarks:
        report_lines.extend([
            "❌ No benchmark results found",
            ""
        ])
        return "\n".join(report_lines)
    
    # Calculate summary statistics
    total_benchmarks = len(benchmarks)
    avg_duration = sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / total_benchmarks
    fastest_test = min(benchmarks, key=lambda x: x.get("stats", {}).get("min", float('inf')))
    slowest_test = max(benchmarks, key=lambda x: x.get("stats", {}).get("max", 0))
    
    report_lines.extend([
        f"- **Total Benchmarks**: {total_benchmarks}",
        f"- **Average Duration**: {avg_duration:.3f}s",
        f"- **Fastest Test**: {fastest_test.get('name', 'Unknown')} ({fastest_test.get('stats', {}).get('min', 0):.3f}s)",
        f"- **Slowest Test**: {slowest_test.get('name', 'Unknown')} ({slowest_test.get('stats', {}).get('max', 0):.3f}s)",
        "",
        "## 🎯 Performance Targets",
        ""
    ])
    
    # Define performance targets
    targets = {
        "sdlc_initialization": {"target": 2.0, "unit": "seconds"},
        "quality_gate_validation": {"target": 10.0, "unit": "seconds"},  
        "progressive_enhancement": {"target": 5.0, "unit": "seconds"},
        "pattern_recognition": {"target": 0.1, "unit": "seconds"},
        "auto_scaling_decision": {"target": 0.5, "unit": "seconds"}
    }
    
    # Check performance against targets
    performance_status = {}
    
    for benchmark in benchmarks:
        test_name = benchmark.get("name", "")
        mean_time = benchmark.get("stats", {}).get("mean", 0)
        
        # Match test to target category
        target_key = None
        for target_name in targets:
            if target_name.replace("_", "") in test_name.lower().replace("_", ""):
                target_key = target_name
                break
        
        if target_key:
            target_time = targets[target_key]["target"]
            unit = targets[target_key]["unit"]
            
            status = "✅ PASS" if mean_time <= target_time else "❌ FAIL"
            performance_status[target_key] = {
                "status": status,
                "actual": mean_time,
                "target": target_time,
                "unit": unit,
                "test_name": test_name
            }
    
    # Add performance target results
    for target_name, target_info in targets.items():
        if target_name in performance_status:
            status_info = performance_status[target_name]
            report_lines.extend([
                f"- **{target_name.replace('_', ' ').title()}**: {status_info['status']}",
                f"  - Actual: {status_info['actual']:.3f}{status_info['unit']}",
                f"  - Target: ≤{status_info['target']:.1f}{status_info['unit']}",
                f"  - Test: {status_info['test_name']}"
            ])
        else:
            report_lines.extend([
                f"- **{target_name.replace('_', ' ').title()}**: ⚠️ NOT TESTED"
            ])
    
    report_lines.extend([
        "",
        "## 📈 Detailed Results",
        ""
    ])
    
    # Add detailed benchmark results
    for benchmark in sorted(benchmarks, key=lambda x: x.get("stats", {}).get("mean", 0)):
        name = benchmark.get("name", "Unknown")
        stats = benchmark.get("stats", {})
        
        min_time = stats.get("min", 0)
        max_time = stats.get("max", 0) 
        mean_time = stats.get("mean", 0)
        stddev = stats.get("stddev", 0)
        rounds = stats.get("rounds", 0)
        
        report_lines.extend([
            f"### {name}",
            "",
            f"- **Mean**: {mean_time:.3f}s",
            f"- **Min**: {min_time:.3f}s", 
            f"- **Max**: {max_time:.3f}s",
            f"- **Std Dev**: {stddev:.3f}s",
            f"- **Rounds**: {rounds}",
            ""
        ])
    
    # Add performance analysis
    report_lines.extend([
        "## 🔍 Performance Analysis",
        ""
    ])
    
    # Identify performance bottlenecks
    slow_tests = [b for b in benchmarks if b.get("stats", {}).get("mean", 0) > 1.0]
    fast_tests = [b for b in benchmarks if b.get("stats", {}).get("mean", 0) < 0.1]
    
    if slow_tests:
        report_lines.extend([
            "### ⚠️ Performance Bottlenecks",
            ""
        ])
        for test in slow_tests[:5]:  # Top 5 slowest
            name = test.get("name", "Unknown")
            mean_time = test.get("stats", {}).get("mean", 0)
            report_lines.append(f"- {name}: {mean_time:.3f}s")
        report_lines.append("")
    
    if fast_tests:
        report_lines.extend([
            "### ⚡ High Performance Tests",
            ""
        ])
        for test in fast_tests[:5]:  # Top 5 fastest  
            name = test.get("name", "Unknown")
            mean_time = test.get("stats", {}).get("mean", 0)
            report_lines.append(f"- {name}: {mean_time:.3f}s")
        report_lines.append("")
    
    # Add recommendations
    report_lines.extend([
        "## 🎯 Recommendations",
        ""
    ])
    
    if avg_duration > 2.0:
        report_lines.append("- ⚡ Consider optimizing slow test cases for faster CI/CD pipeline")
    
    if any("FAIL" in status["status"] for status in performance_status.values()):
        report_lines.append("- 🔧 Address performance target failures to improve system responsiveness")
    
    high_variance_tests = [
        b for b in benchmarks 
        if b.get("stats", {}).get("stddev", 0) > b.get("stats", {}).get("mean", 0) * 0.3
    ]
    
    if high_variance_tests:
        report_lines.append("- 📊 Investigate high variance tests for consistency improvements")
    
    if len(benchmarks) < 10:
        report_lines.append("- 🧪 Consider adding more performance benchmarks for comprehensive coverage")
    
    # Add footer
    report_lines.extend([
        "",
        "---",
        "",
        "**Note**: This report is automatically generated from pytest-benchmark results.",
        "Performance targets are based on production requirements for the Autonomous SDLC system.",
        ""
    ])
    
    return "\n".join(report_lines)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python generate_performance_report.py <benchmark-results.json>")
        sys.exit(1)
    
    benchmark_file = Path(sys.argv[1])
    report = generate_performance_report(benchmark_file)
    print(report)


if __name__ == "__main__":
    main()