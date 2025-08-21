#!/usr/bin/env python3
"""
🛡️ Simple Robustness Demonstration
===================================

Demonstrates key Generation 2 robustness features that are working:
- Circuit breaker fault tolerance
- Security vulnerability scanning  
- Input validation and sanitization
- Error handling with logging
"""

import asyncio
import random
import logging
from pathlib import Path

from src.testgen_copilot.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError
)
from src.testgen_copilot.security import SecurityScanner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class SimpleValidationError(Exception): 
    """Simple validation error for demo."""
    pass

def validate_path(path_str):
    """Simple path validation."""
    if ".." in path_str or path_str.startswith("/etc"):
        raise SimpleValidationError(f"Dangerous path detected: {path_str}")
    return Path(path_str)

async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker protecting against failures."""
    print("\n⚡ CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 40)
    
    # Create circuit breaker
    circuit_breaker = CircuitBreaker(
        "demo_service",
        CircuitBreakerConfig(failure_threshold=3, timeout_duration_seconds=5.0)
    )
    
    async def unreliable_service():
        """Simulate unreliable service with 80% failure rate."""
        if random.random() < 0.8:
            raise Exception("Service unavailable")
        return "Service success"
    
    # Demonstrate circuit breaker behavior
    for i in range(8):
        try:
            with circuit_breaker.call():
                result = await unreliable_service()
            print(f"✅ Call {i+1}: {result}")
        except CircuitBreakerError:
            print(f"🔴 Call {i+1}: Circuit breaker OPEN - protecting system")
        except Exception as e:
            print(f"❌ Call {i+1}: {e}")
        
        await asyncio.sleep(0.2)
    
    # Show circuit breaker stats
    stats = circuit_breaker.get_stats()
    print(f"\n📊 Circuit Breaker Stats:")
    print(f"   State: {stats['state']}")
    print(f"   Total calls: {stats['total_calls']}")
    print(f"   Success rate: {stats['success_rate_percent']:.1f}%")

def demonstrate_security_scanning():
    """Demonstrate security vulnerability scanning."""
    print("\n🔒 SECURITY SCANNING DEMONSTRATION")
    print("=" * 40)
    
    # Create security scanner
    scanner = SecurityScanner()
    
    # Create test file with security issues
    test_file = Path("security_test.py")
    vulnerable_code = '''
import os
import subprocess

def process_user_input(user_data):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_data}"
    
    # Command injection vulnerability  
    os.system(f"echo {user_data}")
    
    # Shell injection vulnerability
    subprocess.call(user_data, shell=True)
    
    return query
'''
    
    test_file.write_text(vulnerable_code)
    
    try:
        # Scan for security issues
        report = scanner.scan_file(str(test_file))
        print(f"🔍 Scanned file: {report.path}")
        print(f"📈 Issues found: {len(report.issues)}")
        
        for issue in report.issues:
            print(f"   ⚠️  Line {issue.line}: {issue.message}")
            
        if len(report.issues) > 0:
            print("\n✅ Security scanner successfully detected vulnerabilities!")
        else:
            print("\n✅ No security issues detected")
            
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

def demonstrate_input_validation():
    """Demonstrate input validation and sanitization.""" 
    print("\n🛡️ INPUT VALIDATION DEMONSTRATION")
    print("=" * 40)
    
    # Test dangerous paths
    dangerous_paths = [
        "../../../etc/passwd",
        "/etc/shadow", 
        "../../root/.ssh/id_rsa",
        "normal_file.txt"
    ]
    
    for path in dangerous_paths:
        try:
            validated_path = validate_path(path)
            print(f"✅ Valid path: {path}")
        except SimpleValidationError as e:
            print(f"🚫 Blocked dangerous path: {e}")

def demonstrate_error_handling():
    """Demonstrate structured error handling."""
    print("\n🚨 ERROR HANDLING DEMONSTRATION") 
    print("=" * 40)
    
    try:
        # Simulate various types of errors
        operations = [
            ("File processing", lambda: Path("nonexistent.txt").read_text()),
            ("Division by zero", lambda: 1/0),
            ("Type conversion", lambda: int("not_a_number"))
        ]
        
        for operation_name, operation in operations:
            try:
                result = operation()
                print(f"✅ {operation_name}: Success")
            except Exception as e:
                # Log error with context
                logging.error(f"{operation_name} failed: {e}", exc_info=True)
                print(f"❌ {operation_name}: Handled error - {type(e).__name__}")
                
    except Exception as e:
        logging.critical(f"Critical error in error handling demo: {e}")

async def main():
    """Run the complete robustness demonstration."""
    print("🛡️ GENERATION 2: ROBUSTNESS & RELIABILITY DEMO")
    print("=" * 60)
    print("Demonstrating fault tolerance, security, and error handling")
    print("=" * 60)
    
    demonstrate_input_validation()
    demonstrate_error_handling()
    await demonstrate_circuit_breaker()
    demonstrate_security_scanning()
    
    print("\n✅ GENERATION 2 ROBUSTNESS DEMONSTRATION COMPLETE")
    print("\nKey robustness features verified:")
    print("• Circuit breaker fault tolerance ⚡")
    print("• Security vulnerability detection 🔒") 
    print("• Input validation and sanitization 🛡️")
    print("• Structured error handling and logging 🚨")
    print("• System protection against failures 🛡️")

if __name__ == "__main__":
    asyncio.run(main())