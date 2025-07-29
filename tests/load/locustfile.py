#!/usr/bin/env python3
"""Load testing configuration for TestGen Copilot using Locust."""

import json
import random
from pathlib import Path

from locust import HttpUser, task, between


class TestGenUser(HttpUser):
    """Simulated user for load testing TestGen Copilot."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.sample_codes = self._load_sample_codes()
        
    def _load_sample_codes(self):
        """Load sample code snippets for testing."""
        sample_codes = [
            '''
def calculate_discount(price, discount_percent):
    if price < 0:
        raise ValueError("Price cannot be negative")
    return price * (1 - discount_percent / 100)
''',
            '''
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
''',
            '''
class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
''',
            '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
''',
            '''
import hashlib

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', 
                              password.encode('utf-8'), 
                              salt.encode('utf-8'), 
                              100000)
'''
        ]
        return sample_codes
    
    @task(3)
    def generate_tests(self):
        """Test the test generation endpoint."""
        code = random.choice(self.sample_codes)
        
        payload = {
            "code": code,
            "language": "python",
            "test_framework": "pytest",
            "options": {
                "edge_cases": True,
                "error_handling": True,
                "mocking": True
            }
        }
        
        with self.client.post("/api/generate", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def analyze_security(self):
        """Test the security analysis endpoint."""
        code = random.choice(self.sample_codes)
        
        payload = {
            "code": code,
            "language": "python",
            "rules": {
                "sql_injection": True,
                "xss_vulnerabilities": True,
                "authentication_bypass": True,
                "data_exposure": True
            }
        }
        
        with self.client.post("/api/security", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_coverage(self):
        """Test the coverage analysis endpoint."""
        payload = {
            "project_path": "./sample_project",
            "test_path": "./tests",
            "target_coverage": 85
        }
        
        with self.client.post("/api/coverage", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 is OK for non-existent path
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test the health check endpoint."""
        with self.client.get("/health",
                            catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure("Health check failed")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")


class AdminUser(HttpUser):
    """Simulated admin user for admin endpoints."""
    
    wait_time = between(5, 10)  # Admins operate less frequently
    weight = 3  # Lower weight = fewer admin users
    
    @task
    def metrics_endpoint(self):
        """Test metrics collection endpoint."""
        with self.client.get("/admin/metrics",
                            catch_response=True) as response:
            if response.status_code in [200, 401, 403]:  # Auth errors are OK
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task
    def system_status(self):
        """Test system status endpoint."""
        with self.client.get("/admin/status",
                            catch_response=True) as response:
            if response.status_code in [200, 401, 403]:  # Auth errors are OK
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")