"""Integration tests for TestGen Copilot API."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from testgen_copilot.api.server import create_app
from testgen_copilot.database import get_database, run_migrations


@pytest.fixture
def app():
    """Create test FastAPI application."""
    # Use in-memory database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = Path(temp_dir) / "test.db"
        
        # Override database path
        with patch('testgen_copilot.database.connection._database_instance', None):
            with patch('testgen_copilot.database.connection.get_database') as mock_get_db:
                mock_get_db.return_value.db_path = test_db_path
                
                app = create_app(debug=True)
                
                # Run migrations for test database
                run_migrations(test_db_path)
                
                yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] is not None
        assert "database" in data
        assert "dependencies" in data
    
    def test_version_info(self, client):
        """Test version information endpoint."""
        response = client.get("/health/version")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        assert "python_version" in data
        assert "platform" in data


class TestAnalysisEndpoints:
    """Test analysis endpoints."""
    
    def test_analyze_file_missing_file(self, client):
        """Test file analysis with missing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            request_data = {
                "file_path": "/nonexistent/file.py",
                "output_dir": temp_dir,
                "language": "python"
            }
            
            response = client.post("/api/v1/analysis/file", json=request_data)
            assert response.status_code == 422  # Validation error
    
    def test_analyze_file_success(self, client):
        """Test successful file analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test Python file
            test_file = Path(temp_dir) / "test_module.py"
            test_file.write_text("""
def add_numbers(a, b):
    '''Add two numbers together.'''
    return a + b

def multiply_numbers(x, y):
    '''Multiply two numbers.'''
    if x == 0 or y == 0:
        return 0
    return x * y
""")
            
            output_dir = Path(temp_dir) / "tests"
            output_dir.mkdir()
            
            request_data = {
                "file_path": str(test_file),
                "output_dir": str(output_dir),
                "language": "python",
                "include_edge_cases": True,
                "include_error_paths": False,
                "include_benchmarks": False,
                "enable_security_scan": False,  # Disable for faster testing
                "enable_coverage_analysis": False,
                "enable_quality_assessment": False
            }
            
            with patch('testgen_copilot.core.TestGenOrchestrator.process_file') as mock_process:
                from testgen_copilot.core import ProcessingResult, ProcessingStatus
                
                # Mock the processing result
                mock_result = ProcessingResult(
                    file_path=test_file,
                    status=ProcessingStatus.COMPLETED,
                    tests_generated=output_dir / "test_test_module.py",
                    processing_time=1.5
                )
                mock_process.return_value = mock_result
                
                response = client.post("/api/v1/analysis/file", json=request_data)
                assert response.status_code == 200
                
                data = response.json()
                assert data["status"] == "completed"
                assert data["file_path"] == str(test_file)
                assert data["language"] == "python"
                assert data["processing_time_ms"] == 1500
                assert data["session_id"] is not None
    
    def test_analyze_project_missing_directory(self, client):
        """Test project analysis with missing directory."""
        request_data = {
            "project_path": "/nonexistent/project",
            "output_dir": "/tmp/tests"
        }
        
        response = client.post("/api/v1/analysis/project", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_analyze_project_success(self, async_client):
        """Test successful project analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            project_dir.mkdir()
            
            # Create test files
            (project_dir / "module1.py").write_text("def func1(): pass")
            (project_dir / "module2.py").write_text("def func2(): pass")
            
            output_dir = Path(temp_dir) / "tests"
            output_dir.mkdir()
            
            request_data = {
                "project_path": str(project_dir),
                "output_dir": str(output_dir),
                "file_patterns": ["*.py"],
                "exclude_patterns": ["test_*"],
                "concurrent_limit": 2,
                "enable_security_scan": False,
                "enable_coverage_analysis": False,
                "enable_quality_assessment": False
            }
            
            with patch('testgen_copilot.core.TestGenOrchestrator.process_project') as mock_process:
                from testgen_copilot.core import ProcessingResult, ProcessingStatus
                
                # Mock the processing results
                mock_results = {
                    str(project_dir / "module1.py"): ProcessingResult(
                        file_path=project_dir / "module1.py",
                        status=ProcessingStatus.COMPLETED,
                        processing_time=0.5
                    ),
                    str(project_dir / "module2.py"): ProcessingResult(
                        file_path=project_dir / "module2.py", 
                        status=ProcessingStatus.COMPLETED,
                        processing_time=0.7
                    )
                }
                mock_process.return_value = mock_results
                
                with patch('testgen_copilot.core.TestGenOrchestrator.generate_comprehensive_report') as mock_report:
                    mock_report.return_value = {
                        "metrics": {
                            "files_analyzed": 2,
                            "tests_generated": 2,
                            "security_issues_found": 0,
                            "average_coverage_percentage": 85.0,
                            "average_quality_score": 78.5,
                            "processing_time_seconds": 1.2
                        },
                        "security_summary": {
                            "critical_issues": 0,
                            "high_issues": 0,
                            "medium_issues": 0,
                            "low_issues": 0
                        },
                        "recommendations": ["Consider adding more edge case tests"]
                    }
                    
                    response = await async_client.post("/api/v1/analysis/project", json=request_data)
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["session"]["status"] == "completed"
                    assert len(data["results"]) == 2
                    assert data["metrics"]["total_files"] == 2
                    assert data["metrics"]["analyzed_files"] == 2
                    assert len(data["recommendations"]) >= 1


class TestSessionEndpoints:
    """Test session management endpoints."""
    
    def test_list_sessions_empty(self, client):
        """Test listing sessions when none exist."""
        response = client.get("/api/v1/sessions/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["size"] == 20
    
    def test_get_nonexistent_session(self, client):
        """Test getting a session that doesn't exist."""
        response = client.get("/api/v1/sessions/nonexistent-id")
        assert response.status_code == 404
        
        data = response.json()
        assert data["error"] is True
        assert "not found" in data["message"].lower()
    
    def test_session_pagination(self, client):
        """Test session listing with pagination parameters."""
        # Test different pagination parameters
        response = client.get("/api/v1/sessions/?page=2&size=10")
        assert response.status_code == 200
        
        response = client.get("/api/v1/sessions/?sort_by=created_at&sort_order=desc")
        assert response.status_code == 200
        
        # Test invalid sort order
        response = client.get("/api/v1/sessions/?sort_order=invalid")
        assert response.status_code == 422


class TestSecurityEndpoints:
    """Test security scanning endpoints."""
    
    def test_scan_missing_file(self, client):
        """Test security scan with missing file."""
        request_data = {
            "file_path": "/nonexistent/file.py"
        }
        
        response = client.post("/api/v1/security/scan", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_scan_file_success(self, client):
        """Test successful security scan."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file with potential security issues
            test_file = Path(temp_dir) / "vulnerable.py"
            test_file.write_text("""
import subprocess
import os

def execute_command(user_input):
    # Potentially vulnerable code
    cmd = f"echo {user_input}"
    subprocess.call(cmd, shell=True)  # Command injection risk
    
def read_file(filename):
    # Path traversal risk
    with open(f"/data/{filename}", "r") as f:
        return f.read()
""")
            
            request_data = {
                "file_path": str(test_file),
                "strict_mode": False,
                "include_low_severity": True
            }
            
            with patch('testgen_copilot.security.SecurityScanner.scan_file') as mock_scan:
                from testgen_copilot.security import SecurityReport, SecurityIssue
                
                # Mock security issues
                mock_issues = [
                    SecurityIssue(
                        rule_id="B602",
                        rule_name="subprocess_shell_true",
                        severity="high",
                        category="injection",
                        description="subprocess call with shell=True identified",
                        file_path=str(test_file),
                        line_number=8,
                        code_snippet="subprocess.call(cmd, shell=True)",
                        recommendation="Use shell=False and pass command as list"
                    )
                ]
                
                mock_report = SecurityReport(
                    file_path=str(test_file),
                    issues=mock_issues,
                    scan_time=0.5
                )
                mock_scan.return_value = mock_report
                
                response = client.post("/api/v1/security/scan", json=request_data)
                assert response.status_code == 200
                
                data = response.json()
                assert data["file_path"] == str(test_file)
                assert data["total_issues"] == 1
                assert len(data["issues"]) == 1
                assert data["issues"][0]["severity"] == "high"
                assert data["issues"][0]["rule_id"] == "B602"


class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON in request."""
        response = client.post(
            "/api/v1/analysis/file",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        request_data = {
            "file_path": "/some/file.py"
            # Missing required output_dir
        }
        
        response = client.post("/api/v1/analysis/file", json=request_data)
        assert response.status_code == 422
        
        data = response.json()
        assert "field required" in str(data).lower()
    
    def test_invalid_enum_values(self, client):
        """Test handling of invalid enum values."""
        response = client.get("/api/v1/sessions/?status=invalid_status")
        assert response.status_code == 400
        
        data = response.json()
        assert "invalid status" in data["message"].lower()
    
    def test_request_id_in_headers(self, client):
        """Test that request ID is included in response headers."""
        response = client.get("/health/")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


class TestMiddleware:
    """Test API middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        # Check for security headers added by middleware
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "X-API-Version" in response.headers
    
    def test_request_logging(self, client):
        """Test that requests are logged."""
        with patch('testgen_copilot.logging_config.get_logger') as mock_logger:
            mock_structured_logger = Mock()
            mock_logger.return_value = mock_structured_logger
            
            response = client.get("/health/")
            assert response.status_code == 200
            
            # Verify logging calls were made
            mock_structured_logger.info.assert_called()
    
    @pytest.mark.skipif(
        True,  # Skip by default since auth is optional
        reason="Authentication is optional and not enabled in tests"
    )
    def test_api_key_authentication(self, client):
        """Test API key authentication."""
        # Test without API key
        response = client.post("/api/v1/analysis/file", json={})
        assert response.status_code == 401
        
        # Test with invalid API key
        headers = {"X-API-Key": "invalid-key"}
        response = client.post("/api/v1/analysis/file", json={}, headers=headers)
        assert response.status_code == 401
        
        # Test with valid API key
        headers = {"X-API-Key": "valid-test-key"}
        response = client.post("/api/v1/analysis/file", json={}, headers=headers)
        assert response.status_code != 401  # Should pass auth but may fail validation


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflows."""
    
    def test_complete_analysis_workflow(self, client):
        """Test a complete analysis workflow from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test project structure
            project_dir = Path(temp_dir) / "test_project"
            project_dir.mkdir()
            
            # Create source file
            source_file = project_dir / "calculator.py"
            source_file.write_text("""
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b

def divide(a, b):
    \"\"\"Divide two numbers.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")
            
            output_dir = Path(temp_dir) / "tests"
            output_dir.mkdir()
            
            # Step 1: Analyze the file
            analysis_request = {
                "file_path": str(source_file),
                "output_dir": str(output_dir),
                "language": "python",
                "enable_security_scan": True,
                "enable_coverage_analysis": False,  # Simplify for test
                "enable_quality_assessment": False
            }
            
            with patch('testgen_copilot.core.TestGenOrchestrator') as mock_orchestrator:
                from testgen_copilot.core import ProcessingResult, ProcessingStatus
                
                # Mock successful analysis
                mock_instance = mock_orchestrator.return_value
                mock_result = ProcessingResult(
                    file_path=source_file,
                    status=ProcessingStatus.COMPLETED,
                    tests_generated=output_dir / "test_calculator.py",
                    processing_time=2.0
                )
                mock_instance.process_file.return_value = mock_result
                
                response = client.post("/api/v1/analysis/file", json=analysis_request)
                assert response.status_code == 200
                
                analysis_data = response.json()
                session_id = analysis_data["session_id"]
                
                # Step 2: Get session details
                response = client.get(f"/api/v1/sessions/{session_id}")
                assert response.status_code == 200
                
                session_data = response.json()
                assert session_data["session_id"] == session_id
                assert session_data["status"] == "completed"
                
                # Step 3: Get session results
                response = client.get(f"/api/v1/sessions/{session_id}/results")
                assert response.status_code == 200
                
                results_data = response.json()
                assert len(results_data) == 1
                assert results_data[0]["file_path"] == str(source_file)
                
                # Step 4: List all sessions (should include our session)
                response = client.get("/api/v1/sessions/")
                assert response.status_code == 200
                
                sessions_data = response.json()
                session_ids = [s["session_id"] for s in sessions_data["items"]]
                assert session_id in session_ids