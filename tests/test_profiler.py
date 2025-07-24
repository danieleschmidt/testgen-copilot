"""Tests for the TestGenerator profiler."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import time

from src.testgen_copilot.profiler import GeneratorProfiler, ProfileMetrics, create_large_test_project
from src.testgen_copilot.generator import TestGenerator, GenerationConfig


class TestGeneratorProfiler:
    """Test the GeneratorProfiler functionality."""
    
    def test_profiler_initialization(self):
        """Test profiler initializes with correct defaults."""
        profiler = GeneratorProfiler()
        
        assert profiler.enable_cprofile is True
        assert profiler.sample_interval == 0.1
        assert profiler.profile_data is None
        assert isinstance(profiler.metrics, ProfileMetrics)
    
    def test_profile_operation_context_manager(self):
        """Test that profile_operation works as a context manager."""
        profiler = GeneratorProfiler(enable_cprofile=False)  # Disable for simpler testing
        
        with profiler.profile_operation("test_operation"):
            time.sleep(0.01)  # Small delay to measure
        
        # Should have recorded some time
        assert profiler.metrics.total_time > 0
        assert profiler.metrics.total_time < 0.1  # Should be very small
    
    def test_profile_single_file(self, tmp_path):
        """Test profiling a single file generation."""
        # Create test file
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def add(a: int, b: int) -> int:
    return a + b

def multiply(x: int, y: int) -> int:
    return x * y
""")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        profiler = GeneratorProfiler(enable_cprofile=False)
        generator = TestGenerator(GenerationConfig(language="python"))
        
        metrics = profiler.profile_single_file(test_file, generator, output_dir)
        
        # Verify metrics were collected
        assert metrics.files_processed == 1
        assert metrics.total_time > 0
        assert metrics.average_file_size_kb > 0
        assert metrics.peak_memory_mb > 0
    
    def test_profile_file_batch(self, tmp_path):
        """Test profiling a batch of files."""
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"test_module_{i}.py"
            test_file.write_text(f"""
def function_{i}(x: int) -> int:
    return x * {i + 1}
""")
            files.append(test_file)
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        profiler = GeneratorProfiler(enable_cprofile=False)
        generator = TestGenerator(GenerationConfig(language="python"))
        
        metrics = profiler.profile_file_batch(files, generator, output_dir)
        
        # Verify batch metrics
        assert metrics.files_processed == 3
        assert metrics.total_time > 0
        assert metrics.files_per_second > 0
        assert metrics.average_file_size_kb > 0
    
    def test_generate_report(self, tmp_path):
        """Test generating a performance report."""
        profiler = GeneratorProfiler()
        
        # Simulate some metrics
        profiler.metrics.total_time = 2.5
        profiler.metrics.files_processed = 10
        profiler.metrics.functions_found = 25
        profiler.metrics.tests_generated = 25
        profiler.metrics.peak_memory_mb = 150.5
        profiler.metrics.memory_delta_mb = 50.2
        profiler.metrics.files_per_second = 4.0
        profiler.metrics.generation_time = 2.0
        profiler.metrics.file_io_time = 0.3
        profiler.metrics.parsing_time = 0.2
        
        # Add some hot spots
        profiler.metrics.hot_spots = [
            ("generate_tests", 1.5, 10),
            ("_parse_functions", 0.8, 25),
            ("safe_read_file", 0.2, 10),
        ]
        
        report = profiler.generate_report()
        
        # Verify report content
        assert "# TestGenerator Performance Profile Report" in report
        assert "Total Processing Time**: 2.50s" in report
        assert "Files Processed**: 10" in report
        assert "Peak Memory Usage**: 150.5MB" in report
        assert "Files per Second**: 4.0" in report
        assert "Performance Hot Spots" in report
        assert "generate_tests" in report
        assert "Performance Recommendations" in report
    
    def test_generate_report_with_file_output(self, tmp_path):
        """Test saving report to a file."""
        profiler = GeneratorProfiler()
        profiler.metrics.total_time = 1.0
        profiler.metrics.files_processed = 5
        
        output_file = tmp_path / "profile_report.md"
        report = profiler.generate_report(output_file)
        
        # Verify file was created
        assert output_file.exists()
        file_content = output_file.read_text()
        assert file_content == report
        assert "TestGenerator Performance Profile Report" in file_content
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        profiler = GeneratorProfiler()
        
        # Memory usage should be trackable
        memory_usage = profiler._get_memory_usage()
        assert memory_usage >= 0  # Should be positive or zero
        assert isinstance(memory_usage, float)
    
    def test_performance_recommendations(self):
        """Test performance recommendation generation."""
        profiler = GeneratorProfiler()
        
        # Test high memory usage recommendation
        profiler.metrics.peak_memory_mb = 600
        profiler.metrics.total_time = 1.0
        recommendations = profiler._generate_recommendations()
        assert "High Memory Usage" in recommendations
        
        # Test memory growth recommendation
        profiler.metrics.memory_delta_mb = 150
        recommendations = profiler._generate_recommendations()
        assert "Memory Growth" in recommendations
        
        # Test low throughput recommendation
        profiler.metrics.files_processed = 2
        recommendations = profiler._generate_recommendations()
        assert "Low Throughput" in recommendations
        
        # Test I/O bottleneck recommendation
        profiler.metrics.file_io_time = 0.4
        recommendations = profiler._generate_recommendations()
        assert "I/O Bottleneck" in recommendations
    
    def test_hot_spot_extraction_without_cprofile(self):
        """Test that profiler works when cProfile is disabled."""
        profiler = GeneratorProfiler(enable_cprofile=False)
        
        with profiler.profile_operation("test"):
            pass
        
        # Should have no hot spots but still work
        assert profiler.metrics.hot_spots == []
        assert profiler.metrics.total_time > 0
    
    @patch('cProfile.Profile')
    def test_hot_spot_extraction_with_cprofile(self, mock_profile_class):
        """Test hot spot extraction when cProfile is enabled."""
        # Mock cProfile behavior
        mock_profile = Mock()
        mock_profile_class.return_value = mock_profile
        
        # Mock pstats data
        mock_stats = {
            ('testgen_copilot/generator.py', 100, 'generate_tests'): (5, 0, 1.5, 1.5, {}),
            ('testgen_copilot/generator.py', 200, '_parse_functions'): (10, 0, 0.8, 0.8, {}),
        }
        
        profiler = GeneratorProfiler(enable_cprofile=True)
        
        with patch('pstats.Stats') as mock_pstats:
            mock_pstats.return_value.stats = mock_stats
            profiler.profile_data = mock_profile
            profiler._extract_hot_spots()
        
        # Should have extracted hot spots
        assert len(profiler.metrics.hot_spots) == 2
        assert profiler.metrics.hot_spots[0][0].startswith('generate_tests')
        assert profiler.metrics.hot_spots[0][1] == 1.5  # cumulative time
        assert profiler.metrics.hot_spots[0][2] == 5    # call count


class TestLargeProjectCreation:
    """Test the synthetic large project creation utility."""
    
    def test_create_large_test_project_default(self, tmp_path):
        """Test creating a large test project with default parameters."""
        files = create_large_test_project(tmp_path)
        
        # Should create 100 files by default
        assert len(files) == 100
        
        # All files should exist and be Python files
        for file_path in files:
            assert file_path.exists()
            assert file_path.suffix == ".py"
            assert file_path.name.startswith("module_")
        
        # Check content of first file
        first_file = files[0]
        content = first_file.read_text()
        assert "def function_0_0" in content
        assert "class TestClass0" in content
        assert "def process_data" in content
    
    def test_create_large_test_project_custom_size(self, tmp_path):
        """Test creating a large test project with custom size."""
        files = create_large_test_project(tmp_path, num_files=5, lines_per_file=20)
        
        # Should create specified number of files
        assert len(files) == 5
        
        # Check that files contain reasonable content
        for file_path in files:
            content = file_path.read_text()
            # With lines_per_file=20, should have at least 2 functions (20//10 = 2)
            assert content.count("def function_") >= 2
    
    def test_create_large_test_project_directory_creation(self, tmp_path):
        """Test that the large test project creates the directory structure."""
        files = create_large_test_project(tmp_path, num_files=3)
        
        project_dir = tmp_path / "large_test_project"
        assert project_dir.exists()
        assert project_dir.is_dir()
        
        # All files should be in the project directory
        for file_path in files:
            assert file_path.parent == project_dir


class TestProfileMetrics:
    """Test the ProfileMetrics dataclass."""
    
    def test_profile_metrics_initialization(self):
        """Test ProfileMetrics initializes with correct defaults."""
        metrics = ProfileMetrics()
        
        assert metrics.total_time == 0.0
        assert metrics.generation_time == 0.0
        assert metrics.parsing_time == 0.0
        assert metrics.file_io_time == 0.0
        assert metrics.peak_memory_mb == 0.0
        assert metrics.memory_delta_mb == 0.0
        assert metrics.cpu_percent == 0.0
        assert metrics.files_processed == 0
        assert metrics.functions_found == 0
        assert metrics.tests_generated == 0
        assert metrics.average_file_size_kb == 0.0
        assert metrics.files_per_second == 0.0
        assert metrics.functions_per_second == 0.0
        assert metrics.mb_per_second == 0.0
        assert metrics.hot_spots == []
    
    def test_profile_metrics_can_be_modified(self):
        """Test that ProfileMetrics fields can be modified."""
        metrics = ProfileMetrics()
        
        metrics.total_time = 5.5
        metrics.files_processed = 10
        metrics.peak_memory_mb = 100.5
        metrics.hot_spots = [("test_function", 1.0, 5)]
        
        assert metrics.total_time == 5.5
        assert metrics.files_processed == 10
        assert metrics.peak_memory_mb == 100.5
        assert len(metrics.hot_spots) == 1
        assert metrics.hot_spots[0] == ("test_function", 1.0, 5)


if __name__ == "__main__":
    pytest.main([__file__])