"""Performance profiling utilities for TestGen Copilot."""

from __future__ import annotations

import cProfile
import pstats
import io
import time
import resource
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
from contextlib import contextmanager

from .logging_config import get_generator_logger


@dataclass
class ProfileMetrics:
    """Metrics collected during profiling."""
    
    # Time metrics
    total_time: float = 0.0
    generation_time: float = 0.0
    parsing_time: float = 0.0
    file_io_time: float = 0.0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # File metrics
    files_processed: int = 0
    functions_found: int = 0
    tests_generated: int = 0
    average_file_size_kb: float = 0.0
    
    # Performance metrics
    files_per_second: float = 0.0
    functions_per_second: float = 0.0
    mb_per_second: float = 0.0
    
    # Hot spots (function call stats)
    hot_spots: List[Tuple[str, float, int]] = field(default_factory=list)


class GeneratorProfiler:
    """Profiler for analyzing TestGenerator performance on large codebases."""
    
    def __init__(self, enable_cprofile: bool = True, sample_interval: float = 0.1):
        self.logger = get_generator_logger()
        self.enable_cprofile = enable_cprofile
        self.sample_interval = sample_interval
        self.profile_data: Optional[cProfile.Profile] = None
        self.metrics = ProfileMetrics()
        self._start_time = 0.0
        self._start_memory = 0.0
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling a specific operation."""
        self.logger.info(f"Starting profiling: {operation_name}")
        
        # Initialize profiling
        if self.enable_cprofile:
            self.profile_data = cProfile.Profile()
            self.profile_data.enable()
        
        # Record start metrics
        self._start_time = time.perf_counter()
        self._start_memory = self._get_memory_usage()
        
        try:
            yield self
        finally:
            # Record end metrics
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            self.metrics.total_time = end_time - self._start_time
            self.metrics.memory_delta_mb = end_memory - self._start_memory
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, end_memory)
            
            # Disable profiling
            if self.enable_cprofile and self.profile_data:
                self.profile_data.disable()
                self._extract_hot_spots()
            
            self._calculate_performance_metrics()
            self.logger.info(f"Completed profiling: {operation_name}", {
                "total_time": f"{self.metrics.total_time:.2f}s",
                "memory_delta": f"{self.metrics.memory_delta_mb:.1f}MB",
                "files_per_second": f"{self.metrics.files_per_second:.1f}",
                "peak_memory": f"{self.metrics.peak_memory_mb:.1f}MB"
            })
    
    def profile_file_batch(self, file_paths: List[Path], generator, output_dir: Path) -> ProfileMetrics:
        """Profile generator performance on a batch of files."""
        self.logger.info(f"Profiling batch of {len(file_paths)} files")
        
        with self.profile_operation("file_batch_generation"):
            start_time = time.perf_counter()
            total_size_bytes = 0
            
            for file_path in file_paths:
                try:
                    # Measure file I/O time
                    io_start = time.perf_counter()
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        total_size_bytes += file_size
                        self.metrics.files_processed += 1
                    io_end = time.perf_counter()
                    self.metrics.file_io_time += (io_end - io_start)
                    
                    # Measure generation time
                    gen_start = time.perf_counter()
                    generator.generate_tests(file_path, output_dir)
                    gen_end = time.perf_counter()
                    self.metrics.generation_time += (gen_end - gen_start)
                    
                    # Track memory usage
                    current_memory = self._get_memory_usage()
                    self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, current_memory)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {file_path}: {e}")
                    continue
            
            # Calculate averages
            if self.metrics.files_processed > 0:
                self.metrics.average_file_size_kb = (total_size_bytes / 1024) / self.metrics.files_processed
        
        return self.metrics
    
    def profile_single_file(self, file_path: Path, generator, output_dir: Path) -> ProfileMetrics:
        """Profile generator performance on a single large file."""
        self.logger.info(f"Profiling single file: {file_path}")
        
        with self.profile_operation("single_file_generation"):
            try:
                # Get file stats
                if file_path.exists():
                    file_size_kb = file_path.stat().st_size / 1024
                    self.metrics.average_file_size_kb = file_size_kb
                    self.metrics.files_processed = 1
                
                # Profile the generation
                generator.generate_tests(file_path, output_dir)
                
            except Exception as e:
                self.logger.error(f"Failed to profile {file_path}: {e}")
                raise
        
        return self.metrics
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a detailed profiling report."""
        report_lines = [
            "# TestGenerator Performance Profile Report",
            "",
            "## Summary Metrics",
            f"- **Total Processing Time**: {self.metrics.total_time:.2f}s",
            f"- **Files Processed**: {self.metrics.files_processed}",
            f"- **Functions Found**: {self.metrics.functions_found}",
            f"- **Tests Generated**: {self.metrics.tests_generated}",
            f"- **Peak Memory Usage**: {self.metrics.peak_memory_mb:.1f}MB",
            f"- **Memory Delta**: {self.metrics.memory_delta_mb:+.1f}MB",
            "",
            "## Performance Metrics",
            f"- **Files per Second**: {self.metrics.files_per_second:.1f}",
            f"- **Functions per Second**: {self.metrics.functions_per_second:.1f}",
            f"- **Throughput**: {self.metrics.mb_per_second:.1f} MB/s",
            f"- **Average File Size**: {self.metrics.average_file_size_kb:.1f}KB",
            "",
            "## Time Breakdown",
            f"- **Generation Time**: {self.metrics.generation_time:.2f}s ({self.metrics.generation_time/self.metrics.total_time*100:.1f}%)",
            f"- **File I/O Time**: {self.metrics.file_io_time:.2f}s ({self.metrics.file_io_time/self.metrics.total_time*100:.1f}%)",
            f"- **Parsing Time**: {self.metrics.parsing_time:.2f}s ({self.metrics.parsing_time/self.metrics.total_time*100:.1f}%)",
        ]
        
        # Add hot spots if available
        if self.metrics.hot_spots:
            report_lines.extend([
                "",
                "## Performance Hot Spots (Top 10)",
                "| Function | Time (s) | Calls | Avg Time/Call (ms) |",
                "|----------|----------|-------|-------------------|"
            ])
            
            for func_name, total_time, call_count in self.metrics.hot_spots[:10]:
                avg_time_ms = (total_time / call_count * 1000) if call_count > 0 else 0
                report_lines.append(f"| `{func_name}` | {total_time:.3f} | {call_count} | {avg_time_ms:.2f} |")
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Performance Recommendations",
            self._generate_recommendations()
        ])
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            output_path.write_text(report_content)
            self.logger.info(f"Profile report saved to {output_path}")
        
        return report_content
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB using resource module."""
        try:
            # Use resource.getrusage for memory information
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # On Linux, ru_maxrss is in KB, on macOS it's in bytes
            max_rss = usage.ru_maxrss
            if os.name == 'posix' and hasattr(os, 'uname') and os.uname().sysname == 'Darwin':
                # macOS reports in bytes
                return max_rss / 1024 / 1024
            else:
                # Linux reports in KB  
                return max_rss / 1024
        except (AttributeError, OSError):
            # Fallback: use a simple approximation based on object count
            gc.collect()  # Force garbage collection for more accurate count
            return len(gc.get_objects()) * 0.001  # Very rough approximation
    
    def _extract_hot_spots(self):
        """Extract performance hot spots from cProfile data."""
        if not self.profile_data:
            return
        
        # Capture profile stats
        s = io.StringIO()
        stats = pstats.Stats(self.profile_data, stream=s)
        stats.sort_stats('cumulative')
        
        # Extract top functions by cumulative time
        hot_spots = []
        for func_info, (call_count, _, total_time, cumulative_time, _) in stats.stats.items():
            file_name, line_num, func_name = func_info
            # Focus on our codebase functions
            if 'testgen_copilot' in file_name or func_name in ['generate_tests', '_generate_python_tests', '_parse_functions']:
                hot_spots.append((f"{func_name} ({file_name}:{line_num})", cumulative_time, call_count))
        
        # Sort by cumulative time and store top 20
        self.metrics.hot_spots = sorted(hot_spots, key=lambda x: x[1], reverse=True)[:20]
    
    def _calculate_performance_metrics(self):
        """Calculate derived performance metrics."""
        if self.metrics.total_time > 0:
            self.metrics.files_per_second = self.metrics.files_processed / self.metrics.total_time
            self.metrics.functions_per_second = self.metrics.functions_found / self.metrics.total_time
            
            # Calculate throughput in MB/s
            total_mb = (self.metrics.average_file_size_kb * self.metrics.files_processed) / 1024
            self.metrics.mb_per_second = total_mb / self.metrics.total_time
    
    def _generate_recommendations(self) -> str:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Memory-based recommendations
        if self.metrics.peak_memory_mb > 500:
            recommendations.append("- **High Memory Usage**: Consider implementing streaming for files >500MB peak usage")
        
        if self.metrics.memory_delta_mb > 100:
            recommendations.append("- **Memory Growth**: Investigate memory leaks or implement batch processing with cleanup")
        
        # Performance-based recommendations
        if self.metrics.files_per_second < 5:
            recommendations.append("- **Low Throughput**: Consider parallelization or caching for <5 files/second")
        
        if self.metrics.file_io_time / self.metrics.total_time > 0.3:
            recommendations.append("- **I/O Bottleneck**: File I/O takes >30% of time, consider async I/O or caching")
        
        # Hot spot recommendations
        if self.metrics.hot_spots:
            top_hot_spot = self.metrics.hot_spots[0]
            if top_hot_spot[1] / self.metrics.total_time > 0.4:
                recommendations.append(f"- **Hot Spot**: `{top_hot_spot[0]}` takes >40% of time, investigate optimization")
        
        if not recommendations:
            recommendations.append("- **Good Performance**: No major bottlenecks detected, current performance is acceptable")
        
        return "\n".join(recommendations)


def create_large_test_project(base_dir: Path, num_files: int = 100, lines_per_file: int = 50) -> List[Path]:
    """Create a synthetic large project for profiling tests."""
    logger = get_generator_logger()
    logger.info(f"Creating synthetic project with {num_files} files")
    
    project_dir = base_dir / "large_test_project"
    project_dir.mkdir(exist_ok=True)
    
    created_files = []
    
    for i in range(num_files):
        file_path = project_dir / f"module_{i:03d}.py"
        
        # Generate synthetic Python code
        content_lines = [
            f'"""Module {i} for performance testing."""',
            "",
            "import os",
            "import sys",
            "from typing import List, Dict, Optional",
            "",
        ]
        
        # Add multiple functions per file
        for j in range(max(1, lines_per_file // 10)):
            content_lines.extend([
                f"def function_{i}_{j}(param1: str, param2: int = 10) -> Dict[str, int]:",
                f'    """Function {j} in module {i}."""',
                f"    result = {{'key_{j}': param2 * {j + 1}}}",
                f"    if param1:",
                f"        result['param'] = len(param1)",
                f"    return result",
                "",
            ])
        
        # Add a class with methods
        content_lines.extend([
            f"class TestClass{i}:",
            f'    """Test class {i} for profiling."""',
            "",
            "    def __init__(self, value: int):",
            "        self.value = value",
            "",
            "    def process_data(self, data: List[str]) -> List[str]:",
            "        return [item.upper() for item in data if len(item) > self.value]",
            "",
        ])
        
        file_path.write_text("\n".join(content_lines))
        created_files.append(file_path)
    
    logger.info(f"Created {len(created_files)} test files in {project_dir}")
    return created_files