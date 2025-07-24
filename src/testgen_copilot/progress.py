"""Progress reporting utilities for TestGen Copilot."""

from __future__ import annotations

import time
import sys
from typing import Optional, TextIO, Any, List
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

from .logging_config import get_generator_logger


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    start_time: float = 0.0
    current_item: Optional[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.completed_items + self.failed_items == 0:
            return 100.0
        return (self.completed_items / (self.completed_items + self.failed_items)) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        processed = self.completed_items + self.failed_items
        if processed == 0 or processed >= self.total_items:
            return None
        
        time_per_item = self.elapsed_time / processed
        remaining_items = self.total_items - processed
        return time_per_item * remaining_items
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 100.0
        processed = self.completed_items + self.failed_items
        return (processed / self.total_items) * 100


class ProgressReporter:
    """Progress reporter for long-running operations."""
    
    def __init__(self, 
                 output: TextIO = sys.stdout,
                 show_progress_bar: bool = True,
                 show_eta: bool = True,
                 show_rate: bool = True,
                 min_update_interval: float = 0.1):
        """Initialize progress reporter.
        
        Args:
            output: Output stream for progress messages
            show_progress_bar: Whether to show visual progress bar
            show_eta: Whether to show estimated time remaining
            show_rate: Whether to show processing rate
            min_update_interval: Minimum seconds between progress updates
        """
        self.output = output
        self.show_progress_bar = show_progress_bar
        self.show_eta = show_eta
        self.show_rate = show_rate
        self.min_update_interval = min_update_interval
        self.logger = get_generator_logger()
        self.stats = ProgressStats()
        self._last_update_time = 0.0
        self._last_output_length = 0
        
    def start(self, total_items: int, operation_name: str = "Processing"):
        """Start progress tracking."""
        self.stats.total_items = total_items
        self.stats.start_time = time.time()
        self._last_update_time = self.stats.start_time
        
        self.logger.info(f"Starting {operation_name}", {
            "total_items": total_items,
            "operation": operation_name
        })
        
        if self.output:
            self.output.write(f"\n🚀 {operation_name}: {total_items} items\n")
            self.output.flush()
        
        self._update_display()
    
    def update(self, current_item: Optional[str] = None, increment: int = 1, failed: bool = False):
        """Update progress with current item."""
        if failed:
            self.stats.failed_items += increment
            if current_item:
                self.stats.errors.append(current_item)
        else:
            self.stats.completed_items += increment
        
        self.stats.current_item = current_item
        
        # Only update display if enough time has passed
        current_time = time.time()
        if current_time - self._last_update_time >= self.min_update_interval:
            self._update_display()
            self._last_update_time = current_time
    
    def finish(self, operation_name: str = "Operation"):
        """Finish progress tracking and show summary."""
        self._update_display(force=True)
        
        # Final summary
        if self.output:
            self.output.write("\n\n")
            self.output.write(f"✅ {operation_name} completed!\n")
            self.output.write(f"   📊 Processed: {self.stats.completed_items + self.stats.failed_items}/{self.stats.total_items}\n")
            self.output.write(f"   ✅ Success: {self.stats.completed_items} ({self.stats.success_rate:.1f}%)\n")
            
            if self.stats.failed_items > 0:
                self.output.write(f"   ❌ Failed: {self.stats.failed_items}\n")
            
            self.output.write(f"   ⏱️  Total time: {self.stats.elapsed_time:.1f}s\n")
            
            if self.stats.completed_items > 0:
                rate = self.stats.completed_items / self.stats.elapsed_time
                self.output.write(f"   🚀 Rate: {rate:.1f} items/sec\n")
            
            self.output.flush()
        
        self.logger.info(f"{operation_name} completed", {
            "total_items": self.stats.total_items,
            "completed": self.stats.completed_items,
            "failed": self.stats.failed_items,
            "success_rate": self.stats.success_rate,
            "elapsed_time": self.stats.elapsed_time
        })
    
    def _update_display(self, force: bool = False):
        """Update the progress display."""
        if not self.output or (not force and not self._should_update()):
            return
        
        # Clear previous line
        if self._last_output_length > 0:
            self.output.write('\r' + ' ' * self._last_output_length + '\r')
        
        # Build progress line
        progress_parts = []
        
        # Progress bar
        if self.show_progress_bar:
            progress_parts.append(self._get_progress_bar())
        
        # Percentage
        progress_parts.append(f"{self.stats.completion_percentage:.1f}%")
        
        # Current counts
        processed = self.stats.completed_items + self.stats.failed_items
        progress_parts.append(f"({processed}/{self.stats.total_items})")
        
        # Current item (truncated)
        if self.stats.current_item:
            item_display = str(self.stats.current_item)
            if len(item_display) > 40:
                item_display = "..." + item_display[-37:]
            progress_parts.append(f"📄 {item_display}")
        
        # Processing rate
        if self.show_rate and self.stats.elapsed_time > 0:
            rate = processed / self.stats.elapsed_time
            progress_parts.append(f"({rate:.1f}/s)")
        
        # ETA
        if self.show_eta:
            eta = self.stats.estimated_remaining_time
            if eta is not None and eta > 1:
                eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
                progress_parts.append(f"ETA: {eta_str}")
        
        # Success/fail indicators
        if self.stats.failed_items > 0:
            progress_parts.append(f"❌{self.stats.failed_items}")
        
        progress_line = " | ".join(progress_parts)
        self._last_output_length = len(progress_line)
        
        self.output.write(progress_line)
        self.output.flush()
    
    def _get_progress_bar(self, width: int = 20) -> str:
        """Generate a visual progress bar."""
        filled = int(width * self.stats.completion_percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _should_update(self) -> bool:
        """Check if display should be updated."""
        return time.time() - self._last_update_time >= self.min_update_interval


@contextmanager
def progress_context(total_items: int, 
                    operation_name: str = "Processing",
                    show_progress: bool = True,
                    output: Optional[TextIO] = None):
    """Context manager for progress reporting."""
    if not show_progress:
        yield None
        return
    
    reporter = ProgressReporter(output=output or sys.stdout)
    reporter.start(total_items, operation_name)
    try:
        yield reporter
    finally:
        reporter.finish(operation_name)


class BatchProgressTracker:
    """Progress tracker specifically for batch file operations."""
    
    def __init__(self, files: List[Path], operation_name: str = "Batch Processing"):
        self.files = files
        self.operation_name = operation_name
        self.logger = get_generator_logger()
        
    def process_files(self, processor_func, show_progress: bool = True) -> dict:
        """Process files with progress tracking.
        
        Args:
            processor_func: Function that takes (file_path) and processes it
            show_progress: Whether to show progress bar
            
        Returns:
            dict with 'successful', 'failed', 'results' keys
        """
        results = {
            'successful': [],
            'failed': [],
            'results': []
        }
        
        with progress_context(len(self.files), self.operation_name, show_progress) as progress:
            for file_path in self.files:
                try:
                    if progress:
                        progress.update(current_item=str(file_path.name))
                    
                    result = processor_func(file_path)
                    results['successful'].append(file_path)
                    results['results'].append(result)
                    
                    self.logger.debug(f"Successfully processed {file_path}")
                    
                except Exception as e:
                    if progress:
                        progress.update(current_item=str(file_path.name), failed=True)
                    
                    results['failed'].append((file_path, str(e)))
                    self.logger.warning(f"Failed to process {file_path}: {e}")
        
        return results


def estimate_batch_time(file_count: int, average_time_per_file: float = 0.1) -> str:
    """Estimate total time for batch processing."""
    total_seconds = file_count * average_time_per_file
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        return f"{total_seconds/3600:.1f} hours"