"""Streaming analysis utilities for processing large projects efficiently."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .logging_config import get_generator_logger


@dataclass
class StreamingProgress:
    """Progress information for streaming operations."""

    total_items: int
    processed_items: int
    current_item: Optional[str] = None
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None
    errors: int = 0

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100.0

    @property
    def items_per_second(self) -> float:
        """Get processing rate in items per second."""
        if self.elapsed_time <= 0:
            return 0.0
        return self.processed_items / self.elapsed_time


@dataclass
class BatchResult:
    """Result of processing a batch of items."""

    batch_id: int
    items: List[Any]
    results: List[Any]
    errors: List[Exception]
    processing_time: float

    @property
    def success_count(self) -> int:
        """Number of successfully processed items."""
        return len(self.results) - len(self.errors)

    @property
    def error_count(self) -> int:
        """Number of failed items."""
        return len(self.errors)


class StreamingProcessor:
    """Process large collections of items in batches with progress reporting."""

    def __init__(
        self,
        batch_size: int = 50,
        max_workers: int = 1,
        progress_callback: Optional[Callable[[StreamingProgress], None]] = None
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.logger = get_generator_logger()

    def process_items(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any]
    ) -> Iterator[BatchResult]:
        """Process items in batches, yielding results as they complete.
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            
        Yields:
            BatchResult: Results for each completed batch
        """
        total_items = len(items)
        start_time = time.time()
        processed_count = 0
        error_count = 0

        self.logger.info("Starting streaming processing", {
            "total_items": total_items,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers
        })

        # Process items in batches
        for batch_id, batch_start in enumerate(range(0, total_items, self.batch_size)):
            batch_end = min(batch_start + self.batch_size, total_items)
            batch_items = items[batch_start:batch_end]

            # Update progress
            current_time = time.time()
            elapsed = current_time - start_time

            # Estimate remaining time
            if processed_count > 0:
                rate = processed_count / elapsed
                remaining_items = total_items - processed_count
                estimated_remaining = remaining_items / rate if rate > 0 else None
            else:
                estimated_remaining = None

            progress = StreamingProgress(
                total_items=total_items,
                processed_items=processed_count,
                current_item=str(batch_items[0]) if batch_items else None,
                elapsed_time=elapsed,
                estimated_remaining=estimated_remaining,
                errors=error_count
            )

            if self.progress_callback:
                self.progress_callback(progress)

            # Process the batch
            batch_start_time = time.time()
            batch_results = []
            batch_errors = []

            for item in batch_items:
                try:
                    result = processor_func(item)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.warning("Error processing item", {
                        "item": str(item),
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    batch_errors.append(e)
                    error_count += 1

            batch_time = time.time() - batch_start_time
            processed_count += len(batch_items)

            # Create batch result
            batch_result = BatchResult(
                batch_id=batch_id,
                items=batch_items,
                results=batch_results,
                errors=batch_errors,
                processing_time=batch_time
            )

            self.logger.debug("Batch completed", {
                "batch_id": batch_id,
                "batch_size": len(batch_items),
                "success_count": batch_result.success_count,
                "error_count": batch_result.error_count,
                "processing_time": batch_time
            })

            yield batch_result

        # Final progress update
        final_time = time.time()
        final_progress = StreamingProgress(
            total_items=total_items,
            processed_items=processed_count,
            elapsed_time=final_time - start_time,
            errors=error_count
        )

        if self.progress_callback:
            self.progress_callback(final_progress)

        self.logger.info("Streaming processing completed", {
            "total_items": total_items,
            "processed_items": processed_count,
            "total_errors": error_count,
            "total_time": final_time - start_time,
            "items_per_second": final_progress.items_per_second
        })


class FileStreamProcessor:
    """Specialized streaming processor for file analysis operations."""

    def __init__(
        self,
        batch_size: int = 25,
        progress_callback: Optional[Callable[[StreamingProgress], None]] = None
    ):
        self.processor = StreamingProcessor(
            batch_size=batch_size,
            progress_callback=progress_callback
        )
        self.logger = get_generator_logger()

    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        analyzer_func: Callable[[Path], Any]
    ) -> Iterator[BatchResult]:
        """Process files in batches for analysis.
        
        Args:
            file_paths: List of file paths to process
            analyzer_func: Function to analyze each file
            
        Yields:
            BatchResult: Results for each completed batch
        """
        # Convert to Path objects and filter existing files
        valid_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists() and path.is_file():
                valid_files.append(path)
            else:
                self.logger.warning("Skipping invalid file path", {
                    "file_path": str(file_path),
                    "exists": path.exists(),
                    "is_file": path.is_file() if path.exists() else False
                })

        self.logger.info("Starting file streaming analysis", {
            "total_files": len(file_paths),
            "valid_files": len(valid_files),
            "batch_size": self.processor.batch_size
        })

        # Process valid files
        yield from self.processor.process_items(valid_files, analyzer_func)

    def analyze_project_files(
        self,
        project_path: Union[str, Path],
        file_patterns: Optional[List[str]] = None,
        analyzer_func: Optional[Callable[[Path], Any]] = None
    ) -> Iterator[BatchResult]:
        """Analyze all matching files in a project directory.
        
        Args:
            project_path: Root directory of the project
            file_patterns: Glob patterns for files to analyze (default: ["**/*.py"])
            analyzer_func: Function to analyze each file (default: basic file info)
            
        Yields:
            BatchResult: Results for each completed batch
        """
        project_dir = Path(project_path)

        if not project_dir.exists() or not project_dir.is_dir():
            raise ValueError(f"Project path must be an existing directory: {project_path}")

        # Use default patterns if none provided
        if file_patterns is None:
            file_patterns = ["**/*.py"]
            
        # Collect matching files
        all_files = []
        for pattern in file_patterns:
            matching_files = list(project_dir.glob(pattern))
            all_files.extend(matching_files)

        # Remove duplicates and sort
        unique_files = sorted(set(all_files))

        self.logger.info("Collected project files for analysis", {
            "project_path": str(project_dir),
            "patterns": file_patterns,
            "total_files": len(unique_files)
        })

        # Use default analyzer if none provided
        if analyzer_func is None:
            def default_analyzer(file_path: Path) -> Dict[str, Any]:
                """Default file analyzer - returns basic file information."""
                stat = file_path.stat()
                return {
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "lines": len(file_path.read_text(encoding='utf-8', errors='ignore').splitlines())
                }
            analyzer_func = default_analyzer

        # Process files
        yield from self.process_files(unique_files, analyzer_func)


def create_progress_reporter(interval_seconds: float = 2.0) -> Callable[[StreamingProgress], None]:
    """Create a progress reporting callback that prints updates at regular intervals.
    
    Args:
        interval_seconds: Minimum time between progress reports
        
    Returns:
        Callback function for progress reporting
    """
    last_report_time = 0.0

    def progress_callback(progress: StreamingProgress) -> None:
        nonlocal last_report_time
        current_time = time.time()

        # Only report at specified intervals or when complete
        if (current_time - last_report_time >= interval_seconds or
            progress.processed_items >= progress.total_items):

            last_report_time = current_time

            # Format progress message
            percent = progress.progress_percent
            rate = progress.items_per_second

            msg_parts = [
                f"Progress: {progress.processed_items}/{progress.total_items}",
                f"({percent:.1f}%)",
                f"{rate:.1f} items/sec"
            ]

            if progress.estimated_remaining:
                eta_min = progress.estimated_remaining / 60
                if eta_min < 1:
                    msg_parts.append(f"ETA: {progress.estimated_remaining:.0f}s")
                else:
                    msg_parts.append(f"ETA: {eta_min:.1f}m")

            if progress.errors > 0:
                msg_parts.append(f"Errors: {progress.errors}")

            if progress.current_item:
                current_display = progress.current_item
                if len(current_display) > 50:
                    current_display = "..." + current_display[-47:]
                msg_parts.append(f"Current: {current_display}")

            print(" | ".join(msg_parts))

    return progress_callback
