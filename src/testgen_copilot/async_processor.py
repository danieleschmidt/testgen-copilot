"""Asynchronous processing engine for high-performance test generation."""

import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
import time

from .logging_config import get_generator_logger
from .error_recovery import async_retry_with_backoff

logger = get_generator_logger()


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    id: str
    file_path: Path
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result of processing a task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    worker_id: str = ""


class AsyncBatchProcessor:
    """High-performance async batch processor with concurrency control."""
    
    def __init__(self,
                 max_workers: int = 4,
                 batch_size: int = 10,
                 max_queue_size: int = 1000):
        """Initialize async batch processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Number of tasks to process in each batch
            max_queue_size: Maximum size of task queue
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.result_queue = asyncio.Queue()
        self.workers = []
        self.is_running = False
        
        # Performance metrics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
        
    async def start(self):
        """Start the async processing engine."""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = time.time()
        
        # Create worker coroutines
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        
        logger.info(f"Started async processor with {self.max_workers} workers")
        
    async def stop(self):
        """Stop the processing engine gracefully."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Signal workers to stop by putting None in queue
        for _ in self.workers:
            await self.task_queue.put(None)
            
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(
            f"Stopped async processor. "
            f"Processed {self.completed_tasks}/{self.total_tasks} tasks "
            f"in {elapsed:.2f}s"
        )
        
    async def submit_task(self, task: ProcessingTask) -> bool:
        """Submit a task for processing.
        
        Args:
            task: Task to process
            
        Returns:
            bool: True if task was queued successfully
        """
        try:
            self.task_queue.put_nowait(task)
            self.total_tasks += 1
            return True
        except asyncio.QueueFull:
            logger.warning(f"Task queue full, dropping task: {task.id}")
            return False
            
    async def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get a processing result.
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            ProcessingResult or None if timeout
        """
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
            
    async def process_batch(self,
                          tasks: List[ProcessingTask],
                          processor_func: Callable[[ProcessingTask], Awaitable[Any]]) -> List[ProcessingResult]:
        """Process a batch of tasks concurrently.
        
        Args:
            tasks: List of tasks to process
            processor_func: Async function to process each task
            
        Returns:
            List[ProcessingResult]: Results for all tasks
        """
        if not self.is_running:
            await self.start()
            
        # Submit all tasks
        submitted = []
        for task in tasks:
            if await self.submit_task(task):
                submitted.append(task)
                
        # Set processor function for workers
        self._processor_func = processor_func
        
        # Collect results
        results = []
        for _ in submitted:
            result = await self.get_result(timeout=30.0)
            if result:
                results.append(result)
                
        return results
        
    async def _worker(self, worker_id: str):
        """Worker coroutine that processes tasks from the queue."""
        logger.debug(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Check for stop signal
                if task is None:
                    break
                    
                # Process task with retry logic
                start_time = time.time()
                
                try:
                    result = await async_retry_with_backoff(
                        self._processor_func,
                        task,
                        max_attempts=3,
                        base_delay=0.5
                    )
                    
                    processing_result = ProcessingResult(
                        task_id=task.id,
                        success=True,
                        result=result,
                        processing_time=time.time() - start_time,
                        worker_id=worker_id
                    )
                    
                    self.completed_tasks += 1
                    
                except Exception as e:
                    logger.error(f"Task {task.id} failed in {worker_id}: {e}")
                    
                    processing_result = ProcessingResult(
                        task_id=task.id,
                        success=False,
                        error=e,
                        processing_time=time.time() - start_time,
                        worker_id=worker_id
                    )
                    
                    self.failed_tasks += 1
                    
                # Put result in result queue
                await self.result_queue.put(processing_result)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.debug(f"Worker {worker_id} stopped")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics.
        
        Returns:
            Dict with performance metrics
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.completed_tasks / max(self.total_tasks, 1),
            'tasks_per_second': self.completed_tasks / max(elapsed, 0.001),
            'queue_size': self.task_queue.qsize(),
            'elapsed_time': elapsed,
            'is_running': self.is_running
        }


class ConcurrentFileProcessor:
    """Concurrent file processor using ThreadPoolExecutor for CPU-bound tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize concurrent file processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or min(4, (len(os.environ.get('CPU_COUNT', '4')) or 4))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
    def process_files_concurrent(self,
                                file_paths: List[Path],
                                processor_func: Callable[[Path], Any]) -> List[Any]:
        """Process multiple files concurrently.
        
        Args:
            file_paths: List of file paths to process
            processor_func: Function to process each file
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(file_paths)} files with {self.max_workers} workers")
        
        start_time = time.time()
        
        # Submit all tasks to thread pool
        future_to_path = {
            self.executor.submit(processor_func, path): path
            for path in file_paths
        }
        
        results = []
        completed = 0
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:  # Log progress every 10 files
                    logger.info(f"Processed {completed}/{len(file_paths)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append(None)
                
        elapsed = time.time() - start_time
        logger.info(f"Completed processing {len(file_paths)} files in {elapsed:.2f}s")
        
        return results
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


class AdaptiveLoadBalancer:
    """Adaptive load balancer that adjusts worker allocation based on performance."""
    
    def __init__(self, initial_workers: int = 4):
        """Initialize adaptive load balancer.
        
        Args:
            initial_workers: Initial number of workers
        """
        self.current_workers = initial_workers
        self.min_workers = 1
        self.max_workers = 16
        
        # Performance tracking
        self.performance_history = []
        self.adjustment_cooldown = 10.0  # seconds
        self.last_adjustment = 0
        
    def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Determine if we should scale up workers.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            bool: True if should scale up
        """
        # Scale up if queue is backing up and success rate is good
        queue_size = metrics.get('queue_size', 0)
        success_rate = metrics.get('success_rate', 0)
        
        return (queue_size > self.current_workers * 2 and 
                success_rate > 0.8 and 
                self.current_workers < self.max_workers)
                
    def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Determine if we should scale down workers.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            bool: True if should scale down
        """
        # Scale down if queue is consistently empty
        queue_size = metrics.get('queue_size', 0)
        
        return (queue_size == 0 and 
                self.current_workers > self.min_workers)
                
    def adjust_workers(self, metrics: Dict[str, Any]) -> int:
        """Adjust number of workers based on performance.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            int: New worker count
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return self.current_workers
            
        # Determine scaling decision
        if self.should_scale_up(metrics):
            self.current_workers = min(self.current_workers + 1, self.max_workers)
            logger.info(f"Scaled up to {self.current_workers} workers")
            self.last_adjustment = current_time
            
        elif self.should_scale_down(metrics):
            self.current_workers = max(self.current_workers - 1, self.min_workers)
            logger.info(f"Scaled down to {self.current_workers} workers")
            self.last_adjustment = current_time
            
        return self.current_workers


async def process_files_async(file_paths: List[Path],
                            processor_func: Callable[[Path], Awaitable[Any]],
                            max_workers: int = 4) -> List[Any]:
    """High-level async file processing function.
    
    Args:
        file_paths: List of file paths to process
        processor_func: Async function to process each file
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of processing results
    """
    # Create tasks
    tasks = [
        ProcessingTask(id=f"file-{i}", file_path=path)
        for i, path in enumerate(file_paths)
    ]
    
    # Process with async batch processor
    async with AsyncBatchProcessor(max_workers=max_workers) as processor:
        results = await processor.process_batch(tasks, processor_func)
        
    # Extract successful results
    return [r.result for r in results if r.success]