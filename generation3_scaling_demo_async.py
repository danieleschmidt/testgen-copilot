"""Generation 3: Make it Scale - Async Demonstration

This script demonstrates the scaling optimization capabilities implemented
in Generation 3, working with the existing async hyper-scale optimization engine.
"""

import asyncio
import sys
import time
from typing import Any, Callable, Dict, List

# Add src to path
sys.path.insert(0, 'src')

from testgen_copilot.hyper_scale_optimization_engine import (
    HyperScaleOptimizationEngine, OptimizationLevel, CacheStrategy
)

async def demonstrate_generation3_scaling():
    """Demonstrate Generation 3 async scaling capabilities."""
    
    print("🚀 Generation 3: Make it Scale - ASYNC DEMONSTRATION")
    print("=" * 60)
    
    # 1. Initialize optimization engine
    print("\n1. Initializing Hyper-Scale Optimization Engine...")
    engine = HyperScaleOptimizationEngine(
        max_workers=8, 
        max_processes=4,
        cache_size_mb=512,
        quantum_processors=2
    )
    
    print(f"✅ Engine initialized:")
    print(f"   - Max workers: {engine.max_workers}")
    print(f"   - Max processes: {engine.max_processes}")
    print(f"   - Cache size: {engine.cache_size_mb}MB")
    print(f"   - Quantum processors: {engine.quantum_processors}")
    
    # 2. Initialize optimization systems
    print("\n2. Initializing Optimization Systems...")
    await engine.initialize_optimization_systems()
    
    # 3. Demonstrate async operations with optimization
    print("\n3. Testing Async Operations with Optimization...")
    
    async def compute_fibonacci(n: int) -> int:
        """Compute Fibonacci number - CPU intensive."""
        if n <= 1:
            return n
        # Simulate computation time
        await asyncio.sleep(0.001)
        return await compute_fibonacci(n - 1) + await compute_fibonacci(n - 2)
    
    async def fetch_data(dataset_size: int) -> List[int]:
        """Simulate data fetching - I/O intensive."""
        await asyncio.sleep(0.01)  # Simulate I/O delay
        return list(range(dataset_size))
    
    async def process_batch(batch: List[int]) -> List[int]:
        """Process batch of data - memory intensive."""
        await asyncio.sleep(0.005)
        return [x * 2 + 1 for x in batch]
    
    # Test CPU intensive operation with optimization
    print("\n   Testing CPU-intensive operation:")
    start_time = time.time()
    context_cpu = {"quantum_enabled": True, "distributed": False, "vectorizable": True}
    fib_result = await engine.optimize_performance(
        lambda: compute_fibonacci(10), 
        context_cpu
    )
    cpu_time = time.time() - start_time
    print(f"   ✅ Fibonacci(10) = {fib_result} in {cpu_time:.3f}s")
    
    # Test I/O intensive operation with optimization
    print("\n   Testing I/O-intensive operation:")
    start_time = time.time()
    context_io = {"global_access": True, "distributed": True}
    data_result = await engine.optimize_performance(
        lambda: fetch_data(1000), 
        context_io
    )
    io_time = time.time() - start_time
    print(f"   ✅ Fetched {len(data_result)} items in {io_time:.3f}s")
    
    # Test memory intensive operation with optimization
    print("\n   Testing memory-intensive batch processing:")
    start_time = time.time()
    context_batch = {"vectorizable": True, "quantum_enabled": False}
    batch_data = list(range(500))
    batch_result = await engine.optimize_performance(
        lambda: process_batch(batch_data), 
        context_batch
    )
    batch_time = time.time() - start_time
    print(f"   ✅ Processed {len(batch_result)} items in {batch_time:.3f}s")
    
    # 4. Demonstrate concurrent processing with multiple operations
    print("\n4. Testing Concurrent Processing...")
    
    async def concurrent_operations():
        """Run multiple operations concurrently."""
        tasks = []
        
        # Create multiple tasks with different optimization contexts
        for i in range(10):
            if i % 3 == 0:
                context = {"quantum_enabled": True, "distributed": False}
                task = engine.optimize_performance(lambda: compute_fibonacci(8), context)
            elif i % 3 == 1:
                context = {"global_access": True, "distributed": True}
                task = engine.optimize_performance(lambda: fetch_data(100), context)
            else:
                context = {"vectorizable": True}
                task = engine.optimize_performance(lambda: process_batch(list(range(50))), context)
            
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        return successful_results, concurrent_time
    
    concurrent_results, concurrent_time = await concurrent_operations()
    print(f"   ✅ Completed {len(concurrent_results)} concurrent operations in {concurrent_time:.3f}s")
    
    # 5. Demonstrate cache performance
    print("\n5. Testing Cache Performance...")
    
    async def cache_test_operation(data_id: int) -> Dict[str, Any]:
        """Operation that benefits from caching."""
        await asyncio.sleep(0.01)  # Simulate work
        return {"id": data_id, "result": data_id ** 2, "timestamp": time.time()}
    
    # First call (cache miss)
    start_time = time.time()
    context_cache = {"global_access": True}
    result1 = await engine.optimize_performance(
        lambda: cache_test_operation(42), 
        context_cache
    )
    cache_miss_time = time.time() - start_time
    
    # Second call (should hit cache)
    start_time = time.time()
    result2 = await engine.optimize_performance(
        lambda: cache_test_operation(42), 
        context_cache
    )
    cache_hit_time = time.time() - start_time
    
    print(f"   ✅ Cache performance:")
    print(f"      - Cache miss time: {cache_miss_time:.4f}s")
    print(f"      - Cache hit time: {cache_hit_time:.4f}s")
    if cache_miss_time > 0:
        print(f"      - Speedup: {cache_miss_time / cache_hit_time:.1f}x")
    
    # 6. Run short optimization demonstration
    print("\n6. Running Comprehensive Optimization Demonstration...")
    
    # Create a demonstration task that runs for 10 seconds
    demo_task = asyncio.create_task(engine._run_optimization_demonstration())
    
    print("   🚀 Running optimization workload for 10 seconds...")
    await asyncio.sleep(10.0)
    
    # Cancel the demonstration
    demo_task.cancel()
    try:
        await demo_task
    except asyncio.CancelledError:
        pass
    
    # 7. Generate optimization report
    print("\n7. Generating Optimization Report...")
    report = await engine.generate_optimization_report()
    
    # Extract key metrics from report
    cache_hit_rate = await engine._calculate_cache_hit_rate()
    optimization_score = await engine._calculate_optimization_score()
    
    print(f"   ✅ Optimization Summary:")
    print(f"      - Overall optimization score: {optimization_score:.2%}")
    print(f"      - Cache hit rate: {cache_hit_rate:.2%}")
    print(f"      - Current instances: {engine.current_instances}")
    print(f"      - CPU allocation: {engine.current_cpu_allocation:.1f}x")
    print(f"      - Memory allocation: {engine.current_memory_allocation:.1f}x")
    print(f"      - Active patterns: {sum(1 for p in engine.optimization_patterns.values() if p.usage_count > 0)}")
    
    # 8. Display cache layer statistics
    print("\n8. Cache Layer Performance:")
    for layer_name, cache in engine.cache_layers.items():
        total_requests = cache["hit_count"] + cache["miss_count"]
        hit_rate = cache["hit_count"] / max(total_requests, 1)
        print(f"   - {layer_name}: {hit_rate:.2%} hit rate ({len(cache['data'])}/{cache['max_size']} entries)")
    
    # 9. Display optimization patterns usage
    print("\n9. Optimization Patterns Usage:")
    active_patterns = [p for p in engine.optimization_patterns.values() if p.usage_count > 0]
    for pattern in sorted(active_patterns, key=lambda x: x.usage_count, reverse=True):
        print(f"   - {pattern.name}: {pattern.effectiveness_score:.2%} effectiveness ({pattern.usage_count} uses)")
    
    # 10. Scaling events summary
    if engine.scaling_events:
        print(f"\n10. Auto-Scaling Events:")
        for event in engine.scaling_events[-3:]:  # Show last 3 events
            print(f"   - {event.event_type}: {event.previous_capacity} → {event.new_capacity} ({event.trigger_reason})")
    else:
        print(f"\n10. Auto-Scaling: No scaling events occurred during demonstration")
    
    # Clean up resources
    if hasattr(engine, 'thread_pool'):
        engine.thread_pool.shutdown(wait=True)
    if hasattr(engine, 'process_pool'):
        engine.process_pool.shutdown(wait=True)
    
    print("\n🎯 Generation 3: Make it Scale - ASYNC DEMONSTRATION COMPLETE!")
    print("   🚀 Advanced async optimization: Real-time performance adaptation")
    print("   ⚡ Multi-tier caching: L1-CPU, L2-Memory, L3-Disk, Quantum, Global-Edge")
    print("   🌌 Quantum-enhanced processing: Superposition and coherent caching")
    print("   📊 ML-powered optimization: Predictive scaling and pattern learning")
    print("   🎛️  Auto-scaling: Dynamic resource allocation based on metrics")
    print("   🌐 Global distribution: Edge computing and multi-region support")


if __name__ == "__main__":
    asyncio.run(demonstrate_generation3_scaling())