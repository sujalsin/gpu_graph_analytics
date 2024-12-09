import cupy as cp
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class KernelStats:
    """Statistics for a single CUDA kernel execution."""
    name: str
    total_time: float
    avg_time: float
    num_calls: int
    max_memory: int
    
class PerformanceProfiler:
    """
    GPU performance profiling tool for graph algorithms.
    Tracks kernel execution times, memory usage, and throughput.
    """
    
    def __init__(self):
        self.kernel_stats: Dict[str, KernelStats] = {}
        self.memory_usage: List[Tuple[float, int]] = []  # (timestamp, bytes)
        self.current_kernel: Optional[str] = None
        self.start_time: Optional[float] = None
        
    @contextmanager
    def profile_kernel(self, name: str):
        """Context manager for profiling a CUDA kernel execution."""
        try:
            self.current_kernel = name
            start = time.perf_counter()
            initial_memory = cp.get_default_memory_pool().used_bytes()
            
            yield
            
            # Synchronize GPU
            cp.cuda.Stream.null.synchronize()
            
            # Calculate statistics
            end = time.perf_counter()
            final_memory = cp.get_default_memory_pool().used_bytes()
            duration = end - start
            
            # Update kernel statistics
            if name not in self.kernel_stats:
                self.kernel_stats[name] = KernelStats(
                    name=name,
                    total_time=duration,
                    avg_time=duration,
                    num_calls=1,
                    max_memory=max(0, final_memory - initial_memory)
                )
            else:
                stats = self.kernel_stats[name]
                stats.total_time += duration
                stats.num_calls += 1
                stats.avg_time = stats.total_time / stats.num_calls
                stats.max_memory = max(
                    stats.max_memory,
                    final_memory - initial_memory
                )
                
            # Record memory usage
            self.memory_usage.append((time.time(), final_memory))
            
        finally:
            self.current_kernel = None
            
    def start_session(self):
        """Start a new profiling session."""
        self.kernel_stats.clear()
        self.memory_usage.clear()
        self.start_time = time.time()
        
    def end_session(self) -> Dict[str, Any]:
        """
        End the current profiling session and return statistics.
        
        Returns:
            Dictionary containing profiling statistics
        """
        if self.start_time is None:
            raise RuntimeError("No active profiling session")
            
        total_time = time.time() - self.start_time
        peak_memory = max(usage for _, usage in self.memory_usage)
        
        # Calculate statistics
        stats = {
            "total_time": total_time,
            "peak_memory": peak_memory,
            "kernel_stats": {
                name: {
                    "total_time": stat.total_time,
                    "avg_time": stat.avg_time,
                    "num_calls": stat.num_calls,
                    "max_memory": stat.max_memory
                }
                for name, stat in self.kernel_stats.items()
            },
            "memory_timeline": self.memory_usage
        }
        
        self.start_time = None
        return stats
    
    def print_summary(self):
        """Print a summary of the profiling results."""
        if not self.kernel_stats:
            print("No profiling data available")
            return
            
        print("\nProfiling Summary:")
        print("-" * 80)
        print(f"{'Kernel Name':<30} {'Calls':<8} {'Avg Time (ms)':<15} {'Total Time (ms)':<15} {'Max Memory (MB)':<15}")
        print("-" * 80)
        
        for name, stats in self.kernel_stats.items():
            print(f"{name:<30} {stats.num_calls:<8} {stats.avg_time*1000:>13.3f} {stats.total_time*1000:>14.3f} {stats.max_memory/1024/1024:>14.2f}")
            
        print("-" * 80)
        peak_memory = max(usage for _, usage in self.memory_usage)
        print(f"Peak Memory Usage: {peak_memory/1024/1024:.2f} MB")
        
    def export_chrome_trace(self, filename: str):
        """
        Export profiling data in Chrome tracing format.
        
        Args:
            filename: Output JSON file path
        """
        import json
        
        # Convert profiling data to Chrome tracing format
        events = []
        pid = 1  # Process ID
        
        for name, stats in self.kernel_stats.items():
            tid = hash(name) % 10000  # Thread ID
            
            for i in range(stats.num_calls):
                start_time = self.memory_usage[i][0] - self.start_time
                duration = stats.avg_time
                
                events.append({
                    "name": name,
                    "cat": "kernel",
                    "ph": "X",  # Complete event
                    "pid": pid,
                    "tid": tid,
                    "ts": start_time * 1e6,  # Microseconds
                    "dur": duration * 1e6,
                    "args": {
                        "memory": stats.max_memory
                    }
                })
                
        # Add memory usage counter
        for timestamp, memory in self.memory_usage:
            events.append({
                "name": "memory",
                "cat": "counter",
                "ph": "C",  # Counter event
                "pid": pid,
                "ts": (timestamp - self.start_time) * 1e6,
                "args": {
                    "memory": memory
                }
            })
            
        # Write to file
        with open(filename, 'w') as f:
            json.dump({
                "traceEvents": events,
                "displayTimeUnit": "ms"
            }, f)
