"""
Profiling utilities for performance monitoring and optimization
"""

import time
from functools import wraps
import cProfile
import pstats
import datetime
import threading
import statistics
from collections import deque

# Global profiler
profiler = cProfile.Profile()

# Dictionary to store per-frame profiling stats
per_frame_stats = {}

# Dictionary to keep recent frame timings for moving average
recent_timings = {}

# Lock for thread safety when updating shared stats
stats_lock = threading.Lock()

# Flag to enable/disable terminal printing of per-frame stats
print_per_frame_stats = True

def profile_function(func):
    """
    Decorator for profiling functions - measures execution time per call
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Store per-frame stats
        func_name = func.__qualname__
        
        with stats_lock:
            if func_name not in per_frame_stats:
                per_frame_stats[func_name] = {
                    'last_time': execution_time,
                    'last_call_timestamp': time.time()
                }
                
                # Initialize deque for recent timings
                if func_name not in recent_timings:
                    recent_timings[func_name] = deque(maxlen=30)  # Keep last 30 frames
            else:
                per_frame_stats[func_name]['last_time'] = execution_time
                per_frame_stats[func_name]['last_call_timestamp'] = time.time()
            
            # Add to recent timings
            recent_timings[func_name].append(execution_time)
        
        # Print per-frame stats for model inference
        if print_per_frame_stats and (
            'YOLODetector.detect' in func_name or 
            'FeatureExtractor.extract_features_batch' in func_name
        ):
            avg_time = statistics.mean(recent_timings[func_name]) if recent_timings[func_name] else execution_time
            print(f"{func_name} - Frame time: {execution_time*1000:.1f}ms (Avg: {avg_time*1000:.1f}ms)")
        
        return result
    return wrapper

def set_print_per_frame_stats(enabled):
    """
    Enable or disable printing per-frame stats to terminal
    
    Args:
        enabled: Boolean flag to enable/disable printing
    """
    global print_per_frame_stats
    print_per_frame_stats = enabled
    print(f"Per-frame stats printing: {'ENABLED' if enabled else 'DISABLED'}")

def display_profiling_stats():
    """
    Displays profiling statistics focusing on per-frame performance
    """
    if not per_frame_stats:
        print("\nNo profiling statistics collected. Make sure functions are decorated with @profile_function.")
        return
        
    print("\n===== PERFORMANCE PROFILING RESULTS =====")
    print("{:<40} {:<20} {:<20}".format(
        "Function", "Last Frame (ms)", "Avg Last 30 Frames (ms)"))
    print("="*80)
    
    # Sort functions by last frame time (descending)
    sorted_stats = sorted(per_frame_stats.items(), key=lambda x: x[1]['last_time'], reverse=True)
    
    # Display results
    for func_name, stats in sorted_stats:
        last_time_ms = stats['last_time'] * 1000  # Convert to ms
        
        # Calculate average of recent timings
        avg_time_ms = 0
        if func_name in recent_timings and recent_timings[func_name]:
            avg_time_ms = statistics.mean(recent_timings[func_name]) * 1000
        
        print("{:<40} {:<20.2f} {:<20.2f}".format(
            func_name, 
            last_time_ms,
            avg_time_ms
        ))
    
    print("\n===== PERFORMANCE RECOMMENDATIONS =====")
    
    # Find potential bottlenecks
    bottlenecks = []
    for func_name, stats in sorted_stats:
        # Consider functions that take more than 20ms as potential bottlenecks
        if stats['last_time'] > 0.02:  # 20ms
            bottlenecks.append((func_name, stats['last_time']))
    
    if bottlenecks:
        print("Potential bottlenecks:")
        for func_name, exec_time in bottlenecks:
            print(f"- {func_name}: {exec_time*1000:.2f}ms")
            
            # Function-specific recommendations
            if "YOLODetector.detect" in func_name:
                print("  Recommendations:")
                print("  * Consider using a smaller/faster YOLO model (nano instead of small)")
                print("  * Reduce input resolution for detection")
                print("  * Implement frame skipping (detect every N frames)")
                
            elif "FeatureExtractor.extract_features_batch" in func_name:
                print("  Recommendations:")
                print("  * Reduce input resolution for the feature extractor")
                print("  * Extract features less frequently")
                print("  * Optimize batch processing to minimize GPU transfers")
                
            elif "HybridTracker.update" in func_name:
                print("  Recommendations:")
                print("  * Simplify feature matching logic")
                print("  * Reduce frequency of re-identification")
                
            elif "HybridTracker._perform_offline_reid" in func_name:
                print("  Recommendations:")
                print("  * Reduce re_id_interval (perform less frequently)")
                print("  * Reduce gallery size to compare fewer features")
    else:
        print("No significant performance bottlenecks detected.")
    
    print("\n===== END OF PROFILING REPORT =====\n")

def save_profiling_data(profiler):
    """
    Save profiling data to a file and display summary
    
    Args:
        profiler: The cProfile.Profile instance
    """
    # Stop profiling and display results
    profiler.disable()
    
    # Save detailed profiling stats to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = f"profiling_stats_{timestamp}.prof"
    profiler.dump_stats(stats_file)
    print(f"\nDetailed profiling stats saved to: {stats_file}")
    print("You can analyze this file with tools like snakeviz or using Python's pstats module")
    
    # Create pstats.Stats object from the file, not directly from the profiler
    stats = pstats.Stats(stats_file).sort_stats('cumtime')
    print("\n===== TOP 20 TIME-CONSUMING FUNCTIONS =====")
    stats.print_stats(20)
    
    # Display our custom performance metrics
    display_profiling_stats()