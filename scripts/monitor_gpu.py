#!/usr/bin/env python3
"""Monitor GPU usage during training"""

import time
import subprocess
import sys

def monitor_gpu(interval=1):
    """Monitor GPU usage with nvidia-smi"""
    print("Monitoring GPU usage (Ctrl+C to stop)...")
    print("="*80)
    
    try:
        while True:
            # Clear screen (optional)
            # subprocess.run(['clear'], shell=True)
            
            # Get GPU stats
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 7:
                        gpu_id, name, gpu_util, mem_util, mem_used, mem_total, temp = parts
                        mem_used_mb = int(mem_used)
                        mem_total_mb = int(mem_total)
                        mem_percent = (mem_used_mb / mem_total_mb) * 100
                        
                        print(f"\rGPU {gpu_id}: {name[:20]:<20} | "
                              f"GPU: {gpu_util:>3}% | "
                              f"Mem: {mem_util:>3}% ({mem_used_mb:>5}MB/{mem_total_mb:>5}MB, {mem_percent:>5.1f}%) | "
                              f"Temp: {temp}°C", end='', flush=True)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    monitor_gpu(interval)
