import csv
import random

# This script creates a file 'google_trace.csv' inside the 'data' folder.
# It mimics the schema of the Google Cluster Trace (2011/2019).

def create_mock_trace():
    print("Generating Google-like workload trace...")
    
    header = ['timestamp', 'job_id', 'task_id', 'cpu_rate', 'mem_usage', 'duration']
    
    with open('data/google_trace.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Generate 2000 tasks (representing 24 hours of traffic)
        current_time = 0
        for i in range(2000):
            # 1. Traffic isn't constant. It comes in waves (Day/Night).
            # This sine-wave logic makes the graph look realistic (Peaks & Valleys).
            import math
            wave = (math.sin(i / 100) + 1) / 2 # value between 0 and 1
            
            if random.random() < wave: 
                # Peak time: Heavy tasks
                cpu = random.randint(500, 1500) # High CPU
                mem = random.randint(1024, 4096)
            else:
                # Off-peak: Light tasks
                cpu = random.randint(50, 300)   # Low CPU
                mem = random.randint(128, 512)

            duration = random.randint(5, 20)
            
            # Arrival time (tasks arrive every few seconds)
            current_time += random.randint(1, 5)
            
            writer.writerow([current_time, f"job_{i}", i, cpu, mem, duration])
            
    print("Success! 'data/google_trace.csv' created.")

if __name__ == "__main__":
    create_mock_trace()