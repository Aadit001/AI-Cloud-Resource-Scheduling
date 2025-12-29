import csv
import random
import math
import numpy as np

def create_large_trace():
    print("Generating Large-Scale Google Trace (20,000 Tasks)...")
    
    header = ['timestamp', 'job_id', 'task_id', 'cpu_rate', 'mem_usage', 'duration']
    
    with open('data/google_trace.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        current_time = 0
        
        # Generate 20,000 tasks (Approx 3-4 days of traffic)
        for i in range(20000):
            # Create a Day/Night Cycle (Sine Wave)
            # Traffic peaks every 5000 tasks
            cycle = math.sin(i / 800) 
            
            # Base probability of a task arriving (Higher during "Day")
            arrival_prob = (cycle + 1.2) / 2.2 
            
            # Decide Task Size based on Cycle
            if random.random() < arrival_prob:
                # Peak Time: Heavy, High CPU tasks
                cpu = int(np.random.normal(1200, 300)) # Bell curve around 1200 MIPS
                mem = int(np.random.normal(2048, 500))
                duration = random.randint(10, 50)
                arrival_delay = random.randint(0, 2) # Fast arrival
            else:
                # Off-Peak: Small, light tasks
                cpu = int(np.random.normal(300, 100))
                mem = int(np.random.normal(512, 100))
                duration = random.randint(5, 20)
                arrival_delay = random.randint(2, 10) # Slow arrival

            # Clamp values to realistic limits
            cpu = max(100, min(cpu, 4000))
            mem = max(128, min(mem, 8192))
            
            current_time += arrival_delay
            writer.writerow([current_time, f"job_{i}", i, cpu, mem, duration])
            
            if i % 5000 == 0:
                print(f"Generated {i} tasks...")

    print("Success! 'data/google_trace.csv' generated with 20,000 tasks.")

if __name__ == "__main__":
    create_large_trace()