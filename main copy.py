import matplotlib.pyplot as plt
from src.config import *
from src.entities import Host
from src.scheduler import Scheduler
from src.workload import WorkloadLoader

def run_scenario(algorithm_name, trace_file):
    print(f"--- Running Simulation: {algorithm_name} ---")
    
    # 1. Setup Environment
    hosts = [Host(i) for i in range(NUM_HOSTS)]
    scheduler = Scheduler(hosts)
    loader = WorkloadLoader(trace_file) # <--- NEW: Using CSV Loader
    
    total_energy = 0
    sla_violations = 0
    energy_log = []
    active_hosts_log = []

    # 2. Time Loop (Run for longer to see the whole trace)
    # We define 'SIMULATION_STEPS' roughly based on the trace file timestamp
    simulation_duration = 5000 
    
    for current_time in range(simulation_duration):
        
        # A. Get Real Tasks for this second
        new_tasks = loader.get_tasks_for_time(current_time)
        
        for task in new_tasks:
            # B. Schedule Task
            success = False
            if algorithm_name == "Round Robin":
                success = scheduler.round_robin(task)
            elif algorithm_name == "AI-Optimized":
                success = scheduler.ai_optimized_best_fit(task)
            
            if not success:
                sla_violations += 1

        # C. Update Host States
        step_power = 0
        active_count = 0
        for h in hosts:
            h.update_state()
            p = h.get_power()
            step_power += p
            if not h.is_sleep:
                active_count += 1
        
        total_energy += step_power
        energy_log.append(total_energy)
        active_hosts_log.append(active_count)
        
        # Stop if no more tasks
        if loader.current_index >= len(loader.all_tasks) and total_energy == 0:
            break

    print(f"Results ({algorithm_name}): Energy={int(total_energy/1000)} kW, SLA Drops={sla_violations}")
    return energy_log, active_hosts_log

if __name__ == "__main__":
    trace_file = "data/google_trace.csv"

    # 1. Run Simulations
    # We catch the *second* return value (active_hosts_log) now
    # Note: We need to update run_scenario to return TWO lists first.
    # See the instruction below this code block.
    print("Starting simulations...")
    rr_energy, rr_hosts = run_scenario("Round Robin", trace_file)
    ai_energy, ai_hosts = run_scenario("AI-Optimized", trace_file)

    # 2. Plot Results (Subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Graph 1: Energy Consumption
    ax1.plot(rr_energy, 'r--', label='Baseline (Round Robin)', alpha=0.7)
    ax1.plot(ai_energy, 'g-', linewidth=2, label='AI-Driven (Proposed)')
    ax1.set_ylabel('Energy (Watts)')
    ax1.set_title('Total Energy Consumption')
    ax1.legend()
    ax1.grid(True)

    # Graph 2: Active Servers (Consolidation)
    ax2.plot(rr_hosts, 'r--', label='Baseline (Round Robin)', alpha=0.7)
    ax2.plot(ai_hosts, 'g-', linewidth=2, label='AI-Driven (Proposed)')
    ax2.set_ylabel('Number of Active Hosts')
    ax2.set_xlabel('Simulation Time (Seconds)')
    ax2.set_title('Server Consolidation Analysis')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("data/thesis_results_combined.png")
    print("Success! Graphs saved to 'data/thesis_results_combined.png'")
    plt.show()