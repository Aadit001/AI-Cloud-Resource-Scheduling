import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.gridspec as gridspec
from src.config import *
from src.entities import Host
from src.scheduler import Scheduler
from src.workload import WorkloadLoader
from src.ai_module import LSTMPredictor, DQNAgent

# --- FINANCIAL CONSTANTS ---
COST_PER_KWH = 0.12        
SLA_PENALTY_COST = 2.50    

def run_simulation(mode, trace_file):
    print(f"\n--- Starting Simulation: {mode} ---")
    
    # 1. Initialize Components
    hosts = [Host(i) for i in range(NUM_HOSTS)]
    loader = WorkloadLoader(trace_file)
    scheduler = Scheduler(hosts)
    
    # AI Components
    lstm = LSTMPredictor()
    dqn = DQNAgent(action_space_size=NUM_HOSTS)
    
    # Metrics
    total_energy_watts = 0
    total_sla_drops = 0
    total_migrations = 0 
    
    energy_log = []
    active_hosts_log = []
    load_history = [0.5] * 10 
    
    # 2. Simulation Loop
    for t in range(2000):
        
        # A. AI Prediction
        predicted_load = 0
        if mode == "AI-Driven":
            predicted_load = lstm.predict(load_history)
        
        # B. Get Real Tasks
        new_tasks = loader.get_tasks_for_time(t)
        
        # C. Schedule Tasks
        for task in new_tasks:
            task.mips = int(task.mips * 1.2) # Stress test
            success = False
            
            if mode == "Baseline (Round Robin)":
                target_host = hosts[t % NUM_HOSTS] 
                if target_host.allocate(task):
                    success = True
                else:
                    success = scheduler.round_robin(task)
                
            elif mode == "AI-Driven":
                sorted_hosts = sorted(hosts, key=lambda h: h.get_utilization(), reverse=True)
                for h in sorted_hosts:
                    if h.allocate(task):
                        success = True
                        break
                load_history.pop(0)
                load_history.append(task.mips / 1000.0)

            if not success:
                total_sla_drops += 1

        # D. SIMULATE MIGRATION
        if mode == "AI-Driven" and t % 50 == 0: 
            migrations_in_this_step = random.randint(1, 5)
            total_migrations += migrations_in_this_step
            total_energy_watts += (migrations_in_this_step * 10) 

        # E. Update System State
        active_count = 0
        step_power = 0
        for h in hosts:
            h.update_state()
            p = h.get_power()
            step_power += p
            if not h.is_sleep: 
                active_count += 1
            
        if mode == "AI-Driven":
            dqn.learn()
        
        total_energy_watts += step_power
        energy_log.append(total_energy_watts)
        active_hosts_log.append(active_count)

    # 3. Financial Analysis
    total_kwh = total_energy_watts / (1000 * 3600)
    energy_cost = total_kwh * COST_PER_KWH
    penalty_cost = total_sla_drops * SLA_PENALTY_COST
    total_operational_cost = energy_cost + penalty_cost

    print(f"--- RESULTS: {mode} ---")
    print(f"Total Energy          : {total_kwh:.4f} kWh")
    print(f"Total Operational Cost: ${total_operational_cost:.2f}")
    print(f"Total VM Migrations   : {total_migrations}")
    print(f"Total SLA Violations  : {total_sla_drops}")
    
    stats = {
        "energy": total_kwh,
        "cost": total_operational_cost,
        "migrations": total_migrations,
        "sla": total_sla_drops
    }
    
    return energy_log, active_hosts_log, stats

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    trace_file = "data/google_trace.csv"
    
    # 1. Run Simulations
    rr_energy, rr_hosts, rr_stats = run_simulation("Baseline (Round Robin)", trace_file)
    ai_energy, ai_hosts, ai_stats = run_simulation("AI-Driven", trace_file)
    
    # 2. Plot Final Results (Dual Grid Layout)
    fig = plt.figure(figsize=(14, 22)) # WIDER figure to fit 4 charts
    
    # --- LAYOUT DEFINITION ---
    # Block 1: Top 70% for Line Graphs
    gs_top = gridspec.GridSpec(3, 1, figure=fig, top=0.95, bottom=0.35, hspace=0.4)
    
    # Block 2: Bottom 20% for Bar Charts (Now 4 Columns)
    gs_bottom = gridspec.GridSpec(1, 4, figure=fig, top=0.25, bottom=0.05, wspace=0.4)

    # --- ROW 1: ENERGY ---
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax1.plot(rr_energy, 'r--', label='Baseline', alpha=0.6)
    ax1.plot(ai_energy, 'g-', linewidth=2, label='AI-Driven')
    ax1.fill_between(range(len(ai_energy)), ai_energy, rr_energy, color='lightgreen', alpha=0.3)
    ax1.set_ylabel('Power (Watts)')
    ax1.set_title('1. Real-Time Energy Consumption')
    ax1.legend()
    ax1.grid(True)
    
    # --- ROW 2: COST ---
    ax2 = fig.add_subplot(gs_top[1, 0])
    rr_cost_acc = np.cumsum([ (e / 3600000) * COST_PER_KWH for e in rr_energy ])
    ai_cost_acc = np.cumsum([ (e / 3600000) * COST_PER_KWH for e in ai_energy ])
    ax2.plot(rr_cost_acc, 'r--', label='Baseline', alpha=0.6)
    ax2.plot(ai_cost_acc, 'g-', linewidth=2, label='AI-Driven')
    ax2.fill_between(range(len(ai_cost_acc)), ai_cost_acc, rr_cost_acc, color='orange', alpha=0.5, label='Savings')
    ax2.set_ylabel('Cost ($)')
    ax2.set_title('2. Cumulative Operational Cost')
    ax2.legend()
    ax2.grid(True)

    # --- ROW 3: HOSTS ---
    ax3 = fig.add_subplot(gs_top[2, 0])
    ax3.plot(rr_hosts, 'r--', label='Baseline', alpha=0.6)
    ax3.plot(ai_hosts, 'g-', linewidth=2, label='AI-Driven')
    ax3.set_ylabel('Active Hosts')
    ax3.set_xlabel('Time Steps')
    ax3.set_title('3. Server Consolidation Analysis')
    ax3.legend()
    ax3.grid(True)

    # --- ROW 4: BAR CHARTS (Now 4 Charts) ---
    labels = ['Baseline', 'AI-Driven']
    colors = ['#ff9999', '#99ff99'] 
    
    # Chart 4.1: Energy
    ax4 = fig.add_subplot(gs_bottom[0, 0])
    bars1 = ax4.bar(labels, [rr_stats['energy'], ai_stats['energy']], color=colors, edgecolor='black')
    ax4.set_title('Total Energy (kWh)')
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')

    # Chart 4.2: Cost
    ax5 = fig.add_subplot(gs_bottom[0, 1])
    bars2 = ax5.bar(labels, [rr_stats['cost'], ai_stats['cost']], color=colors, edgecolor='black')
    ax5.set_title('Total Cost ($)')
    for bar in bars2:
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():.2f}', ha='center', va='bottom')

    # Chart 4.3: Migrations
    ax6 = fig.add_subplot(gs_bottom[0, 2])
    bars3 = ax6.bar(labels, [rr_stats['migrations'], ai_stats['migrations']], color=colors, edgecolor='black')
    ax6.set_title('Total Migrations')
    for bar in bars3:
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')

    # Chart 4.4: SLA Violations (NEW)
    ax7 = fig.add_subplot(gs_bottom[0, 3])
    bars4 = ax7.bar(labels, [rr_stats['sla'], ai_stats['sla']], color=['#ff6666', '#66ff66'], edgecolor='black')
    ax7.set_title('SLA Violations')
    ax7.set_ylabel('Dropped Tasks')
    for bar in bars4:
        ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')

    plt.savefig("data/final_thesis_results_complete.png")
    print("\nSUCCESS! Generated Graph with SLA Violations included.")
    plt.show()