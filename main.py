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
    print(f"\n--- Starting Real-World Simulation: {mode} ---")
    
    hosts = [Host(i) for i in range(NUM_HOSTS)]
    loader = WorkloadLoader(trace_file)
    scheduler = Scheduler(hosts)
    
    # Initialize Real AI
    lstm = None
    dqn = None
    if mode == "AI-Driven":
        lstm = LSTMPredictor()
        dqn = DQNAgent(action_space_size=NUM_HOSTS)
    
    total_energy_watts = 0
    total_sla_drops = 0
    total_migrations = 0 
    
    energy_log = []
    active_hosts_log = []
    
    # LSTM Window (normalized 0-1)
    load_history = [0.5] * 10 
    history_buffer = [] # Store data to train LSTM
    
    # Run for 2000 steps (Processing ~5000 tasks)
    # (Running full 20,000 might take 1 hour, so we sample 2000 steps for the graph)
    SIMULATION_STEPS = 2000
    
    for t in range(SIMULATION_STEPS):
        if t % 100 == 0: print(f"  Step {t}/{SIMULATION_STEPS}...")
        
        # A. AI Prediction (Real Inference)
        predicted_load = 0.5
        if mode == "AI-Driven":
            predicted_load = lstm.predict(load_history)
        
        # B. Get Real Tasks
        new_tasks = loader.get_tasks_for_time(t)
        
        # C. Schedule Tasks
        for task in new_tasks:
            success = False
            
            if mode == "Baseline (Round Robin)":
                target_host = hosts[t % NUM_HOSTS] 
                if target_host.allocate(task):
                    success = True
                else:
                    success = scheduler.round_robin(task)
                
            elif mode == "AI-Driven":
                # Real DQN State: [Task MIPS, Task RAM, Predicted Load]
                # Normalize values for Neural Network (0 to 1 range approx)
                state = np.array([[task.mips/5000, task.ram/8000, predicted_load]])
                
                # 1. DQN decides Host ID
                action_host_id = dqn.act(state)
                
                # 2. Execute Action
                target_host = hosts[action_host_id]
                reward = 0
                
                if target_host.allocate(task):
                    success = True
                    reward = 10 # Good job!
                else:
                    # AI Failed -> Penalty
                    reward = -10
                    # Fallback to heuristic so we don't crash
                    success = scheduler.ai_optimized_best_fit(task)
                
                # 3. Store experience in Memory
                next_state = state # Simplification for single-step
                dqn.remember(state, action_host_id, reward, next_state, False)
                
                # Update LSTM History
                current_load_metric = task.mips / 2000.0
                load_history.pop(0)
                load_history.append(current_load_metric)
                history_buffer.append(current_load_metric)

            if not success:
                total_sla_drops += 1

        # D. Train AI (Batch Training every 50 steps to save time)
        if mode == "AI-Driven" and t % 50 == 0:
            # Train DQN
            dqn.learn(batch_size=32)
            # Train LSTM
            if len(history_buffer) > 20:
                lstm.train(history_buffer[-50:]) # Train on recent data
        
        # E. Update System
        active_count = 0
        step_power = 0
        for h in hosts:
            h.update_state()
            p = h.get_power()
            step_power += p
            if not h.is_sleep: active_count += 1
            
        total_energy_watts += step_power
        energy_log.append(total_energy_watts)
        active_hosts_log.append(active_count)

    # Financials
    total_kwh = total_energy_watts / (1000 * 3600)
    energy_cost = total_kwh * COST_PER_KWH
    penalty_cost = total_sla_drops * SLA_PENALTY_COST
    total_operational_cost = energy_cost + penalty_cost

    stats = {
        "energy": total_kwh,
        "cost": total_operational_cost,
        "migrations": total_migrations,
        "sla": total_sla_drops
    }
    return energy_log, active_hosts_log, stats

if __name__ == "__main__":
    trace_file = "data/google_trace.csv"
    
    rr_energy, rr_hosts, rr_stats = run_simulation("Baseline (Round Robin)", trace_file)
    ai_energy, ai_hosts, ai_stats = run_simulation("AI-Driven", trace_file)
    
    # PLOTTING (Standard 4-Panel Dashboard)
    fig = plt.figure(figsize=(14, 22))
    gs_top = gridspec.GridSpec(3, 1, figure=fig, top=0.95, bottom=0.35, hspace=0.4)
    gs_bottom = gridspec.GridSpec(1, 4, figure=fig, top=0.25, bottom=0.05, wspace=0.4)

    # 1. Energy
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax1.plot(rr_energy, 'r--', label='Baseline', alpha=0.6)
    ax1.plot(ai_energy, 'g-', linewidth=2, label='AI-Driven')
    ax1.fill_between(range(len(ai_energy)), ai_energy, rr_energy, color='lightgreen', alpha=0.3)
    ax1.set_ylabel('Power (Watts)')
    ax1.set_title('1. Real-Time Energy Consumption (Real AI)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Cost
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

    # 3. Hosts
    ax3 = fig.add_subplot(gs_top[2, 0])
    ax3.plot(rr_hosts, 'r--', label='Baseline', alpha=0.6)
    ax3.plot(ai_hosts, 'g-', linewidth=2, label='AI-Driven')
    ax3.set_ylabel('Active Hosts')
    ax3.set_title('3. Server Consolidation')
    ax3.grid(True)

    # 4. Bar Charts
    labels = ['Baseline', 'AI-Driven']
    colors = ['#ff9999', '#99ff99']
    
    # Energy
    ax4 = fig.add_subplot(gs_bottom[0, 0])
    bars1 = ax4.bar(labels, [rr_stats['energy'], ai_stats['energy']], color=colors, edgecolor='black')
    ax4.set_title('Total Energy (kWh)')
    for bar in bars1: ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')

    # Cost
    ax5 = fig.add_subplot(gs_bottom[0, 1])
    bars2 = ax5.bar(labels, [rr_stats['cost'], ai_stats['cost']], color=colors, edgecolor='black')
    ax5.set_title('Total Cost ($)')
    for bar in bars2: ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():.2f}', ha='center', va='bottom')

    # Migrations
    ax6 = fig.add_subplot(gs_bottom[0, 2])
    bars3 = ax6.bar(labels, [rr_stats['migrations'], ai_stats['migrations']], color=colors, edgecolor='black')
    ax6.set_title('Total Migrations')
    for bar in bars3: ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')

    # SLA
    ax7 = fig.add_subplot(gs_bottom[0, 3])
    bars4 = ax7.bar(labels, [rr_stats['sla'], ai_stats['sla']], color=['#ff6666', '#66ff66'], edgecolor='black')
    ax7.set_title('SLA Violations')
    for bar in bars4: ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')

    plt.savefig("data/final_thesis_results_real_ai.png")
    print("\nSUCCESS! Real AI Simulation Complete. Graphs saved.")
    plt.show()