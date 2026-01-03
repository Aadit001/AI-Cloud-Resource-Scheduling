import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pandas as pd

# Import your existing backend logic
from src.entities import Host
from src.scheduler import Scheduler
from src.workload import WorkloadLoader
from src.ai_module import LSTMPredictor, DQNAgent
from src.config import *

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Cloud Scheduler", page_icon="☁️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Simulation Parameters")

# User Inputs
NUM_HOSTS_INPUT = st.sidebar.slider("Number of Servers", min_value=10, max_value=100, value=50, step=10)
SIM_STEPS = st.sidebar.slider("Simulation Duration (Steps)", min_value=500, max_value=5000, value=1000, step=500)
COST_PER_KWH = st.sidebar.number_input("Electricity Cost ($/kWh)", value=0.12, step=0.01)
SLA_PENALTY = st.sidebar.number_input("SLA Violation Penalty ($)", value=2.50, step=0.50)

st.sidebar.markdown("---")
st.sidebar.info("Adjust these values and click 'Run Simulation' to see how the AI adapts.")

# --- SIMULATION ENGINE (Adapted for Streamlit) ---
def run_simulation_logic(mode, progress_bar, status_text):
    # Dynamic Host Creation based on Slider
    hosts = [Host(i) for i in range(NUM_HOSTS_INPUT)]
    loader = WorkloadLoader("data/google_trace.csv")
    scheduler = Scheduler(hosts)
    
    # Initialize AI
    lstm = None
    dqn = None
    if mode == "AI-Driven":
        lstm = LSTMPredictor()
        dqn = DQNAgent(action_space_size=NUM_HOSTS_INPUT)
    
    # Metrics
    total_energy_watts = 0
    total_sla_drops = 0
    energy_log = []
    active_hosts_log = []
    load_history = [0.5] * 10 
    history_buffer = []

    # Run Simulation Loop
    for t in range(SIM_STEPS):
        # Update Progress Bar every 10%
        if t % (SIM_STEPS // 10) == 0:
            progress = int((t / SIM_STEPS) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Simulating {mode}... Step {t}/{SIM_STEPS}")

        # A. Prediction
        predicted_load = 0.5
        if mode == "AI-Driven":
            predicted_load = lstm.predict(load_history)
        
        # B. Workload
        new_tasks = loader.get_tasks_for_time(t)
        
        # C. Scheduling
        for task in new_tasks:
            success = False
            if mode == "Baseline (Round Robin)":
                target_host = hosts[t % NUM_HOSTS_INPUT] 
                if target_host.allocate(task): success = True
                else: success = scheduler.round_robin(task)
            elif mode == "AI-Driven":
                state = np.array([[task.mips/5000, task.ram/8000, predicted_load]])
                action_host_id = dqn.act(state)
                target_host = hosts[action_host_id]
                
                if target_host.allocate(task):
                    success = True
                    dqn.remember(state, action_host_id, 10, state, False)
                else:
                    success = scheduler.ai_optimized_best_fit(task)
                    dqn.remember(state, action_host_id, -10, state, False) # Penalty
                
                load_history.pop(0)
                load_history.append(task.mips / 2000.0)
                history_buffer.append(task.mips / 2000.0)

            if not success: total_sla_drops += 1

        # D. Train AI
        if mode == "AI-Driven" and t % 50 == 0:
            dqn.learn(batch_size=32)
            if len(history_buffer) > 20: lstm.train(history_buffer[-50:])

        # E. Stats
        active_count = 0
        step_power = 0
        for h in hosts:
            h.update_state()
            step_power += h.get_power()
            if not h.is_sleep: active_count += 1
            
        total_energy_watts += step_power
        energy_log.append(total_energy_watts)
        active_hosts_log.append(active_count)

    # Final Calculation
    total_kwh = total_energy_watts / (1000 * 3600)
    op_cost = (total_kwh * COST_PER_KWH) + (total_sla_drops * SLA_PENALTY)
    
    return energy_log, active_hosts_log, total_kwh, op_cost, total_sla_drops

# --- MAIN DASHBOARD LAYOUT ---
st.title("☁️ AI-Driven Cloud Resource Scheduling")
st.markdown("### M.tech Thesis Project Dashboard")
st.markdown("Compare **Standard Round Robin** vs. **Proposed AI (LSTM+DQN)** scheduling in real-time.")


# --- ADDED DETAILS HERE ---
st.markdown("##### **Name:** Aditya Tiwari  |  **Roll No:** 24A07RES18")
st.markdown("---")  # Adds a nice divider line

if st.button("🚀 RUN SIMULATION"):
    # Layout Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Running Baseline...")
        bar1 = st.progress(0)
        status1 = st.empty()
        rr_energy, rr_hosts, rr_kwh, rr_cost, rr_sla = run_simulation_logic("Baseline (Round Robin)", bar1, status1)
        status1.success("Baseline Complete!")

    with col2:
        st.subheader("2. Running AI-Driven Agent...")
        bar2 = st.progress(0)
        status2 = st.empty()
        ai_energy, ai_hosts, ai_kwh, ai_cost, ai_sla = run_simulation_logic("AI-Driven", bar2, status2)
        status2.success("AI Simulation Complete!")

    # --- RESULTS SECTION ---
    st.markdown("---")
    st.header("📊 Final Comparative Results")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    # Calculate Savings
    cost_saved = rr_cost - ai_cost
    energy_saved_pct = ((rr_kwh - ai_kwh) / rr_kwh) * 100
    
    m1.metric("Baseline Cost", f"${rr_cost:.2f}", f"Violations: {rr_sla}")
    m2.metric("AI-Driven Cost", f"${ai_cost:.2f}", f"Violations: {ai_sla}")
    m3.metric("Net Savings", f"${cost_saved:.2f}", delta_color="normal")
    m4.metric("Energy Efficiency", f"{energy_saved_pct:.1f}%", "Power Reduction")

    # --- PLOTTING ---
    tab1, tab2 = st.tabs(["📈 Energy & Cost Analysis", "🖥️ Server Consolidation"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rr_energy, 'r--', label='Baseline Power', alpha=0.6)
        ax.plot(ai_energy, 'g-', label='AI Power', linewidth=2)
        ax.fill_between(range(len(ai_energy)), ai_energy, rr_energy, color='lightgreen', alpha=0.3)
        ax.set_ylabel("Power (Watts)")
        ax.set_xlabel("Time Steps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.caption("The green area represents pure energy savings.")

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(rr_hosts, 'r--', label='Baseline Active Hosts')
        ax2.plot(ai_hosts, 'g-', label='AI Active Hosts', linewidth=2)
        ax2.set_ylabel("Count of Active Servers")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        st.caption("AI dynamically turns off servers (scale-in) when demand is low.")

else:
    st.info("👈 Use the sidebar to configure parameters, then click 'RUN SIMULATION'.")