# System Configuration
NUM_HOSTS = 50
HOST_MIPS = 1000  # CPU capability
HOST_RAM = 16384  # 16 GB RAM

# Power Model Constants (Linear Model)
POWER_MAX = 250   # Watts at 100% Load
POWER_IDLE = 175  # Watts at 0% Load (if active)
# IDLE_RATIO = 0.7 (Implicit in 175/250)

# Simulation Settings
SIMULATION_STEPS = 200  # How long to run
TASKS_PER_STEP_MAX = 5  # Burstiness


# Cost Model
COST_PER_KWH = 0.10  # Cost in Dollars per kWh (or use 8.0 for Rupees)
SLA_PENALTY_COST = 1.50 # Penalty cost per dropped task (in Dollars)