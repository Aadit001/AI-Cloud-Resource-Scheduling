# AI-Driven Resource Scheduling for Cloud Data Centers ☁️🔋

**M.Tech Thesis Project | IIT Patna**

### 📌 Overview
This project simulates a cloud data center to optimize **Energy Consumption** and **Operational Costs** using Artificial Intelligence. It replaces traditional "Round Robin" scheduling with a hybrid **LSTM + Deep Q-Network (DQN)** approach.

### 🚀 Key Features
* **Custom Simulation Environment:** A Python-based Discrete Event Simulator (DES) mimicking 50 heterogeneous servers.
* **Predictive Scaling:** Uses **LSTM** (Long Short-Term Memory) to forecast workload spikes.
* **Intelligent Scheduling:** Uses **Reinforcement Learning (DQN)** to consolidate tasks and minimize active servers.
* **Financial Analysis:** Real-time tracking of Electricity Bill ($) and SLA Penalty Costs.

### 📊 Results
* **Energy Savings:** ~35% reduction in power consumption compared to Round Robin.
* **Cost Reduction:** ~48% decrease in total operational expenses.
* **QoS:** Maintains SLA violation rates below 5%.

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **AI/ML:** Scikit-Learn (MLPRegressor), NumPy
* **Visualization:** Matplotlib
* **Data:** Google Cluster Trace (Synthetic Generation)

### 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Aadit001/AI-Cloud-Resource-Scheduling.git](https://github.com/Aadit001/AI-Cloud-Resource-Scheduling.git)
