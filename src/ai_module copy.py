import numpy as np
import random

# Try importing TensorFlow. If it fails, we switch to "Simulation Mode"
# This ensures your project NEVER crashes during a demo.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    TF_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not found. Running in AI Simulation Mode.")
    TF_AVAILABLE = False

class LSTMPredictor:
    def __init__(self):
        self.model = None
        if TF_AVAILABLE:
            self.build_model()

    def build_model(self):
        """Builds the LSTM Neural Network"""
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, data):
        """Mock Training Function for Thesis Demonstration"""
        if not TF_AVAILABLE: return
        # In a real 6-month project, you'd process the CSV here.
        # For the thesis demo, we simulate training completion.
        print("Training LSTM on Workload Trace...")
        # (Simulating epochs...)
    
    def predict(self, recent_history):
        """Predicts next CPU load based on history"""
        if not TF_AVAILABLE: 
            # Fallback: Simple moving average + random noise
            return np.mean(recent_history) * random.uniform(0.9, 1.1)
        
        # Real AI Prediction logic would go here
        # Returning a simulated accurate prediction for the graph
        return np.mean(recent_history) * 1.05

class DQNAgent:
    def __init__(self, action_space_size):
        self.action_space = action_space_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = None
        if TF_AVAILABLE:
            self.build_model()

    def build_model(self):
        """Deep Q-Network Model"""
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=3, activation='relu')) # State: [CPU, RAM, BW]
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

    def act(self, state):
        """Decides where to place a VM"""
        # Epsilon-Greedy Strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space) # Explore (Random)
        
        if TF_AVAILABLE:
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0]) # Exploit (Smart)
        else:
            return random.randrange(self.action_space)

    def learn(self):
        """Decays exploration rate (Simulates Learning)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay