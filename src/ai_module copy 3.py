import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from collections import deque
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("AI ENGINE: Scikit-Learn Neural Networks Loaded.")

class LSTMPredictor:
    """
    Uses a Multi-Layer Perceptron (Neural Network) to predict future load.
    Mathematically equivalent to an Autoregressive Neural Network.
    """
    def __init__(self):
        self.window_size = 10
        # MLPRegressor with 'lbfgs' is powerful for time-series regression
        self.model = MLPRegressor(hidden_layer_sizes=(50, 30), 
                                  activation='relu', 
                                  solver='adam', 
                                  max_iter=500,
                                  warm_start=True) # warm_start=True allows online learning
        self.is_trained = False

    def train(self, history_buffer):
        """Train the Neural Network on recent history"""
        if len(history_buffer) < self.window_size + 1:
            return

        # Prepare Training Data (Sliding Window)
        X, y = [], []
        for i in range(len(history_buffer) - self.window_size):
            X.append(history_buffer[i : i + self.window_size])
            y.append(history_buffer[i + self.window_size])
        
        if len(X) > 0:
            # Fit the model to the data
            self.model.fit(X, y)
            self.is_trained = True

    def predict(self, recent_history):
        """Predict next CPU load"""
        if not self.is_trained or len(recent_history) != self.window_size:
            return 0.5 # Default guess
        
        prediction = self.model.predict([recent_history])
        return max(0, prediction[0])

class DQNAgent:
    """
    Real Reinforcement Learning using Q-Learning with Neural Network approximation.
    """
    def __init__(self, action_space_size):
        self.action_space = action_space_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        
        # Neural Network for Q-Value Estimation
        # Input: 3 State variables. Output: Q-value for each of 50 hosts.
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), 
                                  activation='relu', 
                                  solver='adam', 
                                  warm_start=True)
        
        # Initialize model with dummy data so it's ready to predict
        dummy_input = np.zeros((1, 3))
        dummy_output = np.zeros((1, self.action_space))
        self.model.fit(dummy_input, dummy_output)

    def act(self, state):
        # Epsilon-Greedy Policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space) # Explore
        
        # Exploit (Ask Neural Network)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=32):
        """Train the Q-Network using Experience Replay"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([i[0][0] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        
        # Batch prediction
        targets = self.model.predict(states)
        next_qs = self.model.predict(next_states)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_qs[i])
            targets[i][action] = target
            
        # Train the Neural Network
        self.model.fit(states, targets)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay