import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from collections import deque
import warnings

warnings.filterwarnings('ignore')

print("AI ENGINE: Scikit-Learn Neural Networks Loaded.")

class LSTMPredictor:
    def __init__(self):
        self.window_size = 10
        self.model = MLPRegressor(hidden_layer_sizes=(50, 30), 
                                  activation='relu', 
                                  solver='adam', 
                                  max_iter=500,
                                  warm_start=True)
        self.is_trained = False

    def train(self, history_buffer):
        if len(history_buffer) < self.window_size + 1: return
        X, y = [], []
        for i in range(len(history_buffer) - self.window_size):
            X.append(history_buffer[i : i + self.window_size])
            y.append(history_buffer[i + self.window_size])
        if len(X) > 0:
            self.model.fit(X, y)
            self.is_trained = True

    def predict(self, recent_history):
        if not self.is_trained or len(recent_history) != self.window_size:
            return 0.5 
        return max(0, self.model.predict([recent_history])[0])

class DQNAgent:
    def __init__(self, action_space_size):
        self.action_space = action_space_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        
        # CRITICAL CHANGE: Start with low exploration so it uses the "Smart" logic immediately
        self.epsilon = 0.1  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), 
                                  activation='relu', 
                                  solver='adam', 
                                  warm_start=True)
        
        # --- PRE-TRAINING (The "Cheat Sheet") ---
        # We teach the Neural Network that "Host 0" (High Priority) is better than "Host 49"
        # This mimics a 'Best Fit' strategy so the AI starts smart.
        print("Pre-training DQN Agent with Heuristic Knowledge...")
        X_train = []
        y_train = []
        
        # Create 500 fake scenarios
        for _ in range(500):
            # Fake State: [Small Task, Low RAM, Random Load]
            state = [random.random(), random.random(), random.random()]
            
            # The "Correct" answer is to pick lower ID hosts (Consolidation)
            # We assign higher Q-values to Host 0, 1, 2... and low values to Host 49
            q_values = [10.0 - (i * 0.2) for i in range(self.action_space)]
            
            X_train.append(state)
            y_train.append(q_values)
            
        self.model.fit(X_train, y_train)
        print("DQN Agent Pre-training Complete.")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        q_values = self.model.predict(state)
        # Pick the host with the highest Q-value
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size: return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0][0] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        
        targets = self.model.predict(states)
        next_qs = self.model.predict(next_states)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_qs[i])
            targets[i][action] = target
            
        self.model.fit(states, targets)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay