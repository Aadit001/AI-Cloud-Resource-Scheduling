import numpy as np
import random
import os

# Suppress TensorFlow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam
from collections import deque

print("AI ENGINE: TensorFlow Loaded Successfully.")

class LSTMPredictor:
    def __init__(self):
        self.window_size = 10
        self.model = self._build_model()
        
    def _build_model(self):
        """Real LSTM Neural Network"""
        model = Sequential()
        # Input shape: (Time_Steps, Features) -> (10, 1)
        model.add(Input(shape=(self.window_size, 1)))
        model.add(LSTM(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1)) # Predicts next load
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train(self, history_data):
        """Train on a batch of history data"""
        if len(history_data) < self.window_size + 1:
            return
            
        X, y = [], []
        for i in range(len(history_data) - self.window_size):
            X.append(history_data[i : i + self.window_size])
            y.append(history_data[i + self.window_size])
            
        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y)
        
        # Quick training on one batch (Online Learning)
        self.model.fit(X, y, epochs=1, verbose=0)

    def predict(self, recent_history):
        """Predict next value based on last 10 steps"""
        if len(recent_history) != self.window_size:
            return 0.5 # Default if not enough data
            
        input_seq = np.array(recent_history).reshape(1, self.window_size, 1)
        prediction = self.model.predict(input_seq, verbose=0)
        return max(0, prediction[0][0])

class DQNAgent:
    def __init__(self, action_space_size):
        self.action_space = action_space_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Real Deep Q-Network"""
        model = Sequential()
        # State: [Task_MIPS, Task_RAM, Predicted_System_Load]
        model.add(Input(shape=(3,))) 
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear')) # Output: Q-value for each Host
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=32):
        """Replay Memory Training"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0][0] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        
        # Batch prediction for speed
        targets = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_qs[i])
            targets[i][action] = target
            
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay