from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam  
from collections import deque
import numpy as np
import random as rnd

class Agent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor, memo_length=100000):
        
               
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._penalize_factor = penalize_factor
        self._memo = []
        self.memory = deque(memo_length)

 
        if state_size is None: 
            self.state_size = self.env.observation_space.n 
        else: 
            self.state_size = state_size
 
        if action_size is None: 
            self.action_size = self.env.action_space.n 
        else: 
            self.action_size = action_size
 
        self.episodes = episodes
        self.env._max_episode_steps = max_env_steps
        self.win_threshold = win_threshold

        self.batch_size = batch_size
        self.path = path                     #location where the model is saved to
        self.prints = prints                 #if true, the agent will print his scores
 
        self.model = self._build_model()

    def remember(self, state, action, reward, next_state, done):      
        self.memory.append((state, action, reward, next_state, done))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model

    def act(self, state):
        if (np.random.random() <= self.epsilon):
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = rnd.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
            
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay