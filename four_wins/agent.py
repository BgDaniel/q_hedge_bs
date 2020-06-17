from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam  
from collections import deque
import numpy as np
import random as rnd
import collections
from four_wins import *
import random as rnd

def get_pair(four_wins, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor, memo_length=100000):
    return Agent(four_wins, +1, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor), \
        Agent(four_wins, -1, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor)

class Agent(Player):
    @property
    def Model(self):
        return self._model

    def __init__(self, four_wins, symbol, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor, memo_length=100000):
        Player.__init__(self, four_wins, symbol)
        self._four_wins = four_wins 
        self._symbol = symbol  
        self._dim_in = self._four_wins.M * self._four_wins.N * 3
        self._alpha = alpha
        self._alpha_decay = alpha_decay
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._penalize_factor = penalize_factor
      
        
        self._memory = deque(maxlen=memo_length)
        self._model = self._build_model()
      
    def make_turn(self):
        if (np.random.random() <= self._epsilon):
            # make random choice
            return Player.make_turn(self)
        else:
            current_state = self._four_wins.Grid
            q_values_for_actions = self._model.predict(current_state)
            _next = np.argmax(q_values_for_actions)

            self._four_wins.Set_Position(_next, self._symbol)
        
            four_in_line, winner = self._four_wins.four_in_line()                                  
            game_over = four_in_line or len(self._free_positions()) == 0

            self._four_wins.Append_Turn(self._symbol, _next, game_over, winner)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self._dim_in, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self._four_wins.N, activation='linear'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=self._alpha, decay=self._alpha_decay))
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


class TrainFourWins:
    def __init__(self, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor, nb_episodes, length_episodes):
        self._four_wins = FourWins()
        self._agent_A, self._agent_B = get_pair(four_wins, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor)
        four_wins.set_players(self._agent_A, self._agent_B)
        self._nb_episodes = nb_episodes
        self._length_episodes = length_episodes        
        
    def train(self):        
        for episode in range(0, self._nb_episodes):
            episode = []
            for game in range(0, self._length_episodes):
                four_wins.reset()
                four_wins.play()
                episode.append(four_wins.History)

            states, q_values = [], []
            # replay and train
            for hist_game in episode:
                for state in hist_game:
                    
                    # determine reward
                    reward = .0
                    if state.GameOver:
                        if state.Winner == state.Player:
                            reward = + 1.0
                        else:
                            reward == - 1.0
                    
                    _next = state.Player.Model.predict(state)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
