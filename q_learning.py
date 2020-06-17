from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam  
from collections import deque
import numpy as np
import random as rnd
import collections
from four_wins import *
import random as rnd
from black_scholes import *

class Portfolio:    
    def __init__(self, S_0, B_0, delta_0, value_0):
        self._S = S_0
        self._B = B_0
        self._delta = delta_0
        self._value = value_0
        self._b = (value_0 - delta_0 * S_0) / B_0

    def update(self, S, B):
        self._S = S
        self._B = B
        self._value = self.value()

    def value(self):
        return self._delta * self._S + self._b * self._B

    def rebalance(self, delta_new):
        value = self.value()
        self._delta = delta_new
        self._b = (value - self._delta * self._S) / self._B

class HedgeAgent:
    def __init__(self, portfolio, threshold_hedge_error, delta_lower, delta_upper, delta_steps,
        epsilon, epsilon_decay, epsilon_min, alpha, alpha_decay, dt):
        self._portfolio = portfolio
        self._threshold_hedge_error = threshold_hedge_error
        self._state_space = np.zeros((3)) # T X S_T X Delta_T
        self._delta_lower = delta_lower
        self._delta_upper = delta_upper
        self._delta_steps = delta_steps
        self._action_space = np.linspace(delta_lower, delta_upper, delta_steps)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._alpha = alpha
        self._alpha_decay = alpha_decay
        self._model = self._set_up_model()
        self._memory = []
        self._dt = dt

    def _set_up_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self._alpha, decay=self._alpha_decay))
        return model

    def _delta(self, t, S_t):
        if (np.random.random() <= self._epsilon):
            return rnd.choice(self._action_space)
        return np.argmax(self._model.predict([t, S_t]))

    def _reward(self, value_hedge, value_real):
        if abs(value_hedge - value_real) < self._threshold_hedge_error * self._dt:
            return + 1.0
        else:
            return - 1.0

    def train(self, episodes):
        for episode in episodes:
            time, B, S, delta_real, value = episode[0], episode[1], episode[2], episode[3], episode[4]
            for i in range(0, len(S)):
                _S, _delta_real, _value = S[i], delta_real[i], value[i]

                for j in range(0, len(time)-1):
                    t = time[j]
                    delta = self._delta(t, _S[j])
                    self._portfolio.rebalance(delta)

                    self._portfolio.update(_S[j+1], B[j+1])

                    value_next_hedge = self._portfolio.value()
                    value_next_real = _value[j+1]
                    reward = self._reward(value_next_hedge, value_next_real)

                    self._memory.append([t, _S[j], delta, reward])

                    if reward == - 1.0:
                        break                    

            if self._epsilon > self._epsilon_min:
                self._epsilon *= self._epsilon_decay

S0 = 1.0
B0 = 1.0

hedge = HedgeEuropeanCallBS(nbSimus=100, nbSteps=100, S0=S0, B0=B0, sigma=0.2, r=.01, T=1.0, N=1.0, K=.9)

delta0 = .5
value0 = hedge.Value0
portfolio = Portfolio(S0, B0, delta0, value0)

episodes = hedge.episodes(nb_episodes=2)
hedge_agent = HedgeAgent(portfolio=portfolio, threshold_hedge_error=.01, delta_lower=.0, delta_upper=1.0, delta_steps=100,
        epsilon=1.0, epsilon_decay=.995, epsilon_min=.03, alpha=.01, alpha_decay=.01, dt=hedge.Dt)
hedge_agent.train(episodes)