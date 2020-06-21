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
from sklearn import model_selection as ms
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Portfolio:    
    def __init__(self, S_0, B_0, delta_0, value_0):
        self._S0 = S_0
        self._S = S_0
        self._B0 = B_0
        self._B = B_0
        self._delta0 = delta_0
        self._delta = delta_0
        self._value0 = value0
        self._value = value_0
        self._b0 = (value_0 - delta_0 * S_0) / B_0
        self._b = self._b0

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


    def reset(self):
        self.__init__(self._S0, self._B0, self._delta0, self._value0)

class HedgeAgent:
    def __init__(self, portfolio, error_tol, error_threshold, delta_lower, delta_upper, delta_steps,
        epsilon, epsilon_decay, epsilon_min, reward_factor, call_backs, batch_size):
        self._portfolio = portfolio
        self._reward_factor = reward_factor
        self._error_tol = error_tol
        self._error_threshold = error_threshold
        self._state_space = 3 # T X S_T X Delta_T
        self._delta_lower = delta_lower
        self._delta_upper = delta_upper
        self._delta_steps = delta_steps
        self._action_space = np.linspace(delta_lower, delta_upper, delta_steps)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._model = self._set_up_model()
        self._states = []
        self._call_backs = call_backs
        self._model_history = []
        self._batch_size = batch_size

    def _set_up_model(self):
        model = Sequential()
        model.add(Dense(9, input_dim=self._state_space, activation='sigmoid'))
        model.add(Dense(27, activation='sigmoid'))
        model.add(Dense(27, activation='sigmoid'))
        model.add(Dense(9, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def _delta(self, t, S_t):
        if (np.random.random() <= self._epsilon):
            return rnd.choice(self._action_space)
        return np.argmax(self._model.predict([t, S_t]))

    def _reward(self, value_hedge, value_real):
        rel_dev = (value_hedge - value_real) / value_real
        return self._reward_factor * (- rel_dev + self._error_tol)

    def _get_batch(self):
       return np.array(self._states[:,[0,2]]), np.array(self._states[:,-1])

    def _replay(self):
        batch = rnd.sample(self._states, min(len(self._states, self._batch_size)))
        for state in batch:
            pass

    def train(self, episodes):
        for j, episode in enumerate(episodes):
            time, B, S, delta_real, value = episode[0], episode[1], episode[2], episode[3], episode[4]
            for i in range(0, len(S)):
                self._portfolio.reset()
                _S, _delta_real, _value = S[i], delta_real[i], value[i]

                for j in range(0, len(time)-1):
                    t = time[j]
                    delta = self._delta(t, _S[j])
                    self._portfolio.rebalance(delta)

                    self._portfolio.update(_S[j+1], B[j+1])

                    value_next_hedge = self._portfolio.value()
                    value_next_real = _value[j+1]
                    reward = self._reward(value_next_hedge, value_next_real)

                    self._states.append([t, _S[j], delta, reward])

                    if reward < self._error_threshold:
                        break 

            if j == 0:
                X, y = self._get_batch()
                X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.1, random_state=32)
                _model_history = self._model.fit(X_train, y_train, epochs=25, callbacks=self._call_backs, \
                    validation_data=(X_test, y_test))
                self._model_history.append(_model_history)

            # replay
            self._replay()

            if self._epsilon > self._epsilon_min:
                self._epsilon *= self._epsilon_decay

S0 = 1.0
B0 = 1.0

hedge = HedgeEuropeanCallBS(nbSimus=100, nbSteps=100, S0=S0, B0=B0, sigma=0.2, r=.01, T=1.0, N=1.0, K=.9)

delta0 = .5
value0 = hedge.Value0
portfolio = Portfolio(S0, B0, delta0, value0)

episodes = hedge.episodes(nb_episodes=2, save=True)

call_back = EarlyStopping(monitor = 'binary_crossentropy', patience=3, min_delta=10e-6)

hedge_agent = HedgeAgent(portfolio=portfolio, error_tol=0.02, error_threshold=.05, delta_lower=.0, delta_upper=1.0, delta_steps=100,
        epsilon=1.0, epsilon_decay=.995, epsilon_min=.03, reward_factor=1.0, call_backs=[call_back], batch_size=1000)

hedge_agent.train(episodes)