import numpy as np
import random as rnd
import math

class BinomialAgent:
    def __init__(self, model, hedge, hedge_strategy, alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor, \
        nu_S_l, nu_S_u, nu_S_steps):
        self._model = model
        self._N = model.N
        self._hedge = hedge
        self._hedge_strategy = hedge_strategy
        
        self._nu_S_l = nu_S_l
        self._nu_S_u = nu_S_u
        self._nu_S_steps = nu_S_steps             
       
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._penalize_factor = penalize_factor

        self._nu_S = np.linspace(self._nu_S_l, self._nu_S_u, self._nu_S_steps)     
        nb_states_model = int((self._N + 2.0) * (self._N + 1.0) / 2.0)
        self._q_table = np.zeros((nb_states_model,len(self._nu_S),len(self._nu_S)))

    def _reward(self, hedge_derivation):
        return 1.0 - math.exp(- self._penalize_factor * abs(hedge_derivation))

    def _update_q_(self, current_state, next_state, action, reward):
        Q = self._q_for(current_state)[action]                               
        max_next_q = np.max(self._q_for(next_state))       
        P = reward + self._gamma * max_next_q
        q_new = (1.0 - self._alpha) * Q + self._alpha * P
        self._set_q_for(current_state, action, q_new)

    def _to_state(self, path, t):
        state = 0
        if t == 0:
            return state
        for i in range(1, self._N + 1):
            if i > t:
                break
            if path[i - 1] == + 1.0:
                state += (i + 1)
            else:
                state += i
        return state

    def _q_for(self, state):
        return self._q_table[state[0], state[1]] 

    def _set_q_for(self, state, action, value):
        self._q_table[state[0], state[1], action] = value

    def hedge(self, path):
        for i in range(0, self._N + 1):
            state = self._to_state(path, i)
            i_S = np.argmax(self._q_for(state))
            nu_S = self._nu_S[i_S]


    def train(self, episodes):
        for k, paths in enumerate(episodes):
            print('Training of epsiode number {0} ...'.format(k + 1))
            for path in paths:
                S = self._model._to_S(path)
                B = self._model._to_B(path) 

                i_S = np.zeros((self._N))
                nu_S = np.zeros((self._N))
                nu_B = np.zeros((self._N))
                hedge = np.zeros((self._N) + 1)
                value = np.zeros((self._N))
                
                hedge[0] = self._hedge_strategy[0][0,0]
                hedge_actual = self._hedge.roll_out(self._hedge_strategy, path)
                
                # arbitrarily set up hedge portfolio for t=0
                i_S[0] = 0
                nu_S[0] = self._nu_S[int(i_S[0])]
                nu_B[0] = (hedge[0] - nu_S[0] * S[0]) / B[0]
                pos = int(self._to_state(path, 0))
                current_state = [pos, int(i_S[0])]

                for i in range(0, self._N + 1):
                    # choose action
                    if rnd.uniform(0, 1) > self._epsilon:
                        i_S_new = np.argmax(self._q_for(current_state))
                    else:
                        i_S_new = rnd.randint(0, len(self._nu_S) - 1)

                    next_state = [current_state[0], i_S_new]
                        
                    nu_S[i-1] = self._nu_S[i_S_new]
                    nu_B[i-1] = (hedge[i-1] - nu_S[i-1] * S[i-1]) / B[i-1]                

                    # compute reward                
                    hedge[i] = nu_S[i-1] * S[i] + nu_B[i-1] * B[i]
                    hedge_derivation = hedge_actual[i] - hedge[i]
                    reward = self._reward(hedge_derivation)

                    # update q_table
                    self._update_q_(current_state, next_state, i_S_new, reward)

                    # update state
                    current_state = [self._to_state(path, i), i_S_new]
                    
            if self._epsilon >= self._epsilon_min:
                self._epsilon *= self._epsilon_decay


