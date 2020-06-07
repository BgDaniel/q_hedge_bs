import numpy as np
import random as rnd
import math

class BinomialAgent:
    def __init__(self, model, payoff, state_0, alpha, gamma, epsilon, epsilon_decay, epsilon_min, \
        penalize_factor, nb_steps_S, nb_steps_t, percentile_S, nu_S_l, nu_S_u, nu_S_steps):
        self._model = model
        self._N = model.N
        
        self._payoff = payoff
        self._value_0, self._hedge = model.hedge(state_0)
        

        
        self._nu_S_l = nu_S_l
        self._nu_S_u = nu_S_u
        self._nu_S_steps = nu_S_steps             
       
       

        self.alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min,
        self._epilson_decay = epsilon_decay
        self._penalize_factor = penalize_factor



        self._nu_S = np.linspace(self._nu_S_l, self._nu_S_u, self._nu_S_steps)     
        nb_states_model = int((self._N + 2.0) * (self._N + 1.0) / 2.0)
        self._q_table = np.zeros((nb_states_model,len(self._nu_S),self._nb_actions))

    def _reward(self, hedge_derivation):
        return 1.0 - math.exp(- self._penalize_factor * math.abs(hedge_derivation))

    def _update_q_(self, current_state, next_state, action, reward):
        Q = self._q_for[current_state][action]                        
        max_next_q = np.argmax(self._q_for[next_state])        
        P = reward + self._gamma * max_next_q
        q_new = (1.0 - self._alpha) * Q + self._alpa * P
        self._set_q_for(current_state, action, q_new)

    def _to_state(self, path, t):
        return None

    def _q_for(self, state):
        return self._q_table[state[0], state[1]] 

    def _set_q_for(self, state, action, value):
        self._q_table[state[0], state[1], action] = value

    def train(self, episodes):
        for paths in episodes:
            for path in paths:
                S = self._model._S(path)
                B = self._model._B(path) 

                i_S = np.zeros((self._N + 1))
                nu_S = np.zeros((self._N + 1))
                nu_B = np.zeros((self._N + 1))
                hedge = np.zeros((self._N + 1))
                hedge_actual = np.zeros((self._N + 1))
                
                hedge[0] = .0
                
                # arbitrarily set up hedge portfolio for t=0
                i_S[0] = 0
                nu_S[0] = self._nu_S[i_S[0]]
                nu_B[0] = (hedge[0] - nu_S[0] * S[0]) / B[0]
                current_state = [0, i_S]

                for i in range(1, self._N):
                    # choose action
                    if rnd.uniform(0, 1) > self._epsilon:
                        i_S_new = np.argmax(self._q_for(current_state))
                    else:
                        i_S_new = rnd.randint(0, self._nb_actions - 1)

                    next_state = [current_state[0], i_S_new]
                        
                    nu_S[i-1] = self._nu_S[i_S_new]
                    nu_B[i-1] = (hedge[i-1] - nu_S[i-1] * S[i-1]) / B[i-1]                

                    # compute reward                
                    hedge[i] = nu_S[i-1] * S[i] + nu_B[i-1] * B[i]
                    hedge_derivation = hedge_actual[i] - hedge[i]
                    reward = self._reward(hedge_derivation)

                    # update q_table
                    self._update_q_(current_state, next_state, i_S_new, reward)

                    current_state = 
                    
            if self._epsilon >= self._epsilon_min:
                self._epsilon *= self._epsilon_decay

