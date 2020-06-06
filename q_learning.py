import numpy as np
import math
import random as rnd
from black_scholes import HedgeEuropeanCallBS
from helpers import *

class Agent:
    def __init__(self, hedge_european_call, max_steps, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, \
        nb_steps_S, nb_steps_t, percentile_S, nb_S_lower, nb_S_upper, nb_S_steps):
        self._hedge_european_call = hedge_european_call
        self._european_call = hedge_european_call.EuropeanCallBS
        self._dt = hedge_european_call.Dt
        self._S = hedge_european_call.S
        self._S0 = self._hedge_european_call.S0
        self._B = hedge_european_call.B
        self._nb_simus = self._hedge_european_call.NbSimus
        self._nb_steps_S = nb_steps_S
        self._nb_steps_t = nb_steps_t
        self._T = hedge_european_call.T
        self._delta_t = float(self._T) / float(nb_steps_t)
        self._percentile_S = percentile_S
        self._nb_S_lower = nb_S_lower
        self._nb_S_upper = nb_S_upper
        self._nb_S_steps = nb_S_steps                
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min,
        self._epilson_decay = epsilon_decay

        self._B_int, self._S_int = self.interpolate()
        self._S_mapped = self.map_to_state_space()

        self._actions = np.linspace(self._nb_S_lower, self._nb_S_upper, self._nb_S_steps)
        self._q_table = np.zeros((self._nb_steps_t,self._nb_steps_S,self._nb_S_steps))

    def interpolate(self):
        S_int = np.zeros((self._nb_steps_t, self._nb_simus))
        B_int = np.zeros((self._nb_steps_t))

        for i_time in range(0, self._nb_steps_t):
            S_int[i_time] = interpol_values_S(self._S, i_time * self._delta_t, self._dt)
            B_int[i_time] = interpol_values_B(self._B, i_time * self._delta_t, self._dt)

        return B_int, S_int

    def map_to_state_space(self):
        S_mapped = np.zeros((self._nb_steps_t, self._nb_simus))
        percentiles = np.linspace(self._percentile_S, 1.0 - self._percentile_S, num=self._nb_steps_S)

        for i_time in range(1, self._nb_steps_t):
            S = self._S_int[i_time]
            states = np.array([np.percentile(S, 100.0 * perc) for perc in percentiles])            
            
            for j_simu in range(0, self._nb_simus):
                q = float((np.array([S <= S[j_simu]])).sum()) / float(self._nb_simus)
                s = np.array([percentiles <= q]).sum()
                S_mapped[i_time,j_simu] = s - 1

        return S_mapped

    def train(self):
        for i_path, path in enumerate(self._S):
            nb_B = np.zeros((self._nb_steps_t))
            nb_S = np.zeros((self._nb_steps_t))
            value_hedge = np.zeros((self._nb_steps_t))
            value_hedge[0] = self._european_call.price(.0, self._S0) 
            action = 0

            if rnd.uniform(0, 1) > self._epsilon:
                action = np.argmax(self._q_table[0:self._S_mapped[0:i_path]])
            else:
                action = rnd.randint(0, len(self._actions) - 1)
                nb_S[0] = self._actions[action]

            nb_B[0] = (value_hedge[0] - nb_S[0] * self._S[i_path,0]) / self._B[0]

            for i_time in range(1, self._nb_steps_t):
                S = self._S_int
                
                value_hedge[i_time] = nb_S[i_time-1] * S[i_time, i_path] + nb_B[i_time-1] * self._B_int[i_time]
                value_hedge_real = self._european_call.price( i_time * self._delta_t, S[i_time, i_path]) 

                # compute reward
                hedge_loss = value_hedge_real - value_hedge[i_time]
                reward = 1.0 - math.exp(- 3.0 * math.abs(hedge_loss))

                # update q_table
                self._q_table[0:self._S_mapped[0:i_path]:action] = (1.0 - self._learning_rate) * self._q_table[0:self._S_mapped[0:i_path]:action] \
                    + self._learning_rate * (reward + self._gamma * )
                
                nb_S = .0

                if rnd.uniform(0, 1) > self._epsilon:
                    action = np.argmax(qtable[state, :])
                else:
                    nb_S[i_time] = self._actions(rnd(0, len(self._actions) - 1))

                nb_B[i_time] = (value_hedge[0] - nb_S[i_time] * self._S[i_path,i_time]) / self._B[i_time]

     

            if self._epsilon >= self._epsilon_min:
                self._epsilon *= self._epsilon_decay



 


