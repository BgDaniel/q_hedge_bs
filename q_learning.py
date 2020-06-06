import numpy as np
from black_scholes import HedgeEuropeanCallBS

class Agent:
    def __init__(self, hedge_european_call, max_steps, learning_rate, gamma, epsilon, epsilon_decay, nb_steps_S, percentile_S):
        self._hedge_european_call = hedge_european_call
        self._S = hedge_european_call.S
        self._nb_steps_S = nb_steps_S
        self._nb_steps_t = self._hedge_european_call.NbSteps
        self._percentile_S = percentile_S
        
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._epilson_decay = epsilon_decay

        self._state_space = self.init_state_space()

    def init_state_space(self):
        state_space = np.zeros((self._nb_steps_t,self._nb_steps_S))

        for i_time in range(0, self._nb_steps_t):
            lower_percentile_i = np.percentile(self._S[i_time], self._percentile_S) 
            upper_percentile_i = np.percentile(self._S[i_time], 1.0 - self._percentile_S) 
            state_space[i_time] = np.linspace(lower_percentile_i, upper_percentile_i, num=self._nb_steps_S)
