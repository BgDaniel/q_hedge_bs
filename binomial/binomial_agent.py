import numpy as np
import random as rnd
import math

class EarlyStopping:
    @property
    def Iterations(self):
        return self._iterations

    @property
    def Delta(self):
        return self._delta

    def __init__(self, delta, iterations):
        self._delta = delta
        self._iterations = iterations

class BinomialAgent:
    def __init__(self, model, hedge, hedge_strategy, alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor, \
        nu_S_l, nu_S_u, nu_S_steps, call_backs = None):
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
        
        self._call_backs = call_backs
        self._early_stopping = None

        for _call_back in call_backs:
            if type(_call_back) == EarlyStopping:
                self._early_stopping = _call_back


    def _reward(self, hedge_derivation):
        return self._penalize_factor * math.exp(- self._penalize_factor * abs(hedge_derivation))

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

    @staticmethod
    def find_closest(nu, _nu):
        if _nu <= nu[0]:
            return 0
        if _nu >= nu[-1]:
            return len(nu) - 1
        for i, value in enumerate(nu):
            if i == 0:
                continue
            if nu[i - 1] < _nu and _nu <= nu[i]:
                return i

    def train(self, episodes):
        hedge_error = np.zeros((len(episodes),self._N))
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
                nu_S[0] = self._hedge_strategy[0][0,2]
                hedge_actual = self._hedge.roll_out(self._hedge_strategy, path)
                
                # arbitrarily set up hedge portfolio for t=0
                i_S[0] = BinomialAgent.find_closest(self._nu_S, nu_S[0])
                nu_S[0] = self._nu_S[int(i_S[0])]
                nu_B[0] = (hedge[0] - nu_S[0] * S[0]) / B[0]
                pos = int(self._to_state(path, 0))
                current_state = [pos, int(i_S[0])]

                for i in range(0, self._N):
                    # choose action
                    if rnd.uniform(0, 1) > self._epsilon:
                        q_for_current_state = self._q_for(current_state)
                        i_S_new = np.argmax(self._q_for(current_state))
                    else:
                        i_S_new = rnd.randint(0, len(self._nu_S) - 1)

                    next_state = [current_state[0], i_S_new]
                        
                    nu_S[i] = self._nu_S[i_S_new]
                    nu_B[i] = (hedge[i] - nu_S[i] * S[i]) / B[i]                

                    # compute reward                
                    hedge[i+1] = nu_S[i] * S[i+1] + nu_B[i] * B[i+1]
                    hedge_derivation = hedge_actual[i+1] - hedge[i+1]
                    hedge_error[k,i-1] += hedge_derivation * hedge_derivation 
                    reward = self._reward(hedge_derivation)

                    # update q_table
                    self._update_q_(current_state, next_state, i_S_new, reward)

                    # update state
                    current_state = [self._to_state(path, i + 1), i_S_new]

            hedge_error[k] /= len(paths)
            hedge_error[k] = np.array([math.sqrt(hedge_error_i) for hedge_error_i in hedge_error[k]])

            if self._early_stopping != None and k + 1 >= self._early_stopping.Iterations:
                improvements = np.empty((self._N), dtype=bool)
                for i in range(0,self._N):
                    i_hedge_error = hedge_error[:,i]
                    i_last_hedge_errors = i_hedge_error[k-self._early_stopping.Iterations+1:k]
                    min_i_last_hedge_errors = i_last_hedge_errors.min()

                    print(hedge_error[k,i])
                    print(min_i_last_hedge_errors - self._early_stopping.Delta)

                    if hedge_error[k,i] < min_i_last_hedge_errors - self._early_stopping.Delta:
                        improvements[i] = True 
                    else:
                        improvements[i] = False

                if len(np.where(improvements=False)) == self._N:
                    return hedge_error
                    
            if self._epsilon >= self._epsilon_min:
                self._epsilon *= self._epsilon_decay

        return hedge_error

    def hedge(self, path):
        agent_hedge = np.zeros((self._N + 1))
        actual_hedge = np.zeros((self._N + 1))
        value_hedge[0] = self._hedge_strategy[0][0,0]


