import math
import numpy as np
import copy as cp
import random as rnd

class Hedge:
    def __init__(self, payoff, binomialModel, hedge):
        self._payoff = payoff
        self._model = binomialModel
        self._hedge = hedge

    def roll_out(self, path):
        n_0 = path[0][0]
        N = self._model.N
        value_hedge = np.zeros((N - n_0))
        value_payoff = np.zeros((N - n_0))

        for i in range(0, N - n_0):
            value_hedge[i] = self._model._S()

        value_hedge = np.zeros(())
        value = np.zeros(())

class State:
    @property
    def M(self):
        return self._m
    
    @property
    def N(self):
        return self._t

    def __init__(self, n, m):
        self._m = m
        self._n = n

    def __hash__(self):
        hash = 3541
        hash += 2953 * int(self._m) 
        hash += 3001 * int(self._n)
        return hash
    
class BinomialModel:
    @property 
    def N(self):
        return self._N

    def __init__(self, r, sigma, dt, T, N, S0 , B0):
        self._r = r
        self._sigma = sigma
        self._dt = float(T) / float(N)
        self._T = T
        self._S0 = S0
        self._B0 = B0
        self._u = math.exp(self._sigma * math.sqrt(self._dt))
        self._d = 1.0 / self._u
        self._qUp = (math.exp(r * dt) + math.exp(- sigma * math.sqrt(dt))) / (math.exp(sigma * math.sqrt(dt)) - math.exp(- sigma * math.sqrt(dt)))
        self._qDown = 1.0 - self._qUp

    def random_paths(self, nb_simus=1000):
        paths = np.zeros((nb_simus, self._N + 1))
        for i in range(0, nb_simus):
            path = np.zeros((self._N + 1))
            for j in range(0, self._N):
                jump = rnd.randint(0, 1)
                if jump == 1:
                    path[j] = + 1.0
                else:
                    path[j] = - 1.0
            paths[i] = path
        return paths

    def _to_S(self, path):
        S = np.zeros((len(path)+1))
        S[0] = self._S0
        for j in range(1,len(path)):
            S[j] = S[j-1] * (self._u ** path[j])
        return S
    
    def _to_B(self, path):
        B = np.zeros((len(path)+1))
        B[0] = self._B0
        for j in range(1,len(path)):
            B[j] = B[j-1] * math.exp(self._r * self._dt)
        return B
  
    def _value(self, v_up, v_down):
        return math.exp(- self._r * self._dt) * (self._qUp * v_up + self._qDown * v_down)
    
    def _S(self, state):
        n = state.n
        m = state.M
        assert m >= 0 and m <= n, 'Wrong input!'
        S = self._S0

        for i in range(0, m - 1):
            S *= self._u

        return S

    def _B(self, state):
        n = state.n
        B = self._B0

        for i in range(0, n - 1):
            B *= math.exp(self._dt)

        return B

    def hedge(self, state, payoff):
        n, m = state.N, state.M #states for time n: 0,...,n
        v_up, v_down, S_up, S_down = .0, .0, .0, .0
        hedge_up, hedge_down = [], []
        state_up, state_down = State(n + 1, m + 1), State(n + 1, m)

        if self._N - n == 1:
            S_up, S_down = self._S(state_up), self._S(state_down)
            v_up, v_down = payoff(S_up), payoff(S_down)
            nu_B, nu_S = self._nu_B(v_up, v_down), self._nu_S(v_up, v_down)
            return { state : [ self._value(v_up, v_down), nu_B, nu_S] }
        else:
            v_up, hedge_up = self.hedge(state_up, payoff)
            v_down, hedge_down = self.hedge(state_down, payoff)
            nu_B, nu_S = self._nu_B(v_up, v_down), self._nu_S(v_up, v_down)            
            hedge = self._update_hedge(hedge_up, hedge_down, { state : [nu_B, nu_S]})
            value = self._value(v_up, v_down)
            return value, hedge

    def _nu_B(self, v_up, v_down):
        return (math.exp(-self._sigma * math.sqrt(self._dt)) * v_up + math.exp(self._sigma * math.sqrt(self._dt)) * v_down) / \
                (math.exp(self._sigma * math.sqrt(self._dt) + self._r * self._dt) - \
                math.exp(-self._sigma * math.sqrt(self._dt) + self._r * self._dt))
    
    def _nu_S(self, v_up, v_down):
        return (v_up - v_down) / (math.exp(self._sigma * math.sqrt(self._dt)) - math.exp(-self._sigma * math.sqrt(self._dt)))

    def _update_hedge(self, hedge_up, hedge_down, hedge):
        hedge = cp.deepcopy(hedge)
        hedge.update(hedge_up)
        hedge.update(hedge_down)
        return hedge








