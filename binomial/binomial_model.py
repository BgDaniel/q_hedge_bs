import math
import numpy as np
import copy as cp
import random as rnd

class BinomialModel:
    @property 
    def N(self):
        return self._N

    def __init__(self, r, sigma, T, N, S0=1.0, B0=1.0):
        self._r = r
        self._sigma = sigma
        self._dt = float(T) / float(N)
        self._T = T
        self._N = N
        self._S0 = S0
        self._B0 = B0
        self._u = math.exp(self._sigma * math.sqrt(self._dt))
        self._d = 1.0 / self._u
        self._qUp = (math.exp(r * self._dt) - math.exp(- sigma * math.sqrt(self._dt))) / \
            (math.exp(sigma * math.sqrt(self._dt)) - math.exp(- sigma * math.sqrt(self._dt)))
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
    
    def _S(self, t, m):
        assert m >= 0 and m <= t, 'Wrong input!'
        S = self._S0

        for i in range(0, m): # m times up
            print(i)
            S *= self._u

        for j in range(0, t - m): # N - m times down
            print(j)
            S *= self._d

        return S

    def _B(self, t):
        B = self._B0

        for i in range(0, t):
            B *= math.exp(self._r * self._dt)

        return B

    def _nu_B(self, v_up, v_down, B):
        return (- math.exp(-self._sigma * math.sqrt(self._dt)) * v_up + math.exp(self._sigma * math.sqrt(self._dt)) * v_down) / \
                (B * (math.exp(self._sigma * math.sqrt(self._dt)) - math.exp(-self._sigma * math.sqrt(self._dt))))
    
    def _nu_S(self, v_up, v_down, S):
        return (math.exp(self._r * self._dt) * v_up - math.exp(self._r * self._dt) * v_down) / \
             (S * (math.exp(self._sigma * math.sqrt(self._dt)) - math.exp(-self._sigma * math.sqrt(self._dt))))

class Hedge:
    def __init__(self, model, payoff):
        self._model = model
        self._payoff = payoff

    def hedge(self):
        hedge = []
        N = self._model.N
   
        for i_t, t in enumerate(range(self._model.N - 1, -1, -1)):
            hedge_t = np.zeros((t + 1, 5))
            for m in range(0, t + 1):
                S = self._model._S(t, m)
                B = self._model._B(t)                
                if t == N - 1:
                    S_up, S_down = self._model._S(N, m + 1), self._model._S(N, m)
                    v_up, v_down = self._payoff(S_up), self._payoff(S_down)
                else:
                    v_up, v_down = hedge[i_t - 1][m + 1, 0], hedge[i_t - 1][m, 0]
                value = self._model._value(v_up, v_down)
                nu_B, nu_S = self._model._nu_B(v_up, v_down, B), self._model._nu_S(v_up, v_down, S)
                hedge_t[m] = np.array([value, nu_B, nu_S, B, S])
            
            hedge.append(hedge_t)

        return hedge.reverse()

    def _roll_out(self, hedge, path): # for validation purpose
        N = self._model.N
        value_0 = hedge[0][0]
        nu_B_0, nu_S_0 = hedge[0][1], hedge[0][2]

        for t in range(1, N + 1):
            S = self._model._S(t, m)
            B = self._model._B(t) 


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








