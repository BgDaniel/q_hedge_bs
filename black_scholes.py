import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import os

class BlackScholesModel:
    @property
    def NbSteps(self):
        return self._nbSteps

    @property
    def NbSimus(self):
        return self._nbSimus

    @property
    def S0(self):
        return self._S0

    @property
    def Dt(self):
        return self._dt

    @property
    def T(self):
        return self._T

    def __init__(self, nbSimus, nbSteps, S0, B0, sigma, r, T):
        self._S0 = S0
        self._B0 = B0
        self._sigma = sigma
        self._r = r
        self._T = T
        self._nbSimus = nbSimus
        self._nbSteps = nbSteps
        self._times = np.linspace(.0, T, nbSteps)
        self._dt = float(T) / float(nbSteps)

    def generate_paths(self, use_caching=False):
        B = np.zeros((self._nbSteps))
        B[0] = self._B0

        for i_time in range(1, self._nbSteps):
            B[i_time] = B[i_time-1] + self._r * B[i_time-1] * self._dt
        
        if use_caching:
            path_to_csv = os.path.join(os.path.dirname(__file__), 'generated_paths\\sym_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(self._S0, \
                self._B0, self._sigma, self._r, self._T, self._nbSimus, self._nbSteps))

            if os.path.exists(path_to_csv):
                data = pd.read_csv(path_to_csv, sep=";")
                S = data.values
                S = np.delete(S, 0, 1)
                return B, S

        S = np.zeros((self._nbSimus, self._nbSteps))
        dt_sqrt = math.sqrt(self._dt)
       
        self._dW_t = np.random.normal(size =(self._nbSimus, self._nbSteps)) * dt_sqrt

        for simu in range(0, self._nbSimus):
            S[simu,0] = self._S0        

        for i_time in range(1, self._nbSteps):
            B[i_time] = B[i_time-1] + self._r * B[i_time-1] * self._dt
            for simu in range(0, self._nbSimus): 
                S[simu,i_time] = S[simu,i_time-1] + S[simu,i_time-1] * (self._r * self._times[i_time] * self._dt \
                    + self._sigma * self._dW_t[simu,i_time-1])

        if use_caching:
            path_to_csv = os.path.join(os.path.dirname(__file__), 'generated_paths\\sym_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(self._S0, \
                self._B0, self._sigma, self._r, self._T, self._nbSimus, self._nbSteps))

            data = pd.DataFrame(data=S)
            data.to_csv(path_to_csv, sep=";") 

        return B, S    

    def plot_paths(self, paths, simu=10):
        for simu in range(0, simu): 
            plt.plot(paths[simu])
                        
        plt.show()

class EuropeanCallBS:
    def __init__(self, N, K, T, sigma, r):
        self._N = N
        self._K = K
        self._T = T
        self._sigma = sigma
        self._r = r

    def _d1(self, t, S):
        return (math.log(S / self._K) + (self._r + .5 * self._sigma * self._sigma) * (self._T - t)) / (self._sigma * math.sqrt(self._T - t))

    def _d2(self, t, S):
        return self._d1(t, S) - self._sigma * math.sqrt(self._T - t)

    def price(self, t, S):
        return S * self._N * norm.cdf(self._d1(t, S)) - math.exp(- self._r * (self._T - t)) * self._K * self._N * norm.cdf(self._d2(t, S))

    def delta(self, t, S):
        return self._N * norm.cdf(self._d1(t, S)) 

class HedgeEuropeanCallBS:
    @property
    def S(self):
        return self._S

    @property
    def S0(self):
        return self._S0

    @property
    def B(self):
        return self._B

    @property
    def NbSteps(self):
        return self._nbSteps

    @property
    def Dt(self):
        return self._dt

    @property
    def T(self):
        return self._T

    @property
    def NbSimus(self):
        return self._nbSimus

    @property
    def EuropeanCallBS(self):
        return self._europeanCallBS

    @property
    def Value0(self):
        return self._value0

    def __init__(self, nbSimus, nbSteps, S0, B0, sigma, r, T, N, K):
        self._bs_model = BlackScholesModel(nbSimus, nbSteps, S0, B0, sigma, r, T)
        self._S0 = self._bs_model.S0
        self._B, self._S = self._bs_model.generate_paths()
        self._nbSteps = nbSteps
        self._europeanCallBS = EuropeanCallBS(N, K, T, sigma, r)
        self._value0 = self._europeanCallBS.price(.0, self._bs_model.S0) 
        self._dt = self._bs_model.Dt
        self._T = self._bs_model.T
        self._nbSimus = self._bs_model.NbSimus
        
    def hedge(self):
        v_B = np.zeros((self._bs_model.NbSimus, self._bs_model.NbSteps))
        v_S = np.zeros((self._bs_model.NbSimus, self._bs_model.NbSteps))
        time = np.zeros((self._bs_model.NbSteps))
        value_hedge = np.zeros((self._bs_model.NbSimus, self._bs_model.NbSteps))
        value_analytical = np.zeros((self._bs_model.NbSimus, self._bs_model.NbSteps))

        value_hedge_0 = self._europeanCallBS.price(.0, self._bs_model.S0)        
        v_S0 = self._europeanCallBS.delta(.0, self._bs_model.S0)
        v_B0 = (value_hedge_0 - v_S0 * self._bs_model.S0) / (self._B[0])

        for i_simu in range(0, self._bs_model.NbSimus):
            value_hedge[i_simu,0] = value_hedge_0 
            v_S[i_simu,0] = v_S0
            v_B[i_simu,0] = v_B0
            value_analytical[i_simu,0] = value_hedge_0

        for i_time in range(1, self._bs_model.NbSteps):
            time[i_time] = i_time * self._dt
            for i_simu in range(0, self._bs_model.NbSimus):
                value_hedge[i_simu,i_time] = v_B[i_simu,i_time-1] * self._B[i_time] + v_S[i_simu,i_time-1] * self._S[i_simu,i_time]
                v_S[i_simu,i_time] = self._europeanCallBS.delta(i_time * self._bs_model.Dt, self._S[i_simu,i_time])
                v_B[i_simu,i_time] = (value_hedge[i_simu,i_time]  - v_S[i_simu,i_time] * self._S[i_simu,i_time]) / (self._B[i_time])
                value_analytical[i_simu,i_time] = self._europeanCallBS.price(i_time * self._bs_model.Dt, self._S[i_simu,i_time])
            
        return time, self._B, self._S, v_B, v_S, value_hedge, value_analytical

    def episodes(self, nb_episodes):
        episodes = []
        for episode in range(0, nb_episodes):
            time, B, S, _, delta, _, value_analytical = self.hedge()
            episodes.append([time, B, S, delta, value_analytical])

        return episodes

