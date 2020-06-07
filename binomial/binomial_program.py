import math
from binomial_model import BinomialModel, State

binModel = BinomialModel(r=.01, sigma=.2, dt=0.05, T=7, S0=1.0 , B0=1.0)

def call(K):
    def _call(S):
        return max(S - K, .0)
    return _call

value, hedge = binModel.hedge(State(2, 1), call(1.0))

print(value)

