import math
from binomial_model import *

model = BinomialModel(r=.01, sigma=.2, T=1.0, N=5)

def call(K):
    def _call(S):
        return max(S - K, .0)
    return _call

hedge = Hedge(model, call(1.0))
hedge = hedge.hedge()


