import math
from binomial_model import *

model = BinomialModel(r=.01, sigma=.2, T=1.0, N=10)
paths = model.random_paths(20000)

def call(K):
    def _call(S):
        return max(S - K, .0)
    return _call

hedge = Hedge(model, call(1.0))
_hedge = hedge.hedge()

for path in paths:
    hedge.roll_out(_hedge, path)


