import math
from binomial_model import *
from binomial_agent import *

model = BinomialModel(r=.01, sigma=.2, T=1.0, N=10)
#paths = model.random_paths(20000)
epsiodes = model.get_episodes()

def call(K):
    def _call(S):
        return max(S - K, .0)
    return _call

payoff = call(1.0)
hedge = Hedge(model, payoff)
_hedge = hedge.hedge()

#for path in paths:
#    hedge.roll_out(_hedge, path)

agent = BinomialAgent(model, payoff, alpha=.65, gamma=.65, epsilon=1.0, epsilon_decay=.9995, epsilon_min=0.01, \
        penalize_factor=2.0, nu_S_l=-5.0, nu_S_u=+5.0, nu_S_steps=1000)

agent.train(epsiodes)


