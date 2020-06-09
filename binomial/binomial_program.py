import math
from binomial_model import *
from binomial_agent import *

model = BinomialModel(r=.01, sigma=.2, T=1.0, N=5)
#paths = model.random_paths(20000)
epsiodes = model.get_episodes(40, 2000)

def call(K):
    def _call(S):
        return max(S - K, .0)
    return _call

payoff = call(1.0)
hedge = Hedge(model, payoff)
hedge_strategy = hedge.hedge()

#for path in paths:
#    hedge.roll_out(hedge_strategy, path)

agent = BinomialAgent(model, hedge, hedge_strategy, alpha=.65, gamma=.65, epsilon=1.0, epsilon_decay=1.0, epsilon_min=0.01, \
        penalize_factor=8.0, nu_S_l=-2.0, nu_S_u=+2.0, nu_S_steps=20)

agent.train(epsiodes)


