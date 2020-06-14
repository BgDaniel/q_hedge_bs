import math
import numpy as np
import matplotlib.pyplot as plt
from binomial_model import *
from binomial_agent import *


model = BinomialModel(r=.01, sigma=.2, T=1.0, N=4)
#paths = model.random_paths(20000)
epsiodes = model.get_episodes(60, 6000)

def call(K):
    def _call(S):
        return max(S - K, .0)
    return _call

payoff = call(1.0)
hedge = Hedge(model, payoff)
hedge_strategy = hedge.hedge()

#for path in paths:
#    hedge.roll_out(hedge_strategy, path)

early_stopping = EarlyStopping(0.0001, 5)

max_nu_S = None
min_nu_S = None

agent = BinomialAgent(model, hedge, hedge_strategy, alpha=.65, gamma=.65, epsilon=1.0, epsilon_decay=.85, epsilon_min=0.01, \
        penalize_factor=1.0, nu_S_l=-1.0, nu_S_u=+1.0, nu_S_steps=2000, call_backs=[early_stopping])

hedge_error = agent.train(epsiodes)

for error in np.transpose(hedge_error):
    plt.plot(error)
plt.show()

agent_hedge, actual_hedge = agent.hedge(epsiodes[0][0])
plt.plot(agent_hedge)
plt.plot(actual_hedge)
plt.show()


