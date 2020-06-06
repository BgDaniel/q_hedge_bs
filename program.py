import matplotlib.pyplot as plt
from black_scholes import HedgeEuropeanCallBS
from q_learning import Agent

bs_hedge = HedgeEuropeanCallBS(nbSimus=100, nbSteps=2000, S0=1.0, B0=1.0, sigma=.3, r=.04, T=1.0, N=1000, K=.8)
#_1, _2, value_hedge, value_analytical = bs_hedge.hedge()

#plt.plot(value_hedge[0])
#plt.plot(value_analytical[0])
#plt.show()
agent = Agent(bs_hedge, max_steps=1000, learning_rate=.65, gamma=.65, epsilon=1.0, epsilon_decay=.9999, nb_steps_S=100, percentile_S=.01)