from four_wins import FourWins, Player
from agent import *


four_wins = FourWins()
#player_A = Player(four_wins, + 1)
#player_B = Player(four_wins, - 1)

#four_wins.set_players(player_A, player_B)
#winner = four_wins.play()

#epsiodes = four_wins.get_episodes(10, 100, save="episodes\\history")

#agent_A, agent_B = get_pair(four_wins=four_wins, alpha=.6, alpha_decay=.6, gamma=.6, epsilon=1.0, \
    #epsilon_decay=.995, epsilon_min=.01, penalize_factor=3.0)

#four_wins.set_players(agent_A, agent_B)
#winner = four_wins.play()

train_four_wins = TrainFourWins(alpha=.6, alpha_decay=.6, gamma=.6, epsilon=1.0, epsilon_decay=.995, \
    epsilon_min=.01, penalize_factor=3.0, nb_episodes=20, length_episodes=100)
train_four_wins.train()
