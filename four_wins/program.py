from four_wins import FourWins, Player


four_wins = FourWins()
player_A = Player(four_wins, + 1)
player_B = Player(four_wins, - 1)

four_wins.set_players(player_A, player_B)
winner = four_wins.play()

episodes_A, episodes_B = four_wins.get_episodes(10, 4000)