import numpy as np
import random as rnd
import json
import os

class Player:
    @property
    def Symbol(self):
        return self._symbol
    
    def __init__(self, four_wins, symbol):
        self._four_wins = four_wins
        self._symbol = symbol

    def make_turn(self):        
        free_positions = self._free_positions() 

        _next = free_positions[rnd.randint(0, len(free_positions) - 1)]
        self._four_wins.Set_Position(_next, self._symbol)
        
        four_in_line, winner = self._four_wins.four_in_line()                                  
        game_over = four_in_line or len(self._free_positions()) == 0
        
        self._four_wins.Append_Turn(self._symbol, _next, game_over, winner)
        
        return game_over, winner

    def _drop(self, position):                            
        self._four_wins.Set_Position(position, self._symbol)

    def _free_positions(self):
        m = len(self._four_wins.Grid)
        n = len(self._four_wins.Grid[0])
        free_positions = []

        for j in range(0, n):
            for i in range(m-1, -1, -1):
                if int(self._four_wins.Grid[i,j]) == 0:
                    free_positions.append([i,j])
                    break            

        return free_positions

class State(object):
    def __init__(self, grid, player, next_position, game_over, winner):
        if type(grid) != list:
            self._grid = grid.tolist()
        else:
            self._grid = grid
        self._player = player
        self._next_position = next_position
        self._game_over = game_over
        self._winner = winner

    @staticmethod
    def from_json(dict):
        return State(dict['_grid'], dict['_player'], dict['_next_position'], dict['_game_over'], dict['_winner'])

class StateEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, State):
                return o.__dict__
            else:
                return json.JSONEncoder.default(self, object)

class FourWins:
    @property
    def Grid(self):
        return self._grid

    @property
    def History(self):
        return self._history

    def _reset(self):
        self._grid = np.zeros((self._m, self._n))
        self._history = []

    def Set_Position(self, position, symbol):
        self._grid[position[0], position[1]] = symbol

    def set_players(self, player_A, player_B):
        self._player_A, self._player_B = player_A, player_B
    
    def four_in_line(self):    
        for i in range(0, self._m - 3):
            for j in range(0, self._n):               
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._grid[i+k,j])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        for i in range(0, self._m):
            for j in range(0, self._n - 3):
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._grid[i,j+k])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        for i in range(0, self._m - 3):
            for j in range(0, self._n - 3):
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._grid[i+k,j+k])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        for i in range(0, self._m - 3):
            for j in range(3, self._n):
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._grid[i+k,j-k])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        return False, None

    def game_over(self):
        if np.count_nonzero(self._grid) == self._m * self._n:
            True

        if self._player_A.has_won() or self._player_B.has_won():
            return True

        return False

    def Append_Turn(self, player, next_position, game_over, winner):
        self._history.append(State(self._grid, player, next_position, game_over, winner))

    def __init__(self, m=6, n=7):
        self._m = m
        self._n = n
        self._grid = np.zeros((m, n))
        self._player_A = None
        self._player_B = None
        self._history = []

    def play(self):
        game_over = False

        while not game_over:
            game_over, winner = self._player_A.make_turn()

            if not game_over:
                game_over, winner = self._player_B.make_turn()

        return winner

    def get_episodes(self, nb_episodes, episode_length, save=None):
        episodes = []

        for episode in range(0, nb_episodes):
            history = []
            for i in range(episode_length):
                self._reset()
                self.play()

                if save != None:
                    path_serialization = os.path.join(os.path.dirname(__file__), "{0}_{1}_{2}_{3}.json".format(save, nb_episodes, episode_length, episode))
                    if os.path.exists(path_serialization):
                        with open(path_serialization, 'r') as read_file:
                            _historyStr = read_file.read()
                            _history = json.loads(_historyStr)
                            _history = [State.from_json(_run) for _run in _history]
                    else:
                        _historyStr = StateEncoder().encode(self.History)
                        _history = json.loads(_historyStr)
                        with open(path_serialization, 'w') as write_file:
                            json.dump(_history, write_file)

                history.append(_history)
            episodes.append(history)

        return episodes



four_wins = FourWins()
grid = four_wins.Grid

player_A = Player(four_wins, + 1)
player_B = Player(four_wins, - 1)

four_wins.set_players(player_A, player_B)

winner = four_wins.play()

history = four_wins.History


episodes = four_wins.get_episodes(10, 100, save="episodes\\history")

    