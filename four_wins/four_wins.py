import numpy as np
import random as rnd
import json
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam  


class Player:
    @property
    def Symbol(self):
        return self._symbol
    
    def __init__(self, four_wins, symbol, alpha, alpha_decay, gamma, epsilon, epsilon_decay, epsilon_min, penalize_factor):
        self._four_wins = four_wins
        self._symbol = symbol
        self._dim_in = self._four_wins.M * self._four_wins.N * 3
        self._alpha = alpha
        self._alpha_decay = alpha_decay
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._penalize_factor = penalize_factor
        self._model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self._dim_in, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self._four_wins.N, activation='linear'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=self._alpha, decay=self._alpha_decay))
        return model

    def make_turn(self):
        # first save current state and player
        _current_state = self._four_wins.CurrentState

        if (np.random.random() <= self._epsilon):
            free_positions = self._free_positions()
            _next = free_positions[rnd.randint(0, len(free_positions) - 1)]
        else:
            _next = np.argmax(self.model.predict(_current_state))

        four_in_line, winner = self._four_wins.four_in_line()                                  
        game_over = four_in_line or len(self._free_positions()) == 0

        self._four_wins.UpdateGrid(_next, self._symbol)
        self._four_wins.UpdateHistory(State(_current_state, self._four_wins.State, self._player, _next, game_over, winner))

        return game_over, winner





class State(object):
    @property
    def CurrentState(self):
        return self._current_state

    @property
    def NextState(self):
        return self.__ext_state
    
    @property
    def Player(self):
        return self._player

    @property
    def NextPosition(self):
        return self._next_position

    @property
    def GameOver(self):
        return self._game_over

    @property
    def Winner(self):
        return self._winner

    def __init__(self, state, next_state, player, next_position, game_over, winner):
        self._current_state = state
        self._next_state = next_state
        self._player = player
        self._next_position = next_position
        self._game_over = game_over
        self._winner = winner

    @staticmethod
    def from_json(dict):
        return CurrentState(None, dict['_player'], dict['_next_position'], dict['_game_over'], dict['_winner'])

class StateEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, State):
                _attr = o.__dict__
                _attr.pop('_state', None)
                _attr.pop('_next_state', None)
                return _attr
            else:
                return json.JSONEncoder.default(self, o)

class FourWins:
    @property
    def State(self):
        return self._state

    @property
    def M(self):
        return self._m

    @property
    def N(self):
        return self._n

    @property
    def History(self):
        return self._history

    def reset(self):
        self._state = np.zeros((self._m, self._n))
        self._history = []

    def UpdateState(self, position, symbol):
        self._state[position[0], position[1]] = symbol

    def UpdateHistory(self, state):
        self._history.append(state)

    def set_players(self, player_A, player_B):
        self._player_A, self._player_B = player_A, player_B

    def _free_positions(self):
        m = len(self._four_wins.State)
        n = len(self._four_wins.State[0])
        free_positions = []

        for j in range(0, n):
            for i in range(m-1, -1, -1):
                if int(self._four_wins.State[i,j]) == 0:
                    free_positions.append([i,j])
                    break            

        return free_positions
    
    def FourInLine(self):    
        for i in range(0, self._m - 3):
            for j in range(0, self._n):               
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._state[i+k,j])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        for i in range(0, self._m):
            for j in range(0, self._n - 3):
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._state[i,j+k])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        for i in range(0, self._m - 3):
            for j in range(0, self._n - 3):
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._state[i+k,j+k])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        for i in range(0, self._m - 3):
            for j in range(3, self._n):
                symbols = []
                for k in range(0, 4):
                    symbols.append(self._state[i+k,j-k])
                if symbols.count(self._player_A.Symbol) == 4:
                    return True, self._player_A.Symbol
                if symbols.count(self._player_B.Symbol) == 4:
                    return True, self._player_B.Symbol

        return False, None

    def GameOver(self):
        if np.count_nonzero(self._state) == self._m * self._n:
            True

        if self._player_A.has_won() or self._player_B.has_won():
            return True

        return False

    def __init__(self, m=6, n=7):
        self._m = m
        self._n = n
        self._state = np.zeros((m, n))
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
        episodes = {}

        if save != None:
            path_serialization = os.path.join(os.path.dirname(__file__), "{0}_{1}_{2}.json".format(save, nb_episodes, episode_length))

            if os.path.exists(path_serialization):
                with open(path_serialization, 'r') as read_file:
                    _historyStr = read_file.read()
                    return json.loads(_historyStr)

        for episode in range(0, nb_episodes):
            _episode = {}
            for game in range(episode_length):
                self._reset()
                self.play()
                _episode["game {0}".format(game+1)] = self.History
            episodes["epsiode {0}".format(episode+1)] = _episode

        if save != None:
            _episodesStr = StateEncoder().encode(episodes)
            _epsiodes = json.loads(_episodesStr)
            path_serialization = os.path.join(os.path.dirname(__file__), "{0}_{1}_{2}.json".format(save, nb_episodes, episode_length))
            with open(path_serialization, 'w') as write_file:
                json.dump(_epsiodes, write_file)

        return episodes



four_wins = FourWins()
grid = four_wins.State

player_A = Player(four_wins, + 1)
player_B = Player(four_wins, - 1)

four_wins.set_players(player_A, player_B)

winner = four_wins.play()

history = four_wins.History


episodes = four_wins.get_episodes(10, 100, save="episodes\\history")

    