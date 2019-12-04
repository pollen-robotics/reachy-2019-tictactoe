import os
import operator
import numpy as np


q = os.path.join(os.path.dirname(__file__), 'Q-value.npz')
Q = np.load(q)

Q = {1: Q['QX'], 2: Q['QO']}


def value_actions(board, next_player=1):
    possible_actions = np.where(np.array(board) == 0)[0]

    possibilities = {}
    for action in possible_actions:
        next_board = board.copy()
        next_board[action] = next_player

        val = Q[next_player][tuple(next_board)]
        possibilities[action] = val

    possibilities = sorted(possibilities.items(), key=operator.itemgetter(1))

    if next_player == 1:
        possibilities = list(reversed(possibilities))

    return possibilities
