"""data preproc GNNs"""

import chess
import chess.pgn
from graph import mobility

def get_graphs(g: chess.pgn.Game, graph_fn=mobility, skip_first_n=10, skip_last_n=10):
    """
    chess.pgn.Game -> list of (x=graph_fn(board), y=win_or_loss) for the last_n boards of a game
    skips draws
    TODO: use efficiently updating graph generation instead of from scratch each time
    """
    outcome = g.game().headers['Result']
    if outcome == '1-0': outcome = True
    elif outcome == '0-1': outcome = False
    else: return [] # '1/2-1/2' and '*' (unknown)
    ret = []
    cur = g
    while cur.next() is not None:
        player = cur.board().turn
        win = player == outcome
        graph = graph_fn(cur.board())
        ret.append((graph, win))
        cur = cur.next()
    return ret[skip_first_n:-skip_last_n]