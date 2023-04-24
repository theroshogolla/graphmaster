"""
chess.Board -> networkx.DiGraph
From Stanford paper https://snap.stanford.edu/class/cs224w-2013/projects2013/cs224w-023-final.pdf
From ChessY https://github.com/mrudolphlilith/ChessY
"""

import chess
import chess.pgn
import networkx as nx
import numpy as np
from collections import OrderedDict
import torch_geometric

"""from AlphaZero https://arxiv.org/abs/2009.04374"""
PIECE_VAL = {
    0: 0.00,
    chess.PAWN: 1.00,
    chess.KNIGHT: 3.05,
    chess.BISHOP: 3.33,
    chess.ROOK: 5.63,
    chess.QUEEN: 9.50,
    chess.KING: 200.00, 
}

sign = lambda x: bool(x > 0) - bool(x < 0)

def one_hot(cat_map: dict) -> np.ndarray:
    cat_arr = np.array(list(cat_map.values()))
    ret = np.zeros((cat_arr.size, cat_arr.max() + 1))
    ret[np.arange(cat_arr.size), cat_arr] = 1
    return ret

### node attr functions
def piece_value(b: chess.Board, val_map=PIECE_VAL):
    types = piece_type(b)
    return {loc:val_map[piece] for loc, piece in types.items()}

def piece_color(b: chess.Board) -> np.ndarray:
    """piece belonging to current player has index=1"""
    colors = {i: 0 for i in range(64)}
    for loc, piece in b.piece_map().items():
        colors[loc] = 1 if piece.color == b.turn else 2
    return one_hot(colors)

def piece_type(b: chess.Board) -> np.ndarray:
    types = {i: 0 for i in range(64)}
    for loc, piece in b.piece_map().items():
        types[loc] = piece.piece_type
    return one_hot(types)

def piece_color_type(b: chess.Board) -> np.ndarray:
    return np.concatenate([piece_color(b), piece_type(b)], axis=1)

### edge functions
def mobility(b: chess.Board, both=False) -> dict:
    """
    `both`: get graph for both players (current turn AND next turn)
    node are int locations with 'piece' attribute e.g. `4: {'piece': K}`
    TODO: separate graph (mobility) & attr (piece) collection
    """
    moves = [(m.from_square, m.to_square) for m in list(b.legal_moves)]
    if both:
        b.push(chess.Move.null()) # pass turn
        moves += [(m.from_square, m.to_square) for m in list(b.legal_moves)]
        b.pop() # undo pass turn
    return moves

def support(b: chess.Board):
    """dict of {from-piece}"""
    piece_map = b.piece_map()
    return {piece: [(target, piece_map[piece].color == piece_map[target].color)
                for target in b.attacks(piece).intersection(piece_map.keys())]
                    for piece in piece_map if piece_map[piece].color == b.turn}

# x = mobility(b)
# y = piece_value(b)
# y = {loc:{'val': val} for loc, val in y.items()}
# g = nx.DiGraph()
# g.add_nodes_from(range(64))
# g.add_edges_from(x)
# nx.set_node_attributes(g, y)
# nx.draw(g, {i: divmod(i, 8)[::-1] for i in range(64)}, labels=nx.get_node_attributes(g, 'val'))

def chess_nx(b: chess.Board, edge_fn=mobility):
    g = nx.DiGraph()
    g.add_nodes_from(range(64))
    g.add_edges_from(edge_fn(b))
    return g

def chess_pyg(b: chess.Board, edge_fn=mobility, node_fn=piece_color_type):
    g = chess_nx(b, edge_fn)
    pyg = torch_geometric.utils.convert.from_networkx(g)
    pyg.x = node_fn(b)
    return pyg
    