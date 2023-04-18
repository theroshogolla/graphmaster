"""
chess.Board -> networkx.DiGraph
From Stanford paper https://snap.stanford.edu/class/cs224w-2013/projects2013/cs224w-023-final.pdf
From ChessY https://github.com/mrudolphlilith/ChessY
"""

import chess
import chess.pgn
import networkx as nx

"""from AlphaZero https://arxiv.org/abs/2009.04374"""
PIECE_VALUE = {
    '-': 0.00, # empty square
    'P': 1.00, 'p': -1.00,
    'N': 3.05, 'n': -3.05,
    'B': 3.33, 'b': -3.33,
    'R': 5.63, 'r': -5.63,
    'Q': 9.50, 'q': -9.50,
    'K': 200.00, 'k': -200.00,
}

### node attr functions
def piece_value(b: chess.Board) -> dict:
    pieces = {i: 0.00 for i in range(64)}
    # pieces = OrderedDict((i,{'piece': None}) for i in range(64))
    turn_sign = 1 if b.turn else -1
    pieces.update({loc: turn_sign * PIECE_VALUE[piece.symbol()] for loc, piece in b.piece_map().items()})
    return pieces

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

# def support(b: chess.Board, both=False) -> nx.DiGraph:
#     pass

# def support(b: chess.Board):
#     """dict of {from-piece}"""
#     piece_map = b.piece_map()
#     return {piece: [(target, piece_map[piece].color == piece_map[target].color)
#                 for target in b.attacks(piece).intersection(piece_map.keys())]
#                     for piece in piece_map}

# x = mobility(b)
# y = piece_value(b)
# y = {loc:{'val': val} for loc, val in y.items()}
# g = nx.DiGraph()
# g.add_nodes_from(range(64))
# g.add_edges_from(x)
# nx.set_node_attributes(g, y)
# nx.draw(g, {i: divmod(i, 8)[::-1] for i in range(64)}, labels=nx.get_node_attributes(g, 'val'))

def chess_nx(b: chess.Board, edge_fn=mobility, node_fns=piece_value):
    g = nx.DiGraph()
    g.add_nodes_from(range(64))
    g.add_edges_from(edge_fn(b))

    if not isinstance(node_fns, (list, tuple)): node_fns = [node_fns]
    attrs = {loc: {} for loc in range(64)}
    for node_fn in node_fns:
        fn_name = node_fn.__name__
        attr = node_fn(b)
        for loc in attrs:
            attrs[loc].update({fn_name: attr[loc]})
        
    nx.set_node_attributes(g, attrs)
    return g