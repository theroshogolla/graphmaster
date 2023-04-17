"""
chess.Board -> networkx.DiGraph
From Stanford paper https://snap.stanford.edu/class/cs224w-2013/projects2013/cs224w-023-final.pdf
From ChessY https://github.com/mrudolphlilith/ChessY
"""

import chess
import networkx as nx

def mobility(b: chess.Board, both=False) -> nx.DiGraph:
    """
    `both`: get graph for both players (urrent turn AND next turn)
    node are int locations with 'piece' attribute e.g. `4: {'piece': K}`
    """
    moves = [(m.from_square, m.to_square, b.piece_at(m.from_square)) for m in list(b.legal_moves)]
    if both:
        b.push(chess.Move.null()) # pass turn
        moves += [(m.from_square, m.to_square, b.piece_at(m.from_square)) for m in list(b.legal_moves)]
        b.pop() # undo pass turn
    pieces = {i:{'piece': ' '} for i in range(64)}
    # pieces = OrderedDict((i,{'piece': None}) for i in range(64))
    pieces.update({loc:{'piece':piece.symbol()} for loc, piece in b.piece_map().items()})
    g = nx.DiGraph()
    g.add_nodes_from(tuple(pieces.items()))
    g.add_edges_from([e[:2] for e in moves])
    return g

def support(b: chess.Board, both=False) -> nx.DiGraph:
    pass