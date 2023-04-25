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
    '-': 0.00,  # empty square
    'P': 1.00, 'p': -1.00,
    'N': 3.05, 'n': -3.05,
    'B': 3.33, 'b': -3.33,
    'R': 5.63, 'r': -5.63,
    'Q': 9.50, 'q': -9.50,
    'K': 200.00, 'k': -200.00,
}

# Node attribute functions
def piece_value(b: chess.Board) -> dict:
    pieces = {i: 0.00 for i in range(64)}
    turn_sign = 1 if b.turn else -1
    pieces.update({loc: turn_sign * PIECE_VALUE[piece.symbol()] for loc, piece in b.piece_map().items()})
    return pieces

def piece_color(b: chess.Board) -> dict:
    colors = {i: 0 for i in range(64)}
    for loc, piece in b.piece_map().items():
        colors[loc] = 1 if piece.color == chess.WHITE else -1
    return colors


def piece_type(b: chess.Board) -> dict:
    types = {i: 0 for i in range(64)}
    for loc, piece in b.piece_map().items():
        types[loc] = piece.piece_type
    return types

def attackers_defenders(b: chess.Board) -> dict:
    attack_defense = {i: {'attackers': 0, 'defenders': 0} for i in range(64)}
    for loc in range(64):
        attack_defense[loc]['attackers'] = len(b.attackers(chess.BLACK, loc)) + len(b.attackers(chess.WHITE, loc))
        attack_defense[loc]['defenders'] = len(b.attackers(chess.BLACK, loc)) if b.color_at(loc) == chess.WHITE else len(b.attackers(chess.WHITE, loc))
    
    combined_attack_defense = {loc: attack_defense[loc]['attackers'] - attack_defense[loc]['defenders'] for loc in range(64)}
    return combined_attack_defense


# Edge functions
def mobility(b: chess.Board, both=False) -> dict:
    moves = [(m.from_square, m.to_square) for m in list(b.legal_moves)]
    if both:
        b.push(chess.Move.null())  # pass turn
        moves += [(m.from_square, m.to_square) for m in list(b.legal_moves)]
        b.pop()  # undo pass turn
    return moves

def capture_moves(b: chess.Board, both=False) -> dict:
    moves = [(m.from_square, m.to_square) for m in list(b.generate_capture_moves())]
    if both:
        b.push(chess.Move.null())  # pass turn
        moves += [(m.from_square, m.to_square) for m in list(b.generate_capture_moves())]
        b.pop()  # undo pass turn
    return moves

def attack_defense_relationships(b: chess.Board) -> dict:
    relationships = []
    for loc in range(64):
        piece = b.piece_at(loc)
        if piece:
            attackers = b.attackers(not piece.color, loc)
            for attacker in attackers:
                if piece.color == chess.WHITE:
                    relationships.append((attacker, loc))
                else:
                    relationships.append((loc, attacker))
    return relationships


def chess_nx(b: chess.Board, edge_fn=mobility, node_fns=None):
    if node_fns is None:
        node_fns = [piece_value, piece_color, piece_type, attackers_defenders]

    g = nx.DiGraph()
    g.add_nodes_from(range(64))
    g.add_edges_from(edge_fn(b))

    attrs = {loc: {} for loc in range(64)}
    for node_fn in node_fns:
        fn_name = node_fn.__name__
        attr = node_fn(b)
        for loc in attrs:
            if loc in attr:
                attrs[loc].update({fn_name: attr[loc]})

    nx.set_node_attributes(g, attrs)
    return g


