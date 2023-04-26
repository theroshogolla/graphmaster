"""
Read ChessDB dataset from
https://www.kaggle.com/datasets/milesh1/35-million-chess-games
https://chess-research-project.readthedocs.io/
chess.Board -> networkx.DiGraph
From Stanford paper https://snap.stanford.edu/class/cs224w-2013/projects2013/cs224w-023-final.pdf
From ChessY https://github.com/mrudolphlilith/ChessY
"""
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import io
import random
import torch_geometric
import torch_geometric.data
import chess
import chess.pgn
from collections import OrderedDict
import torch

tqdm.pandas()

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

FPATH = '/home/asy51/repos/graphmaster/dataset/all_with_filtered_anotations_since1998.txt'

sign = lambda x: bool(x > 0) - bool(x < 0)

def parse(fpath=FPATH, nrows=1000, skip_draws=True) -> pd.DataFrame:
    df = pd.read_csv(fpath, engine='python', skiprows=4, sep='###', nrows=nrows*2)

    df = df.reset_index()
    newcols = ['game_ndx', 'Date', 'Result', 'welo' ,'belo', 'len', 'date_c', 'resu_c',
               'welo_c', 'belo_c', 'edate_c', 'setup', 'fen', 'resu2_c', 'oyrange', 'bad_len']
    df[newcols] = df['index'].str.split(' ', expand=True).iloc[:,:-1]
    df['moves'] = df.iloc[:,1]
    df['moves'] = df['moves'].str.replace('[WB]\d+?\.', '', regex=True)

    df['Result'] = df['Result'].map({'1-0': 1, '0-1': 0, '1/2-1/2': 2}).fillna(-1).astype(int)
    df['date_c'] = df['date_c'].map({'date_true': True, 'date_false': False})
    df['resu_c'] = df['resu_c'].map({'resu_true': True, 'result_false': False})
    df['welo_c'] = df['welo_c'].map({'welo_true': True, 'welo_false': False})
    df['belo_c'] = df['belo_c'].map({'belo_true': True, 'belo_false': False})
    df['edate_c'] = df['edate_c'].map({'edate_true': True, 'edate_false': False})
    df['setup'] = df['setup'].map({'setup_true': True, 'setup_false': False})
    df['fen'] = df['fen'].map({'fen_true': True, 'fen_false': False})
    df['resu2_c'] = df['resu2_c'].map({'result2_true': True, 'result2_false': False})
    df['oyrange'] = df['oyrange'].map({'oyrange_true': True, 'oyrange_false': False})
    df['bad_len'] = df['bad_len'].map({'blen_true': True, 'blen_false': False})
    df['welo'] = pd.to_numeric(df['welo'], errors='coerce').fillna(-1).astype(int)
    df['belo'] = pd.to_numeric(df['belo'], errors='coerce').fillna(-1).astype(int)
    df['len'] = pd.to_numeric(df['len'], errors='coerce').fillna(-1).astype(int)

    if skip_draws:
        df = df[(df['Result'] == 0) | (df['Result'] == 1)]

    df = df.iloc[:,2:].set_index('game_ndx')
    df = df[(~df['setup']) & (~df['bad_len']) & (df['moves'].notna()) & (df['len'] < 100)]
    # df = df[(df['setup'] == 'setup_false') & (df['bad_len'] == 'blen_false') & (df['moves'].notna())]
    return df.sample(n=nrows)

def get_game(row: pd.Series):
    moves = io.StringIO(row['moves'])
    game = chess.pgn.read_game(moves)
    game.headers.update(row.drop(['game_ndx', 'moves', 'move_ndx'], errors='ignore').astype(str)) # chess.pgn.Headers only accepts str
    return game

def get_boards(g: chess.pgn.Game, skip_first_n=10, skip_last_n=10):
    """
    chess.pgn.Game -> list of (dict['board' + meta]) for each move in game
    skips draws
    TODO: use efficiently updating graph generation instead of from scratch each time
    """
    ret = []
    cur = g
    while cur.next() is not None:
        ret.append(cur.board())
        cur = cur.next()
    return ret[skip_first_n:-skip_last_n]

def one_hot(cat_map: dict):
    cat_arr = np.array(list(cat_map.values()))
    ret = np.zeros((cat_arr.size, cat_arr.max() + 1))
    ret[np.arange(cat_arr.size), cat_arr] = 1
    return ret

### node attr functions
def piece_value(b: chess.Board, val_map=PIECE_VAL):
    types = piece_type(b)
    return {loc:val_map[piece] for loc, piece in types.items()}

def piece_color(b: chess.Board):
    """piece belonging to current player has index=1"""
    colors = {i: 0 for i in range(64)}
    for loc, piece in b.piece_map().items():
        colors[loc] = 1 if piece.color == b.turn else 2
    return one_hot(colors)

def piece_type(b: chess.Board):
    types = {i: 0 for i in range(64)}
    for loc, piece in b.piece_map().items():
        types[loc] = piece.piece_type
    return one_hot(types)

def piece_color_type(b: chess.Board):
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

def chess_nx(b: chess.Board, edge_fn=mobility):
    g = nx.DiGraph()
    g.add_nodes_from(range(64))
    g.add_edges_from(edge_fn(b))
    return g

def chess_pyg(g: chess.pgn.Game, b: chess.Board, edge_fn=mobility, node_fn=piece_color_type):
    g = chess_nx(b, edge_fn)
    pyg = torch_geometric.utils.convert.from_networkx(g)
    pyg.x = node_fn(b)
    return pyg

def pyg_data(g: chess.pgn.Game, b: chess.Board, edge_fn=mobility, node_fn=piece_color_type) -> torch_geometric.data.Data:
    return torch_geometric.data.Data(
        x=torch.tensor(node_fn(b), dtype=torch.float32),
        edge_index=torch.tensor(edge_fn(b), dtype=torch.long).T,
        y=torch.tensor(int(g.headers['Result']) == b.turn, dtype=torch.float32)
    )

def tabular_data(g: chess.pgn.Game, b: chess.Board, node_fn=piece_color_type) -> torch.Tensor:
    return {'x': node_fn(b), 'y': torch.tensor(int(g.headers['Result']) == b.turn, dtype=torch.float32)}

class ChessDataset:
    def __init__(self, df=None, n_games=1_000, skip_first_n=10, skip_last_n=10, net='gcn'):
        self.skip_first_n = skip_first_n
        self.skip_last_n = skip_last_n
        self.net = net
        if df is not None:
            pass
        elif n_games is not None:
            df = parse(nrows=n_games)
        else:
            raise ValueError # error
        
        df = df.drop(df[df['len'] <= skip_first_n + skip_last_n].index) # avoid copy vs. view warning
        df['move_ndx'] = df.apply(lambda row: list(range(skip_first_n, int(row['len']) - skip_last_n)), axis=1)
        self.df = df.explode('move_ndx').reset_index()
        self.games = {int(game_ndx): None for game_ndx in self.df['game_ndx'].unique()}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ndx):
        """(game, board)"""
        row = self.df.iloc[ndx]
        game_ndx, move_ndx = row[['game_ndx', 'move_ndx']].astype(int)
        if not self.games[game_ndx]:
            self.games[game_ndx] = dict()
            self.games[game_ndx]['game'] = get_game(row)
            self.games[game_ndx]['boards'] = get_boards(self.games[game_ndx]['game'], skip_first_n=self.skip_first_n, skip_last_n=self.skip_last_n)
        
        g = self.games[game_ndx]['game']
        b = self.games[game_ndx]['boards'][move_ndx - self.skip_first_n]
        return g,b
        # return self.games[game_ndx]['game'], self.games[game_ndx]['boards'][move_ndx - self.skip_first_n]

    def len(self):
        return self.__len__()
    
    def get(self, ndx):
        return self.__getitem__(ndx)
    
class GraphDataset(ChessDataset):
    def __getitem__(self, ndx):
        g,b = super().__getitem__(ndx)
        return pyg_data(g, b)
    
class TabularDataset(ChessDataset):
    def __getitem__(self, ndx):
        g,b = super().__getitem__(ndx)
        return tabular_data(g, b)
    
class RandomDataset:
    def __init__(self):
        pass

    def __len__(self):
        return 5_711_690
    
    def __getitem__(self, _):
        # return {'x': torch.randn((64, 10)), 'y': torch.randn(1) > 0}
        return torch_geometric.data.Data(
            x=torch.randn((64, 10)),
            edge_index=torch.randint(0, 63, (2, 35), dtype=torch.long),
            y=(torch.randn(1) > 0).to(torch.float32)
        )