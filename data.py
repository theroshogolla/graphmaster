"""
Read ChessDB dataset from
https://www.kaggle.com/datasets/milesh1/35-million-chess-games
https://chess-research-project.readthedocs.io/
"""
import pandas as pd
import chess
import chess.pgn
import io
from torch_geometric.data import InMemoryDataset
import data as my_data
from tqdm import tqdm
import random
import numpy as np

tqdm.pandas()

FPATH = '/home/asy51/repos/graphmaster/dataset/all_with_filtered_anotations_since1998.txt'

def parse(fpath=FPATH, nrows=1000, skip_draws=True) -> pd.DataFrame:
    df = pd.read_csv(fpath, engine='python', skiprows=4, sep='###', nrows=nrows)
    df = df.reset_index()
    newcols = ['game_ndx', 'Date', 'Result', 'welo' ,'belo', 'len', 'date_c', 'resu_c',
               'welo_c', 'belo_c', 'edate_c', 'setup', 'fen', 'resu2_c', 'oyrange', 'bad_len']
    df[newcols] = df['index'].str.split(' ', expand=True).iloc[:,:-1]
    df['moves'] = df.iloc[:,1]
    df['moves'] = df['moves'].str.replace('[WB]\d+?\.', '', regex=True)
    df['Result'] = df['Result'].map({'1-0': 1, '0-1': 0, '1/2-1/2': 2}).fillna(-1).astype(int)
    if skip_draws:
        df = df[(df['Result'] == 0) | (df['Result'] == 1)]

    df = df.iloc[:,2:].set_index('game_ndx')
    
    df = df[(df['setup'] == 'setup_false') & (df['bad_len'] == 'blen_false') & (df['moves'].notna())]
    return df

def get_game(row: pd.Series):
    moves = io.StringIO(row['moves'])
    game = chess.pgn.read_game(moves)
    game.headers.update(row.drop(['game_ndx', 'moves', 'move_ndx'], errors='ignore'))
    return game

def get_boards(g: chess.pgn.Game, skip_first_n=10, skip_last_n=10):
    """
    chess.pgn.Game -> list of (dict['board' + meta]) for each move in game
    skips draws
    TODO: use efficiently updating graph generation instead of from scratch each time
    """
    
    # if isinstance(meta, str): meta = [meta]
    # if 'win' in meta:
    #     outcome = g.game().headers['Result']
    #     if outcome == '1-0': outcome = True
    #     elif outcome == '0-1': outcome = False
    #     else: return [] # '1/2-1/2' and '*' (unknown)
    ret = []
    cur = g
    while cur.next() is not None:
        # if 'win' in meta:
        #     item_dict['win'] = cur.board().turn == outcome
        ret.append(cur.board())
        cur = cur.next()
    return ret[skip_first_n:-skip_last_n]

class ChessDataset:
    def __init__(self, n_games=1_000, skip_first_n=10, skip_last_n=10):
        self.skip_first_n = skip_first_n
        self.skip_last_n = skip_last_n
        df = parse(nrows=n_games)
        
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
        return self.games[game_ndx]['game'], self.games[game_ndx]['boards'][move_ndx - self.skip_first_n]
