"""
Read ChessDB dataset from
https://www.kaggle.com/datasets/milesh1/35-million-chess-games
https://chess-research-project.readthedocs.io/
"""
import pandas as pd
import chess
import chess.pgn
import io

FPATH = '/home/asy51/repos/graphmaster/dataset/all_with_filtered_anotations_since1998.txt'

def parse(fpath=FPATH, nrows=1000) -> pd.DataFrame:
    df = pd.read_csv(fpath, engine='python', skiprows=4, sep='###', nrows=nrows)
    df = df.reset_index()
    newcols = ['id', 'Date', 'Result', 'welo' ,'belo', 'len', 'date_c', 'resu_c',
               'welo_c', 'belo_c', 'edate_c', 'setup', 'fen', 'resu2_c', 'oyrange', 'bad_len']
    df[newcols] = df['index'].str.split(' ', expand=True).iloc[:,:-1]
    df['moves'] = df.iloc[:,1]
    df['moves'] = df['moves'].str.replace('[WB]\d+?\.', '', regex=True)
    df = df.iloc[:,2:].set_index('id')
    
    df = df[(df['setup'] == 'setup_false') & (df['bad_len'] == 'blen_false') & (df['moves'].notna())]
    return df

def get_game(row: pd.Series):
    moves = io.StringIO(row['moves'])
    game = chess.pgn.read_game(moves)
    game.headers.update(row.drop('moves'))
    return game

def get_boards(g: chess.pgn.Game, skip_first_n=10, skip_last_n=10, meta=['win']):
    """
    chess.pgn.Game -> list of (dict['board' + meta]) for each move in game
    skips draws
    TODO: use efficiently updating graph generation instead of from scratch each time
    """
    
    if isinstance(meta, str): meta = [meta]
    if 'win' in meta:
        outcome = g.game().headers['Result']
        if outcome == '1-0': outcome = True
        elif outcome == '0-1': outcome = False
        else: return [] # '1/2-1/2' and '*' (unknown)
    ret = []
    cur = g
    while cur.next() is not None:
        item_dict = dict()
        item_dict['board'] = cur.board()
        if 'win' in meta:
            item_dict['win'] = cur.board().turn == outcome
        ret.append(item_dict)
        cur = cur.next()
    return ret[skip_first_n:-skip_last_n]