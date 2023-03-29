"""
Read ChessDB dataset from
https://www.kaggle.com/datasets/milesh1/35-million-chess-games
https://chess-research-project.readthedocs.io/
"""
import pandas as pd
import chess
import chess.pgn
import io

FPATH = 'dataset/all_with_filtered_anotations_since1998.txt'


def parse(fpath=FPATH, nrows=1000) -> pd.DataFrame:
    df = pd.read_csv(fpath, engine='python', skiprows=4, sep='###', nrows=nrows)
    df = df.reset_index()
    newcols = ['id', 'Date', 'Result', 'welo' ,'belo', 'len', 'date_c', 'resu_c',
               'welo_c', 'belo_c', 'edate_c', 'setup', 'fen', 'resu2_c', 'oyrange', 'bad_len']
    df[newcols] = df['index'].str.split(' ', expand=True).iloc[:,:-1]
    df['moves'] = df.iloc[:,1]
    df['moves'] = df['moves'].str.replace('[WB]\d+?\.', '', regex=True)
    df = df.iloc[:,2:].set_index('id')
    return df

def get_game(row: pd.Series):
    moves = io.StringIO(row['moves'])
    game = chess.pgn.read_game(moves)
    game.headers.update(row.drop('moves'))
    return game
