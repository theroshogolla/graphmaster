import argparse
import chess
import chess.engine
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

import data as my_data
import pred as my_pred

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_games', default=1_000, type=int)
# parser.add_argument('--test_split', default=0.1, type=float)
# parser.add_argument('--batch_size', default=256, type=int)
# parser.add_argument('--net', default='gcn', type=str)
# parser.add_argument('--n_hidden_ch', default=256, type=int)
# parser.add_argument('--lr', default=1e-3, type=float)
# parser.add_argument('-e', '--n_epochs', default=1_000, type=int)
parser.add_argument('--pred_thresh', default=0.5, type=float)
parser.add_argument('--pred_time', default=0.01, type=float)
parser.add_argument('--engine', default='stockfish', type=str)
# parser.add_argument('--dropout_rate', default=0.2, type=float)
CONFIG = vars(parser.parse_args())
print(CONFIG)

engine = my_pred.init_engine(CONFIG['engine'])

skip_first_n=10
skip_last_n=10
df = my_data.parse(n_games=CONFIG['n_games'])
df = df.drop(df[df['len'] <= skip_first_n + skip_last_n].index) # avoid copy vs. view warning

pred = []
y = []
for row_ndx in range(len(df)):
    try:
        game = my_data.get_game(df.iloc[row_ndx])
        boards = my_data.get_boards(game, skip_first_n=skip_first_n, skip_last_n=skip_last_n)
        game_pred = torch.tensor([my_pred.win_pred(b, engine, pov='white', time=CONFIG['pred_time']) > CONFIG['pred_thresh'] for b in boards], dtype=int)
        game_y = torch.ones_like(game_pred) * int(game.headers['Result'])
        tn, fp, fn, tp = confusion_matrix(game_y, game_pred, normalize='all', labels=(0,1)).ravel()
        acc = accuracy_score(game_y, game_pred, normalize=True)
        print(f'game[{row_ndx}]:  tn:{tn:.3f} fp:{fp:.3f} fn:{fn:.3f} tp:{tp:.3f} acc:{acc:.3f}')

        pred.append(game_pred)
        y.append(game_y)
    except Exception as e:
        print(e)
        engine.quit()
        del engine
        engine = my_pred.init_engine(CONFIG['engine'])
    
pred = torch.cat(pred).to(int)
y = torch.cat(y).to(int)
tn, fp, fn, tp = confusion_matrix(y, pred, normalize='all', labels=(0,1)).ravel()
acc = accuracy_score(y, pred, normalize=True)
print(f'total:  tn:{tn:.3f} fp:{fp:.3f} fn:{fn:.3f} tp:{tp:.3f} acc:{acc:.3f}')

engine.quit()