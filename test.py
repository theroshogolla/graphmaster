"""logs to wandb only"""
from IPython import embed
import argparse
import chess
import chess.engine
import torch
from tqdm import tqdm
import wandb

import util as my_util
import data as my_data
import pred as my_pred

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_games', default=1_000, type=int)
parser.add_argument('--pred_thresh', default=0.5, type=float)
parser.add_argument('--pred_time', default=0.1, type=float)
parser.add_argument('--engine', default='stockfish', type=str)
CONFIG = vars(parser.parse_args())
print(CONFIG)

engine = my_pred.init_engine(CONFIG['engine'])

skip_first_n=10
skip_last_n=10
df = my_data.parse_cleaned(n_games=CONFIG['n_games'])
df = df.sample(frac=1).reset_index(drop=True)
# wandb.init(mode="disabled")
wandb.init(project='chess', config=CONFIG)

pred = []
y = []
n_engine_reset = 0
for row_ndx in tqdm(range(len(df))):
    try:
        game = my_data.get_game(df.iloc[row_ndx])
        boards = my_data.get_boards(game, skip_first_n=skip_first_n, skip_last_n=skip_last_n)

        game_pred = torch.tensor([my_pred.win_pred(b, engine, pov='white', time=CONFIG['pred_time']) for b in boards], dtype=int)
        game_y = torch.ones_like(game_pred) * int(game.headers['Result'])
        metric_dict = my_util.metrics(game_y, game_pred, CONFIG['pred_thresh'], prefix='game_')
        print({k:v for k,v in metric_dict.items() if 'curve' not in k})
        wandb.log(metric_dict)

        pred.append(game_pred)
        y.append(game_y)
    except Exception as e:
        n_engine_reset += 1
        print(f"{e}, {n_engine_reset}th engine reset")
        engine.quit()
        del engine
        engine = my_pred.init_engine(CONFIG['engine'])

    if row_ndx % 10 == 0:
        if not y or not pred:
            print(f'len(y)={len(y)}, len(pred)={len(pred)}')
        else:
            metric_dict = my_util.metrics(torch.cat(y).to(int), torch.cat(pred).to(int), CONFIG['pred_thresh'], prefix='')
        print({k:v for k,v in metric_dict.items() if 'curve' not in k})
        wandb.log(metric_dict)

engine.quit()