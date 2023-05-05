import wandb
import multiprocessing
import torch
import train as my_train

# DEFAULT_CONFIG = {'n_games': 1000,
#     'test_split': 0.1,
#     'batch_size': 256,
#     'net': 'gcn',
#     'n_hidden_ch': 256,
#     'lr': 0.001,
#     'n_epochs': 1000,
#     'pred_thresh': 0.5,
#     'dropout_rate': 0.2,
#     'node_fn': 'piece_color_type',
#     'edge_fn': 'position',
#     'engine': 'actual',
#     'pred_time': 0.01,
#     'wandb': False
# }

def init():
    '''set up config and start sweep'''
    sweep_config = {
        'method': 'grid',
        'name': 'muh_sweep',
        'parameters': {
            # 'batch_size': {'values': [32, 256]},
            'net': {'values': ['gcn', 'graphsage', 'gat',]},
            # 'n_hidden_ch': {'values': [256, 1024]},
            'lr': {'values': [1e-3, 1e-4]},
            'dropout_rate': {'values': [0.2, 0.5]},
            # 'target': {'values': ['stockfish', 'actual']},
            'loss': {'values': ['bce', 'focal']},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='chess')
    return sweep_id

def run():
    '''one sweep instance'''
    wandb.init()
    global CONFIG
    config = CONFIG.copy()
    config.update(wandb.config)

    # da whole thang
    my_train.main(config, internal_wandb=False)

def agent(sweep_id):
    '''run agent'''
    wandb.agent(sweep_id, function=run, count=None, project='chess')

# cmdline config
CONFIG = my_train.cmdline_config()
NUM_WORKERS = 24

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_start_method('spawn')
    # initialize sweep
    sweep_id = init()
    # spawn and run _ agents for the sweep
    procs = []
    for _ in range(NUM_WORKERS):
        p = multiprocessing.Process(target=agent, args=[sweep_id])
        p.start()
        procs.append(p)
    for p in procs:
        p.join()