from IPython import embed
import torch_geometric.loader
from tqdm import tqdm
import torch
import torch.utils.data
import argparse
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import data as my_data
import model as my_model
import pred as my_pred

tqdm.pandas()

def cmdline_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_games', default=1_000, type=int)
    parser.add_argument('--test_split', default=0.1, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--net', default='gcn', type=str)
    parser.add_argument('--n_hidden_ch', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-e', '--n_epochs', default=1_000, type=int)
    parser.add_argument('--pred_thresh', default=0.5, type=float)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--node_fn', default='piece_color_type', type=str)
    parser.add_argument('--edge_fn', default='position', type=str)
    parser.add_argument('--engine', default='actual_result', type=str)
    parser.add_argument('--pred_time', default=0.01, type=float)
    parser.add_argument('--wandb', action='store_true')
    config = vars(parser.parse_args())
    return config

def main(config, internal_wandb=True):
    node_fn = getattr(my_data, config['node_fn'])
    if config['node_fn'] == 'piece_color_type':
        attrs_per_node = 10
    elif config['node_fn'] == 'piece_color':
        attrs_per_node = 3
    elif config['node_fn'] == 'piece_type':
        attrs_per_node = 7
    else: raise ValueError(f"node_fn:{config['node_fn']} not implemented")
    edge_fn = getattr(my_data, config['edge_fn'])
    engine = None
    if config['engine'] == 'stockfish':
        engine = my_pred.init_engine('stockfish')
    elif config['engine'] == 'lc0':
        engine = my_pred.init_engine('lc0')
    elif config['engine'] == 'actual_result':
        pass
    else: raise ValueError(f"target:{config['engine']} not implemented")

    if internal_wandb:
        if config['wandb']:
            wandb.init(project='chess', config=config)
        else:
            #init dummy wandb
            wandb.init(mode="disabled")
    print(config)
    print(f'Using node_fn={node_fn.__name__}(); edge_fn={edge_fn.__name__}()')

    ### DATA
    df = my_data.parse(n_games=config['n_games'], skip_draws=True)
    df_train, df_test = train_test_split(df, test_size=config['test_split'])

    if config['net'] == 'gcn':
        ds_train = my_data.GraphDataset(df_train, node_fn=node_fn, edge_fn=edge_fn, engine=engine)
        ds_test = my_data.GraphDataset(df_test, node_fn=node_fn, edge_fn=edge_fn, engine=engine)
        dl_train = torch_geometric.loader.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        dl_test = torch_geometric.loader.DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        model = my_model.GCN(config, n_node_features=attrs_per_node, n_cls=1, sigmoid=True)
    elif config['net'] == 'fc':
        ds_train = my_data.TabularDataset(df_train, node_fn=node_fn, engine=engine)
        ds_test = my_data.TabularDataset(df_test, node_fn=node_fn, engine=engine)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        model = my_model.DumbNet(config, n_features=64 * attrs_per_node, n_cls=1, sigmoid=True)

    print(f"train {len(ds_train)} samples / {config['batch_size']} batches = {-(-len(ds_train) // config['batch_size']):0.0f} iters per epoch")

    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCELoss()
    print(model)


    def train():
        model.train()
        pred_accum = []
        y_accum = []
        for data in tqdm(dl_train, desc=f'E{epoch:03d}: TRAIN'):  # Iterate in batches over the training dataset.
            x = data['x'].cuda()
            y = data['y'].cuda()
            out = model(x)  # Perform a single forward pass.
            pred_accum.append(out.squeeze().flatten() > config['pred_thresh'])
            y_accum.append(y.flatten())
            # embed()
            loss = criterion(out.squeeze(), y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        pred_accum = torch.cat(pred_accum).to(int).detach().cpu()
        y_accum = torch.cat(y_accum).to(int).detach().cpu()
        m = dict(zip(['train_tn', 'train_fp', 'train_fn', 'train_tp'], confusion_matrix(y_accum, pred_accum, normalize='all', labels=(0,1)).ravel()))
        m.update({'train_acc': accuracy_score(y_accum, pred_accum, normalize=True)})
        print(f"TRAIN: tn:{m['train_tn']:.3f} fp:{m['train_fp']:.3f} fn:{m['train_fn']:.3f} tp:{m['train_tp']:.3f} acc:{m['train_acc']:.3f}")
        wandb.log(m)

    def test():
        model.eval()
        pred_accum = []
        y_accum = []
        for data in tqdm(dl_test, desc=f'E{epoch:03d}: TEST  '):  # Iterate in batches over the training/test dataset.
            x = data['x'].cuda()
            y = data['y'].cuda()
            out = model(x)
            pred_accum.append(out.squeeze().flatten() > config['pred_thresh'])
            y_accum.append(y.flatten())
        pred_accum = torch.cat(pred_accum).to(int).detach().cpu()
        y_accum = torch.cat(y_accum).to(int).detach().cpu()
        m = dict(zip(['test_tn', 'test_fp', 'test_fn', 'test_tp'], confusion_matrix(y_accum, pred_accum, normalize='all', labels=(0,1)).ravel()))
        m.update({'test_acc': accuracy_score(y_accum, pred_accum, normalize=True)})
        print(f"TRAIN: tn:{m['test_tn']:.3f} fp:{m['test_fp']:.3f} fn:{m['test_fn']:.3f} tp:{m['test_tp']:.3f} acc:{m['test_acc']:.3f}")
        wandb.log(m)

    for epoch in range(1, config['n_epochs']):
        train()
        test()

if __name__ == '__main__':
    config = cmdline_config()
    main(config)