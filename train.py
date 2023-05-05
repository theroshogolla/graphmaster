from IPython import embed
import torch_geometric.loader
import torch_geometric.nn.models as gnn
import torch_geometric.profile
from tqdm import tqdm
import torch
import torch.utils.data
import argparse
import wandb
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

import data as my_data
import model as my_model
import pred as my_pred

tqdm.pandas()

def cmdline_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_games', default=10_000, type=int)
    parser.add_argument('--test_split', default=0.2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--net', default='gcn', type=str)
    parser.add_argument('--loss', default='bce', type=str)

    parser.add_argument('--n_hidden_ch', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-e', '--n_epochs', default=2_000, type=int)
    parser.add_argument('--pred_thresh', default=0.5, type=float)
    parser.add_argument('--dropout_rate', default=0.25, type=float)
    parser.add_argument('--node_fn', default='piece_color_type', type=str)
    parser.add_argument('--edge_fn', default='position', type=str)
    parser.add_argument('--target', default='actual', type=str)
    parser.add_argument('--pred_time', default=0.01, type=float)
    parser.add_argument('--wandb', action='store_true')
    # parser.add_argument('--gpu', default='0', type=str)
    config = vars(parser.parse_args())
    return config

def main(config, internal_wandb=True):
    # os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    if config['node_fn'] == 'piece_color_type':
        attrs_per_node = 10
    elif config['node_fn'] == 'piece_color':
        attrs_per_node = 3
    elif config['node_fn'] == 'piece_type':
        attrs_per_node = 7
    else: raise ValueError(f"node_fn:{config['node_fn']} not implemented")
    engine = None
    if config['target'] == 'stockfish':
        engine = my_pred.init_engine('stockfish')
    elif config['target'] == 'lc0':
        engine = my_pred.init_engine('lc0')
    elif config['target'] == 'actual':
        pass
    else: raise ValueError(f"target:{config['target']} not implemented")

    if internal_wandb:
        if config['wandb']:
            wandb.init(project='chess', config=config)
        else:
            #init dummy wandb
            wandb.init(mode="disabled")
    print(config)
    print(f"Using node_fn={config['node_fn']}(); edge_fn={config['edge_fn']}()")

    ### DATA
    # df = my_data.parse(n_games=config['n_games'], skip_draws=True)
    df = my_data.parse_cleaned(n_games=config['n_games'])
    df_train, df_test = train_test_split(df, test_size=config['test_split'], stratify=df['Result'])

    if config['net'] == 'fc':
        ds_train = my_data.TabularDataset(df=df_train, config=config, engine=engine)
        ds_test = my_data.TabularDataset(df=df_test, config=config, engine=engine)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        model = my_model.DumbNet(config, n_features=64 * attrs_per_node, n_cls=1, sigmoid=True)
    else:
        ds_train = my_data.GraphDataset(df=df_train, config=config, engine=engine)
        ds_test = my_data.GraphDataset(df=df_test, config=config, engine=engine)
        dl_train = torch_geometric.loader.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        dl_test = torch_geometric.loader.DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        if config['net'] == 'mygcn':
            model = my_model.GCN(config, n_node_features=attrs_per_node, n_cls=1, sigmoid=True)
        else:
            if config['net'] == 'gcn':
                core = gnn.GCN(in_channels=attrs_per_node, hidden_channels=config['n_hidden_ch'], num_layers=3, out_channels=1)
            elif config['net'] == 'graphsage':
                core = gnn.GraphSAGE(in_channels=attrs_per_node, hidden_channels=config['n_hidden_ch'], num_layers=3, out_channels=1)
            elif config['net'] == 'gat':
                core = gnn.GAT(in_channels=attrs_per_node, hidden_channels=config['n_hidden_ch'], num_layers=3, out_channels=1)
            elif config['net'] == 'edgecnn':
                core = gnn.EdgeCNN(in_channels=attrs_per_node, hidden_channels=config['n_hidden_ch'], num_layers=3, out_channels=1)
            elif config['net'] == 'gin':
                core = gnn.GIN(in_channels=attrs_per_node, hidden_channels=config['n_hidden_ch'], num_layers=3, out_channels=1)
            model = my_model.GraphClf(core, sigmoid=True)
    print(f'network parameter count: {torch_geometric.profile.count_parameters(model)}')    
    print(f"train {len(ds_train)} samples / {config['batch_size']} batches = {-(-len(ds_train) // config['batch_size']):0.0f} iters per epoch")


    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if config['loss'] == 'bce':
        criterion = torch.nn.BCELoss()
    elif config['loss'] == 'focal':
        criterion = my_model.focal_loss
    print(model)

    # this has been moved to util; remove
    def metrics(y, pred, pred_thresh, prefix='train_'):
        pred_bool = (pred > pred_thresh).float()
        ret = dict(zip(['tn', 'fp', 'fn', 'tp'],
                       confusion_matrix(y, pred_bool, normalize='all', labels=(0,1)).ravel()))
        ret.update({'acc': accuracy_score(y, pred_bool, normalize=True)})
        ret.update({'roc_auc': roc_auc_score(y, pred)})
        pred_neg = 1 - pred
        pred_comb = torch.concat((pred.unsqueeze(-1), pred_neg.unsqueeze(-1)), axis=-1)
        # hotfix: break after one iter of wandb.plot.roc_curve/pr_curve:indices_to_plot to remove pred_neg
        # also remove warning for n_sample > 10k
        ret.update({'roc_curve': wandb.plot.roc_curve(y, pred_comb)})
        ret.update({'pr_curve': wandb.plot.pr_curve(y, pred_comb)})
        # table = wandb.Table(data=np.array(roc_curve(y, pred)).T)
        # ret.update({'roc_curve': wandb.plot.line(table, x='False Positive Rate', y='True Positive Rate')})
        wandb.log({f'{prefix}{k}':v for k,v in ret.items()})

    def train():
        model.train()
        pred_accum = []
        y_accum = []
        for data in tqdm(dl_train, desc=f'E{epoch:03d}: TRAIN', mininterval=10, maxinterval=60):  # Iterate in batches over the training dataset.
            x = data['x'].cuda()
            y = data['y'].cuda()
            out = model(x)  # Perform a single forward pass.
            pred_accum.append(out.squeeze().flatten())
            y_accum.append(y.flatten())
            loss = criterion(out.squeeze(), y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        pred_accum = torch.cat(pred_accum).detach().cpu()
        y_accum = torch.cat(y_accum).to(int).detach().cpu()
        return y_accum, pred_accum
    
    @torch.no_grad()
    def test():
        model.eval()
        pred_accum = []
        y_accum = []
        for data in tqdm(dl_test, desc=f'E{epoch:03d}: TEST  ', mininterval=10, maxinterval=60):  # Iterate in batches over the training/test dataset.
            x = data['x'].cuda()
            y = data['y'].cuda()
            out = model(x)
            pred_accum.append(out.squeeze().flatten())
            y_accum.append(y.flatten())
        pred_accum = torch.cat(pred_accum).detach().cpu()
        y_accum = torch.cat(y_accum).to(int).detach().cpu()
        return y_accum, pred_accum

    for epoch in range(1, config['n_epochs']):
        y, pred = train()
        metrics(y, pred, config['pred_thresh'], prefix='train_')
        y, pred = test()
        metrics(y, pred, config['pred_thresh'], prefix='test_')

if __name__ == '__main__':
    config = cmdline_config()
    main(config)