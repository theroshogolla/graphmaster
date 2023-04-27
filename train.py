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
CONFIG = vars(parser.parse_args())
node_fn = getattr(my_data, CONFIG['node_fn'])
if CONFIG['node_fn'] == 'piece_color_type':
    attrs_per_node = 10
elif CONFIG['node_fn'] == 'piece_color':
    attrs_per_node = 3
elif CONFIG['node_fn'] == 'piece_type':
    attrs_per_node = 7
else: raise ValueError(f"node_fn:{CONFIG['node_fn']} not implemented")
edge_fn = getattr(my_data, CONFIG['edge_fn'])
engine = None
if CONFIG['engine'] == 'stockfish':
    engine = my_pred.init_engine('stockfish')
elif CONFIG['engine'] == 'lc0':
    engine = my_pred.init_engine('lc0')
elif CONFIG['engine'] == 'actual_result':
    pass
else: raise ValueError(f"target:{CONFIG['engine']} not implemented")

# if CONFIG['edge_fn'] == 'mobility':
#     attrs_per_edge = 10
# if CONFIG['edge_fn'] == 'support':
#     attrs_per_edge = 3
# if CONFIG['edge_fn'] == 'position':
#     attrs_per_edge = 7

if CONFIG['wandb']:
    wandb.init(project='chess', config=CONFIG)
else:
    #init dummy wandb
    wandb.init(mode="disabled")
print(CONFIG)
print(f'Using node_fn={node_fn.__name__}(); edge_fn={edge_fn.__name__}()')

### DATA
df = my_data.parse(n_games=CONFIG['n_games'], skip_draws=True)
df_train, df_test = train_test_split(df, test_size=CONFIG['test_split'])

if CONFIG['net'] == 'gcn':
    ds_train = my_data.GraphDataset(df_train, node_fn=node_fn, edge_fn=edge_fn, engine=engine)
    ds_test = my_data.GraphDataset(df_test, node_fn=node_fn, edge_fn=edge_fn, engine=engine)
    dl_train = torch_geometric.loader.DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True)
    dl_test = torch_geometric.loader.DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False)
    model = my_model.GCN(CONFIG, n_node_features=attrs_per_node, n_cls=1, sigmoid=True)
elif CONFIG['net'] == 'fc':
    ds_train = my_data.TabularDataset(df_train, node_fn=node_fn, engine=engine)
    ds_test = my_data.TabularDataset(df_test, node_fn=node_fn, engine=engine)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False)
    model = my_model.DumbNet(CONFIG, n_features=64 * attrs_per_node, n_cls=1, sigmoid=True)

print(f"train {len(ds_train)} samples / {CONFIG['batch_size']} batches = {-(-len(ds_train) // CONFIG['batch_size']):0.0f} iters per epoch")

model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
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
        pred_accum.append(out.squeeze().flatten() > CONFIG['pred_thresh'])
        y_accum.append(y.flatten())
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
        pred_accum.append(out.squeeze().flatten() > CONFIG['pred_thresh'])
        y_accum.append(y.flatten())
    pred_accum = torch.cat(pred_accum).to(int).detach().cpu()
    y_accum = torch.cat(y_accum).to(int).detach().cpu()
    m = dict(zip(['test_tn', 'test_fp', 'test_fn', 'test_tp'], confusion_matrix(y_accum, pred_accum, normalize='all', labels=(0,1)).ravel()))
    m.update({'test_acc': accuracy_score(y_accum, pred_accum, normalize=True)})
    print(f"TRAIN: tn:{m['test_tn']:.3f} fp:{m['test_fp']:.3f} fn:{m['test_fn']:.3f} tp:{m['test_tp']:.3f} acc:{m['test_acc']:.3f}")
    wandb.log(m)

for epoch in range(1, CONFIG['n_epochs']):
    train()
    test()
    # print(f'E{epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')