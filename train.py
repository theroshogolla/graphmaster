from IPython import embed
import torch_geometric.loader
from tqdm import tqdm
import torch
import torch.utils.data
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import data as my_data
import model as my_model

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
CONFIG = vars(parser.parse_args())
print(CONFIG)

### DATA
df = my_data.parse(nrows=CONFIG['n_games'], skip_draws=True)
df_train, df_test = train_test_split(df, test_size=CONFIG['test_split'])

if CONFIG['net'] == 'gcn':
    ds_train = my_data.GraphDataset(df_train, net=CONFIG['net'])
    ds_test = my_data.GraphDataset(df_test, net=CONFIG['net'])
    dl_train = torch_geometric.loader.DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True)
    dl_test = torch_geometric.loader.DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False)
    model = my_model.GCN(CONFIG, n_node_features=10, n_cls=1, sigmoid=True)
elif CONFIG['net'] == 'fc':
    ds_train = my_data.TabularDataset(df_train, net=CONFIG['net'])
    ds_test = my_data.TabularDataset(df_test, net=CONFIG['net'])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False)
    model = my_model.DumbNet(CONFIG, n_features=640, n_cls=1, sigmoid=True)

print(f"train {len(ds_train)} samples / {CONFIG['batch_size']} batches = {-(-len(ds_train) // CONFIG['batch_size']):0.0f} iters per epoch")

model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
criterion = torch.nn.BCELoss()
print(model)

def train():
    model.train()
    pred = []
    y = []
    for data in tqdm(dl_train, desc=f'E{epoch:03d}: TRAIN'):  # Iterate in batches over the training dataset.
        data = data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        pred.append(out.squeeze().flatten() > CONFIG['pred_thresh'])
        y.append(data.y.flatten())
        loss = criterion(out.squeeze(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    pred = torch.cat(pred).to(int).detach().cpu()
    y = torch.cat(y).to(int).detach().cpu()
    tn, fp, fn, tp = confusion_matrix(y, pred, normalize='all').ravel()
    acc = accuracy_score(y, pred, normalize=True)
    print(f'TRAIN: tn:{tn:.3f} fp:{fp:.3f} fn:{fn:.3f} tp:{tp:.3f} acc:{acc:.3f}')

def test():
    model.eval()
    pred = []
    y = []
    for data in tqdm(dl_test, desc=f'E{epoch:03d}: TEST  '):  # Iterate in batches over the training/test dataset.
        data = data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)
        pred.append(out.squeeze().flatten() > CONFIG['pred_thresh'])
        y.append(data.y.flatten())
    pred = torch.cat(pred).to(int).detach().cpu()
    y = torch.cat(y).to(int).detach().cpu()
    tn, fp, fn, tp = confusion_matrix(y, pred, normalize='all').ravel()
    acc = accuracy_score(y, pred, normalize=True)
    print(f'TEST:  tn:{tn:.3f} fp:{fp:.3f} fn:{fn:.3f} tp:{tp:.3f} acc:{acc:.3f}')

for epoch in range(1, CONFIG['n_epochs']):
    train()
    test()
    # print(f'E{epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')