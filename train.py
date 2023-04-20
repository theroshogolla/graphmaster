from IPython import embed

import torch_geometric as tg
from tqdm import tqdm
from torch_geometric.utils.convert import from_networkx, to_networkx
import networkx as nx
import random
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
import chess
from sklearn import metrics

import data as my_data
import graph as my_graph

tqdm.pandas()

CONFIG = {
    'n_games': 100_000,
    'test_split': 10,
    'batch_size': 256,
    'hidden_channels': 256,
    'lr': 1e-3,
    'n_epochs': 1_000,
}

### use list -> dataloader
print('parsing csv')
df = my_data.parse(nrows=CONFIG['n_games'])
print('df -> games')
### TODO: lazy loader
games = df.progress_apply(lambda row: my_data.get_game(row), axis=1).to_list()
random.shuffle(games)
train_games = games[len(games) // (CONFIG['test_split'] + 1):]
test_games = games[:len(games) // (CONFIG['test_split'] + 1)]

print('train games -> boards')
train_graphs = []
for game in tqdm(train_games):
    for item in my_data.get_boards(game):
        board = item['board']
        win = int(item['win'])
        graph = from_networkx(my_graph.chess_nx(board))
        graph.x = graph.piece_value.unsqueeze(-1)
        graph.y = win
        train_graphs.append(graph)

print('test games -> boards')
test_graphs = []
for game in tqdm(test_games):
    for item in my_data.get_boards(game):
        board = item['board']
        win = int(item['win'])
        graph = from_networkx(my_graph.chess_nx(board))
        graph.x = graph.piece_value.unsqueeze(-1)
        graph.y = win
        test_graphs.append(graph)

print(f'Number of graphs: {len(train_graphs + test_graphs)}')

data = train_graphs[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print('=============================================================')
print(f'Number of training graphs: {len(train_graphs)}')
print(f'Number of test graphs: {len(test_graphs)}')

train_loader = tg.loader.DataLoader(train_graphs, batch_size=CONFIG['batch_size'], shuffle=True)
test_loader = tg.loader.DataLoader(test_graphs, batch_size=CONFIG['batch_size'], shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
    if step == 2: break

class GCN(torch.nn.Module):
    def __init__(self, n_node_features=1, hidden_channels=CONFIG['hidden_channels'], n_cls=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, n_cls)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN().to('cuda')
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to('cuda')
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

for epoch in range(1, CONFIG['n_epochs']):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')