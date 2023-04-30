from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch import nn
import torch.nn.functional as F

# from torchvision.ops.sigmoid_focal_loss
def focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
    alpha: float = .5, gamma: float = 2) -> torch.Tensor:
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()

class GCN(torch.nn.Module):
    def __init__(self, config, n_node_features=1, n_cls=1, sigmoid=True):
        self.config = config
        n_hidden_ch = config['n_hidden_ch']
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_node_features, n_hidden_ch)
        self.conv2 = GCNConv(n_hidden_ch, n_hidden_ch)
        self.conv3 = GCNConv(n_hidden_ch, n_hidden_ch)
        self.lin = nn.Linear(n_hidden_ch, n_cls)
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.config['dropout_rate'], training=self.training)
        x = self.lin(x)
        
        x = self.sigmoid(x)
        return x
    
class DumbNet(torch.nn.Module):
    def __init__(self, config, n_features=640, n_cls=1, sigmoid=True):
        self.config = config
        n_hidden_ch = config['n_hidden_ch']
        super(DumbNet, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_ch)
        self.fc2 = nn.Linear(n_hidden_ch, n_hidden_ch)
        self.fc3 = nn.Linear(n_hidden_ch, n_hidden_ch)
        self.lin = nn.Linear(n_hidden_ch, n_cls)
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        x = x.relu()
        x = F.dropout(x, p=self.config['dropout_rate'], training=self.training)
        x = self.lin(x)
        x = self.sigmoid(x)
        return x
    
class GraphClf(torch.nn.Module):
    def __init__(self, core, sigmoid=True):
        super(GraphClf, self).__init__()
        self.core = core
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.core(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        x = self.sigmoid(x)
        return x