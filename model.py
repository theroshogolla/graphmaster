from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch import nn
import torch.nn.functional as F
import monai

class GCN(torch.nn.Module):
    def __init__(self, config, n_node_features=1, n_cls=2):
        n_hidden_ch = config['n_hidden_ch']
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_node_features, n_hidden_ch)
        self.conv2 = GCNConv(n_hidden_ch, n_hidden_ch)
        self.conv3 = GCNConv(n_hidden_ch, n_hidden_ch)
        self.lin = nn.Linear(n_hidden_ch, n_cls)

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
    
clas UNET