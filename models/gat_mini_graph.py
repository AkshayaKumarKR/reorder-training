from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn

class GAT(torch.nn.Module):

    def __init__(self, input_dim, n_classes, n_layers=2, hidden_dim=256, heads=1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.layers = torch.nn.ModuleList()

        if n_layers == 1:
            self.layers.append(GATConv(input_dim, n_classes, heads=heads, concat=False))
        else:
            self.layers.append(GATConv(input_dim, hidden_dim, heads=heads))
            for _ in range(1, self.n_layers - 1):
                self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))

            self.layers.append(GATConv(hidden_dim * heads, n_classes, heads=heads, concat=False))
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != self.n_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x
