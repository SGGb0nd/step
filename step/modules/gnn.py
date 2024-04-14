from dgl.nn.pytorch import GATConv, GraphConv
from torch import nn


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, n_layers=3, n_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads=n_heads)
        self.conv_hs = nn.ModuleList(
            [GATConv(h_feats, h_feats, num_heads=n_heads)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(h_feats, elementwise_affine=False)
        self.out_ln = nn.Linear(n_heads * h_feats, h_feats)

    def forward(self, g, in_feat):
        batch_size = in_feat.shape[0]
        h = self.conv1(g, in_feat).mean(1)
        for layer in self.conv_hs[:-1]:
            h = layer(g, h).mean(1)
        h = self.conv_hs[-1](g, h)
        h = self.out_ln(h.view(batch_size, -1))
        h = self.norm(h)
        return h


class GCN(nn.Module):
    """
    GCN is a graph convolutional network.

    Attributes:
        in_feats (int): The input feature dimension.
        h_feats (int): The hidden feature dimension.
        n_layers (int): The number of layers.
        with_edge (bool): Whether to use edge features.
    """

    def __init__(self, in_feats, h_feats, n_layers=3, with_edge=True):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv_hs = nn.ModuleList(
            [GraphConv(h_feats, h_feats) for _ in range(n_layers - 2)]
        )
        self.conv_out = GraphConv(h_feats, h_feats)
        self.n_layers = n_layers + 1
        self.norm = nn.LayerNorm(h_feats, elementwise_affine=True)
        self.with_edge = with_edge

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        for _, layer in enumerate(self.conv_hs):
            h = layer(g, h)
        h = self.conv_out(g, h)
        h = self.norm(h)
        return h
