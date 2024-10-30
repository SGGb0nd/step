from dgl.nn.pytorch import GATConv, GraphConv
from torch import nn


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, n_layers=3, n_heads=4, **kwargs):
        super(GAT, self).__init__()
        self.conv_hs = nn.ModuleList(
            [GATConv(h_feats, h_feats, num_heads=n_heads)
             for _ in range(n_layers)]
        )
        self.out_ln = nn.Linear(n_heads * h_feats, h_feats)
        self.norm = nn.LayerNorm(h_feats, elementwise_affine=True)

    def forward(self, g, h):
        batch_size = h.shape[0]
        for layer in self.conv_hs[:-1]:
            h = layer(g, h)
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
        # self.conv1 = GraphConv(in_feats, h_feats)
        self.conv_hs = nn.ModuleList(
            [GraphConv(h_feats, h_feats) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(h_feats, elementwise_affine=True)
        self.with_edge = with_edge

    def forward(self, g, h):
        for _, layer in enumerate(self.conv_hs):
            h = layer(g, h)
        h = self.norm(h)
        return h

    def batch_forward(self, blocks, h):
        for i, layer in enumerate(self.conv_hs):
            h = layer(blocks[i], h)
        h = self.norm(h)
        return h
        for _, layer in enumerate(self.conv_hs):
            h = layer(g, h)
        h = self.conv_out(g, h)
        h = self.norm(h)
        return h
