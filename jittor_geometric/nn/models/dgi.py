"""
Deep Graph Infomax in Jittor
References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import math
import jittor as jt
import jittor.nn as nn
from jittor_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
            self, data, in_feats, n_hidden, n_classes, n_layers, dropout
    ):
        super(GCN, self).__init__()
        self.csc = data.csc
        self.csr = data.csr
        self.layers = nn.ModuleList()

        self.layers.append(
            GCNConv(in_feats, n_hidden)
        )

        for i in range(n_layers - 1):
            self.layers.append(
                GCNConv(n_hidden, n_hidden)
            )

        self.layers.append(
            GCNConv(n_hidden, n_classes)
        )

        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

        self.act_fn = nn.PReLU()

    def execute(self, features):
        h = features

        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, self.csc, self.csr)
            h = self.act_fn(h)

        return h


class Encoder(nn.Module):
    def __init__(self, data, in_feats, n_hidden, n_layers, dropout):
        super(Encoder, self).__init__()
        self.conv = GCN(
            data, in_feats, n_hidden, n_hidden, n_layers, dropout
        )

    def execute(self, features, corrupt=False):
        if corrupt:
            perm = jt.randperm(features.shape[0])  # random shuffle features
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(jt.empty((n_hidden, n_hidden)))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        tensor.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def execute(self, features, summary):
        features = jt.matmul(features, jt.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    r"""DGI model from the `"Deep Graph Infomax"
    <https://arxiv.org/abs/1809.10341>`_ paper

    Parameters
    -----------
    data: jittor_geometric.data.Data
        Input jittor_geometric data.
    in_feats: int
        Input feature size.
    n_hidden: int
        Hidden feature size.
    n_layers: int
        Number of the GNN encoder layers.
    dropout: float
        Dropout ratio.
    """

    def __init__(self, data, in_feats, n_hidden, n_layers, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(
            data, in_feats, n_hidden, n_layers, dropout
        )
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def execute(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)

        summary = jt.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, jt.ones_like(positive))
        l2 = self.loss(negative, jt.zeros_like(negative))

        return l1 + l2