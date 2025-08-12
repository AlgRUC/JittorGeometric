import jittor as jt
from jittor import nn
from jittor_geometric.nn import GCNConv


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def execute(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = jt.concat((sc_1, sc_2, sc_3, sc_4))

        return logits


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.conv = GCNConv(
            in_dim, out_dim, bias=True, spmm=True
        )
        self.act_fn = nn.PReLU()

    def execute(self, x, csc, csr):
        x = self.conv(x, csc, csr)
        return self.act_fn(x)


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def execute(self, x, batch=None):
        if batch is None:
            return x.mean(dim=0, keepdims=True)
        else:
            unique_batches = jt.unique(batch)[0]
            pooled = []
            for b in unique_batches:
                mask = (batch == b).nonzero().squeeze(-1)
                pooled.append(x[mask].mean(dim=0))
            return jt.stack(pooled)


class MVGRL(nn.Module):
    r"""MVGRL model from the `"Contrastive Multi-View Representation Learning on Graphs"
    <https://arxiv.org/abs/2006.05582>`_ paper

    Parameters
    -----------
    in_dim: int
        Input feature size.
    out_dim: int
        Output feature size.
    """

    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()

        self.encoder1 = GCN(
            in_dim, out_dim
        )
        self.encoder2 = GCN(
            in_dim, out_dim
        )

        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, csc_o, csr_o, feat, csc_d, csr_d):
        h1 = self.encoder1(feat, csc_o, csr_o)
        h2 = self.encoder2(feat, csc_d, csr_d)

        return (h1 + h2).detach()

    def execute(self, csc_o, csr_o, feat, csc_d, csr_d, shuf_feat):
        h1 = self.encoder1(feat, csc_o, csr_o)
        h2 = self.encoder2(feat, csc_d, csr_d)

        h3 = self.encoder1(shuf_feat, csc_o, csr_o)
        h4 = self.encoder2(shuf_feat, csc_d, csr_d)

        c1 = self.act_fn(self.pooling(h1))
        c2 = self.act_fn(self.pooling(h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out
