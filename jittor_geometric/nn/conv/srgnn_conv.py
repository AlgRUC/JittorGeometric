import jittor as jt
from jittor import nn
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.ops import SpmmCsr
from jittor_geometric.ops import cootocsr
import numpy as np
class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = jt.nn.Linear(dim, dim)

    def execute(self, x, edge_index):
        x = self.lin(x)
        edge_weight = jt.ones(edge_index.shape[1])
        csr = cootocsr(edge_index, edge_weight, x.shape[0])
        out = SpmmCsr(x,csr)  
        return out


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def execute(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = jt.flip(edge_index)
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = jt.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = jt.sigmoid(i_r + h_r)
        input_gate = jt.sigmoid(i_i + h_i)
        new_gate = jt.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight=np.random.uniform(-stdv, stdv, weight.shape)