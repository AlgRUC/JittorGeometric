import jittor as jt
import jittor.nn as nn
import numpy as np
class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((jt.Var(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(jt.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def execute(self, timestamps: jt.Var):
        """
        compute time encodings of time in timestamps
        :param timestamps: Var, shape (batch_size, seq_len)
        :return:
        """
        # Var, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Var, shape (batch_size, seq_len, time_dim)
        output = jt.cos(self.w(timestamps))

        return output

class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def execute(self, input_1: jt.Var, input_2: jt.Var):
        """
        merge and project the inputs
        :param input_1: Var, shape (*, input_dim1)
        :param input_2: Var, shape (*, input_dim2)
        :return:
        """
        # Var, shape (*, input_dim1 + input_dim2)
        x = jt.cat([input_1, input_2], dim=1)
        # Var, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h
    
