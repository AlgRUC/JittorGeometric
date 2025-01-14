import jittor as jt
import jittor.nn as nn
import numpy as np

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
    
