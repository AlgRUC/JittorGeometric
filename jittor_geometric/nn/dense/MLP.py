import jittor as jt
from jittor import nn


class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int, act_type='PReLU', dropout: float = 0.1, use_act: bool = True, skip_connection: bool = True, pinit: float = 0.15):
        super().__init__()
        self.num_layers = num_layers
        self.lins = nn.ModuleList()
        if num_layers > 1:
            for _ in range(num_layers-1):
                self.lins.append(nn.Linear(input_dim, hidden_dim))
                input_dim = hidden_dim
            self.lins.append(nn.Linear(hidden_dim, output_dim))
            self.dropout = nn.Dropout(dropout)
            self.act = get_activation(act_type, pinit)
            self.use_act = use_act
            self.skip_connection = skip_connection
            if skip_connection and input_dim != output_dim:
                self.linear_b4_skip = nn.Linear(input_dim, output_dim)
        else:
            self.lins.append(nn.Linear(input_dim, output_dim))

    def execute(self, input: jt.Var):
        x = input
        if self.num_layers > 1:
            for i in range(len(self.lins)-1):
                x = self.lins[i](x)
                if self.use_act:
                    x = self.act(x)
                x = self.dropout(x)
            x = self.lins[-1](x)
            if self.skip_connection:
                if self.linear_b4_skip is not None:
                    new_input = self.linear_b4_skip(input)
                else:
                    new_input = input
                x = x + new_input
        else:
            x = self.lins[0](x)
        return x
    
def get_activation(act_type, pinit=0.15):
    if act_type == 'PReLU':
        return jt.nn.PReLU(init=pinit)
    elif act_type == 'ReLU':
        return jt.nn.ReLU()
    elif act_type == 'LeakyReLU':
        return jt.nn.LeakyReLU()
    elif act_type == 'Sigmoid':
        return jt.nn.Sigmoid()
    elif act_type == 'Tanh':
        return jt.nn.Tanh()
    else:
        raise NotImplementedError