import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple, List, Union, Any
import numpy as np
import inspect
import jittor as jt
from jittor import nn
from jittor.nn import Embedding, Linear, ModuleList, Sequential
from jittor_geometric.nn.inits import xavier_normal
from jittor_geometric.typing import OptVar

from jittor_geometric.data import Dataset, download_url, extract_zip
from jittor_geometric.nn.conv import MessagePassing
from jittor_geometric.nn.aggr import SumAggregation


qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}

def softplus(x: jt.Var, beta=jt.array(1.0), threshold=jt.array(20.0)) -> jt.Var:
    """Softplus function.
    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))
    """
    return jt.log(jt.array(1.0) + jt.exp(beta * x)) / beta



def radius_graph(data, r, batch=None, max_num_neighbors=None, loop=False):
    """
    Jittor 实现的 radius_graph。

    Args:
        data (jt.Var): 点的坐标，形状为 [N, D]。
        radius (float): 半径阈值。
        batch (jt.Var, optional): 批次索引，形状为 [N]，用于区分不同的批次。
        max_num_neighbors (int, optional): 每个点的最大邻居数量，默认 None 表示不限制。
        loop (bool): 是否包含自环边，默认 False。

    Returns:
        jt.Var: 边的索引，形状为 [2, E]。
    """
    assert len(data.shape) == 2, "Input data must be 2D, with shape [N, D]"

    N, D = data.shape

    if batch is None:
        batch = jt.zeros(N, dtype=jt.int32)  # 如果未提供 batch，默认为单一批次

    assert len(batch) == N, "Batch size must match the number of points"

    edges = []

    # Step 1: 计算所有点对的欧几里得距离
    dist = jt.sqrt(jt.sum((data.unsqueeze(1) - data.unsqueeze(0)) ** 2, dim=2))

    # Step 2: 找到满足半径条件的点对
    for i in range(N):
        # 当前点所属的批次
        current_batch = batch[i]

        # 获取当前点与同批次其他点的距离
        neighbors = jt.where((dist[i] <= r) & (batch == current_batch))[0].numpy()

        # 排除自环（如果不允许自环）
        if not loop:
            neighbors = neighbors[neighbors != i]

        # 限制邻居数量
        if max_num_neighbors is not None:
            neighbors = neighbors[:max_num_neighbors]

        # 添加边
        for j in neighbors:
            edges.append([i, j])

    # 转换为 Jittor 张量并返回
    edges = jt.array(edges).transpose()
    return edges

def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')

def resolver(
    classes: List[Any],
    class_dict: Dict[str, Any],
    query: Union[Any, str],
    base_cls: Optional[Any],
    base_cls_repr: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> Any:

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ''
    base_cls_repr = normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                return obj
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                return obj
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


def aggr_resolver(query: Union[Any, str], *args, **kwargs):
    import jittor_geometric.nn.aggr as aggr
    if isinstance(query, (list, tuple)):
        return aggr.MultiAggregation(query, *args, **kwargs)

    base_cls = aggr.Aggregation
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    aggr_dict = {
        'add': aggr.SumAggregation,
    }
    return resolver(aggrs, aggr_dict, query, base_cls, None, *args, **kwargs)

class SchNet(nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptVar = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            self.atomic_mass = jt.array(ase.data.atomic_masses)
            # self.register_buffer('atomic_mass', atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        # self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if self.atomref is not None:
            self.atomref = Embedding(100, 1)
            # self.atomref.weight.data.copy_(atomref)
            self.atomref.weight.data = self.atomref

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # self.embedding.reset_parameters()
        nn.init.gauss_(self.embedding.weight)
        for interaction in self.interactions:
            interaction.reset_parameters()
        xavier_normal(self.lin1.weight)
        # self.lin1.bias.data.fill_(0)
        self.lin1.bias.data[:] = 0
        xavier_normal(self.lin2.weight)
        # self.lin2.bias.data.fill_(0)
        self.lin2.bias.data[:] = 0
        if self.atomref is not None:
            # self.atomref.weight.data.copy_(self.initial_atomref)
            self.atomref.weight.data = self.atomref

    @staticmethod
    def from_qm9_pretrained(
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['SchNet', Dataset, Dataset, Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`SchNet` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`.
        """
        import ase
        import schnetpack as spk  # noqa

        assert target >= 0 and target <= 12
        is_dipole = target == 0

        units = [1] * 12
        units[0] = ase.units.Debye
        units[1] = ase.units.Bohr**3
        units[5] = ase.units.Bohr**2

        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

        split = np.load(path)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']

        # Filter the splits to only contain characterized molecules.
        idx = dataset.data.idx
        assoc = idx.new_empty(idx.max().item() + 1)
        assoc[idx] = jt.arange(idx.size(0))

        train_idx = assoc[train_idx[np.isin(train_idx, idx)]]
        val_idx = assoc[val_idx[np.isin(val_idx, idx)]]
        test_idx = assoc[test_idx[np.isin(test_idx, idx)]]

        path = osp.join(root, 'trained_schnet_models', name, 'best_model.pkl')
        import pickle
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     state = torch.load(path, map_location='cpu')
            # state=jt.load(path)

        net = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            dipole=is_dipole,
            atomref=dataset.atomref(target),
        )

        with open(path, 'rb') as f : 
            state = pickle.load(f)
        net.embedding.weight = state['embedding_weight']
        for int2 in net.interactions : 
            int2.mlp[0].weight = state['int2']['mlp0_weight']
            int2.mlp[0].bias = state['int2']['mlp0_bias']
            int2.mlp[2].weight = state['int2']['mlp2_weight']
            int2.mlp[2].bias = state['int2']['mlp2_bias']
            int2.lin.weight = state['int2']['lin_weight']
            int2.lin.bias = state['int2']['lin_bias']

            int2.conv.lin1.weight = state['int2']['conv_lin1_weight']
            int2.conv.lin2.weight = state['int2']['conv_lin2_weight']
            int2.conv.lin2.bias = state['int2']['conv_lin2_bias']
            
        net.lin1.weight = state['lin1_weight']
        net.lin1.bias = state['lin1_bias']
        net.lin2.weight = state['lin2_weight']
        net.lin2.bias = state['lin2_bias']
        
        mean = state['is_mean']
        net.readout = aggr_resolver('mean' if mean is True else 'add')
        dipole = state['dipole']
        net.dipole = dipole
        
        net.mean = state['_mean']
        net.std = state['_std']
        net.atomref = None
        # if state['atomref'] is not None : 
        #     net.atomref.weight = state['atomref_weight']
        # else : net.atomref = None
        
        net.scale = 1.0 / units[target]
        
        return net, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def execute(self, z: jt.Var, pos: jt.Var,
                batch: OptVar = None) -> jt.Var:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = jt.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        print(jt.isnan(h).any())
        h = self.lin1(h)
        print(jt.isnan(h).any())
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(x=mass, batch=batch, dim=0)
            c = self.sum_aggr(x=mass * pos, index=batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))
        
        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean
        
        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)
        
        out = self.readout(h, batch, dim=0)
        

        if self.dipole:
            out = jt.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class RadiusInteractionGraph(nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def execute(self, pos: jt.Var, batch: jt.Var) -> Tuple[jt.Var, jt.Var]:
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal(self.mlp[0].weight)
        # self.mlp[0].bias.data.fill_(0)
        self.mlp[0].bias.data[:] = 0
        xavier_normal(self.mlp[2].weight)
        # self.mlp[2].bias.data.fill_(0)
        self.mlp[2].bias.data[:] = 0
        self.conv.reset_parameters()
        xavier_normal(self.lin.weight)
        # self.lin.bias.data.fill_(0)
        self.lin.bias.data[:] = 0

    def execute(self, x: jt.Var, edge_index: jt.Var, edge_weight: jt.Var,
                edge_attr: jt.Var) -> jt.Var:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal(self.lin1.weight)
        xavier_normal(self.lin2.weight)
        # self.lin2.bias.data.fill_(0)
        self.lin2.bias.data[:] = 0

    def execute(self, x: jt.Var, edge_index: jt.Var, edge_weight: jt.Var,
                edge_attr: jt.Var) -> jt.Var:
        C = 0.5 * (jt.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: jt.Var, W: jt.Var) -> jt.Var:
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        self.offset = jt.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (self.offset[1] - self.offset[0]).item()**2
        # self.register_buffer('offset', self.offset)

    def execute(self, dist: jt.Var) -> jt.Var:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return jt.exp(self.coeff * jt.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = jt.log(jt.array(2.0)).item()

    def execute(self, x: jt.Var) -> jt.Var:
        # return F.softplus(x) - self.shift
        return softplus(x) - self.shift
