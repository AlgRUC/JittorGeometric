from typing import Optional
import jittor as jt
from jittor_geometric.utils import scatter

def global_add_pool(x: jt.Var, batch: Optional[jt.Var],
                    size: Optional[int] = None) -> jt.Var:
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~jittor_geometric.nn.aggr.SumAggregation` module.

    Args:
        x (jt.Var): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (jt.Var, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if x.ndim == 1 else -2

    if batch is None:
        return x.sum(dim=dim, keepdims=x.ndim <= 2)
    return scatter(x, batch, dim=dim, dim_size=size, reduce='sum')