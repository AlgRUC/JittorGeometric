import jittor as jt
from jittor import nn
from jittor_geometric.nn.aggr import Aggregation
from typing import Optional


class SumAggregation(Aggregation):
    r"""An aggregation operator that sums up features across a set of elements.

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def execute(self, x: jt.Var, index: Optional[jt.Var] = None,
                ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> jt.Var:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')


class MeanAggregation(Aggregation):
    r"""An aggregation operator that averages features across a set of elements.

    .. math::
        \mathrm{mean}(\mathcal{X}) = \frac{1}{|\mathcal{X}|}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """
    def execute(self, x: jt.Var, index: Optional[jt.Var] = None,
                ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> jt.Var:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='mean')


class MaxAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise maximum across a set of elements.

    .. math::
        \mathrm{max}(\mathcal{X}) = \max_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def execute(self, x: jt.Var, index: Optional[jt.Var] = None,
                ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> jt.Var:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='max')


class MinAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise minimum across a set of elements.

    .. math::
        \mathrm{min}(\mathcal{X}) = \min_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def execute(self, x: jt.Var, index: Optional[jt.Var] = None,
                ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> jt.Var:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='min')

