import copy
import jittor as jt
from jittor import nn
from typing import Any, Dict, List, Optional, Union

class MultiAggregation(nn.Module):
    r"""Performs aggregations with one or more aggregators and combines
    aggregated results, as described in the `"Principal Neighbourhood
    Aggregation for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ and
    `"Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions"
    <https://arxiv.org/abs/2104.01481>`_ papers.

    Args:
        aggrs (list): The list of aggregation schemes to use.
        aggrs_kwargs (dict, optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`).
        mode (str, optional): The combine mode to use for combining
            aggregated results from multiple aggregations. (default: :obj:`"cat"`).
        mode_kwargs (dict, optional): Arguments passed for the combine
            :obj:`mode`. (default: :obj:`None`).
    """

    def __init__(
        self,
        aggrs: List[Union[str, nn.Module]],
        aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
        mode: Optional[str] = 'cat',
        mode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"be a list or tuple (got '{type(aggrs)}').")

        if len(aggrs) == 0:
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"not be empty.")

        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)
        elif len(aggrs) != len(aggrs_kwargs):
            raise ValueError(f"'aggrs_kwargs' with invalid length passed to "
                             f"'{self.__class__.__name__}' "
                             f"(got '{len(aggrs_kwargs)}', "
                             f"expected '{len(aggrs)}'). Ensure that both "
                             f"'aggrs' and 'aggrs_kwargs' are consistent.")

        # Aggregator resolver should be replaced with Jittor aggregation functions
        self.aggrs = nn.ModuleList([self._resolve_aggr(aggr, **aggr_kwargs)
                                    for aggr, aggr_kwargs in zip(aggrs, aggrs_kwargs)])

        self.mode = mode
        mode_kwargs = copy.copy(mode_kwargs) or {}

        self.in_channels = mode_kwargs.pop('in_channels', None)
        self.out_channels = mode_kwargs.pop('out_channels', None)

        if mode == 'proj' or mode == 'attn':
            if len(aggrs) == 1:
                raise ValueError("Multiple aggregations are required for "
                                 "'proj' or 'attn' combine mode.")

            if (self.in_channels and self.out_channels) is None:
                raise ValueError(
                    f"Combine mode '{mode}' must have `in_channels` "
                    f"and `out_channels` specified.")

            if isinstance(self.in_channels, int):
                self.in_channels = [self.in_channels] * len(aggrs)

            if mode == 'proj':
                self.lin = nn.Linear(
                    sum(self.in_channels),
                    self.out_channels,
                    **mode_kwargs,
                )

            elif mode == 'attn':
                # Attention implementation could be more manual in Jittor
                channels = {str(k): v for k, v, in enumerate(self.in_channels)}
                self.lin_heads = nn.ModuleDict({
                    str(k): nn.Linear(v, self.out_channels) for k, v in channels.items()
                })
                num_heads = mode_kwargs.pop('num_heads', 1)
                self.multihead_attn = nn.MultiheadAttention(
                    self.out_channels, num_heads,
                    **mode_kwargs
                )

        # Dense combination modes are similar to PyTorch
        self.dense_combine = self._get_dense_combine(mode)

    def _resolve_aggr(self, aggr: Union[str, nn.Module], **kwargs):
        """This method should resolve the aggregation from string or a module."""
        if isinstance(aggr, str):
            if aggr == 'sum':
                return nn.SumAggregation(**kwargs)  # Implement SumAggregation in Jittor
            elif aggr == 'mean':
                return nn.MeanAggregation(**kwargs)  # Implement MeanAggregation in Jittor
            # Add more aggregation cases here
        elif isinstance(aggr, nn.Module):
            return aggr
        raise ValueError(f"Unsupported aggregation type: {aggr}")

    def _get_dense_combine(self, mode: str):
        """Return the dense combine operation based on mode."""
        dense_combine_modes = ['sum', 'mean', 'max', 'min', 'logsumexp', 'std', 'var']
        if mode in dense_combine_modes:
            return getattr(jt, mode)
        raise ValueError(f"Combine mode '{mode}' is not supported.")

    def reset_parameters(self):
        for aggr in self.aggrs:
            aggr.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
        if hasattr(self, 'multihead_attn'):
            self.multihead_attn.reset_parameters()

    def get_out_channels(self, in_channels: int) -> int:
        if self.out_channels is not None:
            return self.out_channels
        if self.mode == 'cat':
            return in_channels * len(self.aggrs)
        return in_channels

    def execute(self, x: jt.Var, index: Optional[jt.Var] = None,
                ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> jt.Var:
        outs = [aggr(x, index, ptr, dim_size, dim) for aggr in self.aggrs]
        return self.combine(outs)

    def combine(self, inputs: List[jt.Var]) -> jt.Var:
        if len(inputs) == 1:
            return inputs[0]

        if self.mode == 'cat':
            return jt.concat(inputs, dim=-1)

        if hasattr(self, 'lin'):
            return self.lin(jt.concat(inputs, dim=-1))

        if hasattr(self, 'multihead_attn'):
            x_dict = {str(k): v for k, v, in enumerate(inputs)}
            x_dict = self.lin_heads(x_dict)
            xs = [x_dict[str(key)] for key in range(len(inputs))]
            x = jt.stack(xs, dim=0)
            attn_out, _ = self.multihead_attn(x, x, x)
            return jt.mean(attn_out, dim=0)

        if hasattr(self, 'dense_combine'):
            out = self.dense_combine(jt.stack(inputs, dim=0), dim=0)
            return out if isinstance(out, jt.Var) else out[0]

        raise ValueError(f"Combine mode '{self.mode}' is not supported.")

    def __repr__(self) -> str:
        aggrs = ',\n'.join([f'  {aggr}' for aggr in self.aggrs]) + ',\n'
        return f'{self.__class__.__name__}([\n{aggrs}], mode={self.mode})'
