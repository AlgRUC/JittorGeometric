import jittor as jt
from jittor import nn
from typing import Optional, Tuple
import numpy as np

class Aggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: jt.Var,
        index: Optional[jt.Var] = None,
        ptr: Optional[jt.Var] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> jt.Var:
        raise NotImplementedError

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        pass

    def __call__(
        self,
        x: jt.Var,
        index: Optional[jt.Var] = None,
        ptr: Optional[jt.Var] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ) -> jt.Var:
        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if index is None and ptr is None:
            index = jt.zeros(x.shape[dim], dtype=jt.int64)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f"'{ptr.numel() - 1}')")

        if index is not None and dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        return self._call_aggregation(x, index=index, ptr=ptr, dim_size=dim_size,
                                      dim=dim, **kwargs)

    def _call_aggregation(self, x: jt.Var, index: Optional[jt.Var] = None,
                          ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
                          dim: int = -2, **kwargs) -> jt.Var:
        return self.reduce(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, reduce="sum")

    def assert_index_present(self, index: Optional[jt.Var]):
        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: Optional[jt.Var]):
        if index is not None and not jt.all(index[:-1] <= index[1:]):
            raise ValueError("Can not perform aggregation since the 'index' tensor is not sorted.")

    def assert_two_dimensional_input(self, x: jt.Var, dim: int):
        if x.dim() != 2:
            raise ValueError(f"Aggregation requires two-dimensional inputs (got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"Aggregation needs to perform aggregation in first dimension (got '{dim}')")

    def reduce(self, x: jt.Var, index: Optional[jt.Var] = None,
               ptr: Optional[jt.Var] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'sum') -> jt.Var:
        """Perform the aggregation (sum, max, mean, etc.)."""
        if ptr is not None:
            ptr = self.expand_left(ptr, dim, dims=x.dim())
            return self.segment(x, ptr, reduce=reduce)

        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")
        return jt.scatter(x, index, dim, dim_size, reduce)

    def segment(self, x: jt.Var, ptr: jt.Var, reduce: str = 'sum') -> jt.Var:
        """Segment operation using ptr, similar to `torch_scatter.segment`."""
        batch_size = ptr.numel() - 1
        output = jt.zeros((batch_size, x.shape[1]), dtype=x.dtype)

        for i in range(batch_size):
            start = ptr[i]
            end = ptr[i + 1]
            if reduce == 'sum':
                output[i] = jt.sum(x[start:end], dim=0)
            elif reduce == 'mean':
                output[i] = jt.mean(x[start:end], dim=0)
            elif reduce == 'max':
                output[i] = jt.max(x[start:end], dim=0)[0]
            else:
                raise ValueError(f"Unknown reduce operation: {reduce}")

        return output

    def to_dense_batch(
        self,
        x: jt.Var,
        index: Optional[jt.Var] = None,
        ptr: Optional[jt.Var] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        fill_value: float = 0.0,
        max_num_elements: Optional[int] = None,
    ) -> Tuple[jt.Var, jt.Var]:
        """Converts the aggregation input into a dense batch."""
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)
        return self.to_dense_batch_helper(x, index, dim_size, fill_value)

    def to_dense_batch_helper(self, x: jt.Var, index: jt.Var, dim_size: Optional[int], fill_value: float) -> Tuple[jt.Var, jt.Var]:
        """Helper function for creating dense batches."""
        batch_size = dim_size if dim_size is not None else int(index.max()) + 1
        max_num_nodes = max_num_nodes if max_num_nodes is not None else x.shape[0]
        dense_batch = jt.zeros((batch_size, max_num_nodes, x.shape[1]), dtype=x.dtype)
        for i in range(x.shape[0]):
            dense_batch[index[i]] = x[i]
        return dense_batch, index

def expand_left(ptr: jt.Var, dim: int, dims: int) -> jt.Var:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr
