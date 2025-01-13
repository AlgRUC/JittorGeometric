import jittor as jt
import numpy as np
from typing import Any, List, Optional, Type, Union
from collections.abc import Sequence
from jittor_geometric.data import Data


class Batch(Data):
    r"""A data object describing a batch of graphs as one big (disconnected) graph.
    Inherits from :class:`jittor_geometric.data.Data`.
    """

    @classmethod
    def from_data_list(
        cls,
        data_list: List[Data],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> 'Batch':
        r"""Constructs a :class:`~jittor_geometric.data.Batch` object from a
        list of :class:`~jittor_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        if exclude_keys is None:
            exclude_keys = []

        # Initialize batch
        batch = cls()
        batch._num_graphs = len(data_list)

        # Merge data
        slice_dict = {}
        inc_dict = {}
        for key in data_list[0].keys:
            if key in exclude_keys:
                continue

            items = [data[key] for data in data_list]
            if isinstance(items[0], jt.Var):
                if key == 'edge_index':
                    # Adjust edge_index for each graph
                    cum_nodes = 0
                    adjusted_edge_indices = []
                    for i, edge_index in enumerate(items):
                        adjusted_edge_indices.append(edge_index + cum_nodes)
                        cum_nodes += data_list[i].num_nodes
                    batch[key] = jt.concat(adjusted_edge_indices, dim=1)
                else:
                    batch[key] = jt.concat(items, dim=0)
            elif isinstance(items[0], (int, float)):
                batch[key] = jt.array(items)
            else:
                batch[key] = items

            # Create slice_dict for reconstruction
            if isinstance(items[0], jt.Var):
                slice_dict[key] = jt.cumsum(jt.array([data[key].shape[0] for data in data_list]), dim=0)
            else:
                slice_dict[key] = jt.cumsum(jt.array([1 for _ in data_list]), dim=0)

        # Create batch vector
        batch.batch = jt.concat([jt.full((data.num_nodes,), i) for i, data in enumerate(data_list)])

        # Handle follow_batch
        if follow_batch is not None:
            for key in follow_batch:
                if key in exclude_keys:
                    continue
                batch[f'{key}_batch'] = jt.concat([jt.full((data[key].shape[0],), i) for i, data in enumerate(data_list)])

        # Save slice_dict and inc_dict for reconstruction
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    def get_example(self, idx: int) -> Data:
        r"""Gets the :class:`~jittor_geometric.data.Data` object at index :obj:`idx`.
        The :class:`~jittor_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object.
        """
        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                "Cannot reconstruct 'Data' object from 'Batch' because "
                "'Batch' was not created via 'Batch.from_data_list()'")

        data = Data()
        for key in self.keys:
            if key == 'batch' or key.endswith('_batch'):
                continue

            if isinstance(self[key], jt.Var):
                start = 0 if idx == 0 else self._slice_dict[key][idx - 1].item()
                end = self._slice_dict[key][idx].item()
                data[key] = self[key][start:end]
            else:
                data[key] = self[key][idx]

        return data

    def to_data_list(self) -> List[Data]:
        r"""Reconstructs the list of :class:`~jittor_geometric.data.Data` objects
        from the :class:`~jittor_geometric.data.Batch` object.
        """
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'batch'):
            return int(self.batch.max().item()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    @property
    def batch_size(self) -> int:
        r"""Alias for :obj:`num_graphs`."""
        return self.num_graphs

    def __len__(self) -> int:
        return self.num_graphs