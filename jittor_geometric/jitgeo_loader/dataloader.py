from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union, Callable
import jittor
from jittor.dataset.utils import collate_batch
from jittor_geometric.data import Batch, Dataset
from jittor_geometric.data.data import Data
import numpy as np


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[Data]],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, jittor.Var):
            return collate_batch(batch)
        elif isinstance(elem, float):
            return collate_batch(batch)
        elif isinstance(elem, int):
            return collate_batch(batch)
        elif isinstance(elem, str):
            return batch 
        elif isinstance(elem, Mapping):
            return collate_batch(batch)
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return collate_batch(batch)
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return collate_batch(batch)

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader:
    r"""A data loader which merges data objects from a
    :class:`jittor_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~jittor_geometric.data.Data` or
    :class:`~jittor_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[Data]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.collate_fn = collate_fn if collate_fn is not None else Collater(dataset, follow_batch, exclude_keys)

        # Initialize indices
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        # Reset indices if shuffle is enabled
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Yield batches
        for start_idx in range(0, len(self.indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx].tolist()
            batch = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size