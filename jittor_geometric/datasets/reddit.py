import os
import os.path as osp
from typing import Callable, Optional,List
from jittor_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import pandas as pd
from jittor_geometric.data import InMemoryDataset, download_url
import jittor as jt
import numpy as np
from jittor_geometric.utils import coalesce


class Reddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    This dataset is designed for large-scale graph representation learning. Nodes in the graph represent Reddit posts, and edges represent interactions (e.g., comments) between posts in the same community. The task is to classify posts into one of the 41 communities based on their content and connectivity.

    **Dataset Statistics:**

    - **Number of Nodes**: 232,965
    - **Number of Edges**: 114,615,892
    - **Number of Features**: 602
    - **Number of Classes**: 41

    The dataset is pre-split into training, validation, and test sets using node type masks.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    Example:
        >>> dataset = Reddit(root='/path/to/reddit')
        >>> data = dataset[0]  # Access the first graph object
    """

    url = 'https://data.dgl.ai/dataset/reddit.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.data,self.slices= jt.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_file_names(self) -> str:
        # return 'data.pt'
        return osp.join('geometric_data_processed.pkl')

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import scipy.sparse as sp

        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = jt.array(data['feature']).to(jt.float32)
        y =jt.array(data['label']).to(jt.int32)
        split = jt.array(data['node_types'])

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = jt.array(adj.row).to(jt.int32)
        col = jt.array(adj.col).to(jt.int32)
        row = jt.unsqueeze(row, dim=1)
        col = jt.unsqueeze(col, dim=1) 

        arr=[]
        arr.append(row)
        arr.append(col)
        arr2 = jt.concat(arr, dim=1).transpose()
        edge_index,_ = coalesce(arr2, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])
