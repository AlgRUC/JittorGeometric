'''
Description: 
Author: lusz
Date: 2024-11-14 19:28:16
'''

import os.path as osp
from typing import Callable, Optional

import jittor as jt
from jittor_geometric.io import read_npz
from jittor_geometric.data import InMemoryDataset, download_url


class Amazon(InMemoryDataset):
    r"""The Amazon Computers and Amazon Photo datasets from the paper 
    "Pitfalls of Graph Neural Network Evaluation" 
    <https://arxiv.org/abs/1811.05868>`_.
    
    This class represents the Amazon dataset used in the paper "Pitfalls of Graph Neural Network Evaluation". In this dataset, nodes represent products, and edges indicate that two products are frequently bought together. The dataset provides product reviews represented as bag-of-words node features, and the task is to classify products into their respective categories.

    Dataset Details:

    - **Amazon Computers**: This dataset contains products related to computers, where the task is to classify the products based on the reviews and co-purchase information.
    - **Amazon Photo**: This dataset contains products related to photography, with a similar task of classifying products based on reviews and co-purchase data.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset, either :obj:`"Computers"` or :obj:`"Photo"`.
        transform (callable, optional): A function/transform that takes in a :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed on each access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)

    Example:
        >>> dataset = Amazon(root='/path/to/dataset', name='Computers')
        >>> dataset.data
        >>> dataset[0]  # Accessing the first data point
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['computers', 'photo']
        super(Amazon, self).__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'amazon_electronics_{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self) -> None:
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self) -> None:
        data = read_npz(self.raw_paths[0], to_undirected=True)
        data = data if self.pre_transform is None else self.pre_transform(data)
        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return '{}()'.format(self.name)