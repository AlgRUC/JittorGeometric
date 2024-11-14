import os.path as osp
import numpy as np
import jittor as jt
from jittor_geometric.data import Data, InMemoryDataset, download_url
from typing import Callable, Optional


class HeteroDataset(InMemoryDataset):
    r"""Heterophilic dataset from the paper 'A critical look at the evaluation of GNNs under 
    heterophily: Are we really making progress?'
    <https://arxiv.org/abs/2302.11640>.

    Dataset Details:

    - **Roman Empire**: A graph from the Wikipedia article on the Roman Empire. Nodes represent words, 
      connected based on their order or syntactic dependencies, with the task to classify words by syntactic roles.
    - **Amazon Ratings**: Based on Amazon co-purchasing data, where nodes are products connected if frequently
      bought together. The task is to predict the product's average rating.
    - **Minesweeper**: A synthetic graph based on Minesweeper. Nodes represent grid cells with edges to neighbors.
      The task is to predict which cells contain mines.
    - **Tolokers**: Represents workers from the Toloka platform, with edges indicating shared tasks. The goal 
      is to predict if a worker was banned.
    - **Questions**: Built from Yandex Q data, with nodes representing users who answered each other's questions 
      on the topic of medicine. The task is to predict if a user remained active.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset to load. Options include:
            - `"roman-empire"`
            - `"amazon-ratings"`
            - `"minesweeper"`
            - `"tolokers"`
            - `"questions"`
        transform (callable, optional): A function/transform that takes in a :obj:`Data` object 
            and returns a transformed version. The data object will be transformed on every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a :obj:`Data` object 
            and returns a transformed version. The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
    """

    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')

    def __init__(self, root: str, name: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None):
        self.root = root
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])  # Jittor's loading method

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self):
        download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)

    def process(self, undirected=True):
        data = np.load(self.raw_paths[0])

        x = jt.array(data['node_features'])
        y = jt.array(data['node_labels'])
        edge_index = jt.array(data['edges']).transpose()

        if undirected:
            reverse_edges = edge_index.flip(0)
            edge_index = jt.contrib.concat([edge_index, reverse_edges], dim=1)
            edge_index = jt.unique(edge_index, dim=1)

        train_mask = jt.array(data['train_masks']).bool()
        val_mask = jt.array(data['val_masks']).bool()
        test_mask = jt.array(data['test_masks']).bool()

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])