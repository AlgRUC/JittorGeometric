import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import jittor as jt
from jittor_geometric.data import Data, InMemoryDataset, download_url
from jittor_geometric.utils import coalesce


class GeomGCN(InMemoryDataset):
    r"""The GeomGCN datasets used in the
    "Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.

    This class represents the datasets used in the Geom-GCN paper, which focuses on geometric graph convolutional networks. The datasets consist of graphs where nodes represent various entities, and edges represent relationships between them. The goal is to apply graph convolutional networks (GCNs) in the context of geometric graphs to classify nodes based on their features.

    Dataset Details:

    - **Cornell, Texas, Wisconsin**: These datasets represent web pages from the Cornell, Texas, and Wisconsin universities, where nodes are web pages, and edges represent hyperlinks between them. The task is to classify web pages into one of five categories: student, project, course, staff, and faculty.
    - **Actor**: In the Actor dataset, each node corresponds to an actor, and edges between nodes represent co-occurrence on the same Wikipedia page. The task is to classify the actors into one of five categories based on keywords extracted from their Wikipedia pages.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset to load. Options include:
            - :obj:`"Cornell"`
            - :obj:`"Texas"`
            - :obj:`"Wisconsin"`
            - :obj:`"Actor"`
        transform (callable, optional): A function/transform that takes in a :obj:`jittor_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a :obj:`jittor_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)

    Example:
        >>> dataset = GeomGCN(root='/path/to/dataset', name='Cornell')
        >>> dataset.data
        >>> dataset[0]  # Accessing the first data point
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'wisconsin', 'actor']

        super(GeomGCN, self).__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name == 'actor':
            tmp_name = 'film'
        else:
            tmp_name = self.name
        out = ['out1_node_feature_label.txt', 'out1_graph_edges.txt']
        out += [f'{tmp_name}_split_0.6_0.2_{i}.npz' for i in range(10)]
        return out

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self) -> None:
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self) -> None:
        if self.name != 'actor':
            with open(self.raw_paths[0]) as f:
                lines = f.read().split('\n')[1:-1]
                xs = [[float(value) for value in line.split('\t')[1].split(',')]
                    for line in lines]
                x = jt.array(xs, dtype=jt.float32)

                ys = [int(line.split('\t')[2]) for line in lines]
                y = jt.array(ys, dtype=jt.int32)
        else:
            with open(self.raw_paths[0]) as f:
                node_data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

                rows, cols = [], []
                for n_id, line, _ in node_data:
                    indices = [int(x) for x in line.split(',')]
                    rows += [int(n_id)] * len(indices)
                    cols += indices
                row, col = jt.array(rows), jt.array(cols)

                x = jt.zeros(int(row.max()) + 1, int(col.max()) + 1)
                x[row, col] = 1.

                y = jt.empty(len(node_data), dtype=jt.int32)
                for n_id, _, label in node_data:
                    y[int(n_id)] = int(label)

        with open(self.raw_paths[1]) as f:
            lines = f.read().split('\n')[1:-1]
            edge_indices = [[int(value) for value in line.split('\t')]
                            for line in lines]
            edge_index = jt.array(edge_indices).transpose().astype(jt.int32)
            edge_index, _ = coalesce(edge_index, num_nodes=x.shape[0])

        train_masks, val_masks, test_masks = [], [], []
        for path in self.raw_paths[2:]:
            tmp = np.load(path)
            train_masks += [jt.array(tmp['train_mask']).bool()]
            val_masks += [jt.array(tmp['val_mask']).bool()]
            test_masks += [jt.array(tmp['test_mask']).bool()]
        train_mask = jt.stack(train_masks, dim=1)
        val_mask = jt.stack(val_masks, dim=1)
        test_mask = jt.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'