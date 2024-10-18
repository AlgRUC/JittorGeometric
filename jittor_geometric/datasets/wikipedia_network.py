import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import jittor as jt
from jittor_geometric.data import Data, InMemoryDataset, download_url
from jittor_geometric.utils import coalesce

class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
            If set to :obj:`True`, train/validation/test splits will be
            available as masks for multiple splits with shape
            :obj:`[num_nodes, num_splits]`. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`jittor_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`jittor_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(
        self,
        root: str,
        name: str,
        geom_gcn_preprocess: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'squirrel']
        super(WikipediaNetwork, self).__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[List[str], str]:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self) -> None:
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self) -> None:
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0]) as f:
                lines = f.read().split('\n')[1:-1]
            xs = [[float(value) for value in line.split('\t')[1].split(',')]
                  for line in lines]
            x = jt.array(xs, dtype=jt.float32)
            ys = [int(line.split('\t')[2]) for line in lines]
            y = jt.array(ys, dtype=jt.int32)

            with open(self.raw_paths[1]) as f:
                lines = f.read().split('\n')[1:-1]
                edge_indices = [[int(value) for value in line.split('\t')]
                                for line in lines]
            edge_index = jt.array(edge_indices).transpose().astype(jt.int32)
            edge_index = coalesce(edge_index, num_nodes=x.shape[0])

            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                masks = np.load(filepath)
                train_masks += [jt.array(masks['train_mask'])]
                val_masks += [jt.array(masks['val_mask'])]
                test_masks += [jt.array(masks['test_mask'])]
            train_mask = jt.stack(train_masks, dim=1).bool()
            val_mask = jt.stack(val_masks, dim=1).bool()
            test_mask = jt.stack(test_masks, dim=1).bool()

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            raw_data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = jt.array(raw_data['features'], dtype=jt.float32)
            edge_index = jt.array(raw_data['edges'], dtype=jt.int32).transpose()
            edge_index, _ = coalesce(edge_index, num_nodes=x.shape[0])
            y = jt.array(raw_data['target'], dtype=jt.float32)

            data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])
