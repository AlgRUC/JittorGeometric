import os.path as osp
from typing import Callable, Optional

import pandas as pd
from jittor_geometric.data import InMemoryDataset, TemporalData, download_url


from jittor_geometric.data import InMemoryDataset, download_url
import jittor as jt

class JODIEDataset(InMemoryDataset):
    r"""The temporal graph datasets from the `"JODIE: Predicting Dynamic Embedding
    Trajectory in Temporal Interaction Networks"
    <https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Reddit"`,
            :obj:`"Wikipedia"`, :obj:`"MOOC"`, and :obj:`"LastFM"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """
    url = 'http://snap.stanford.edu/jodie/{}.csv'
    names = ['reddit', 'wikipedia', 'mooc', 'lastfm']

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in self.names
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])
        # print('self.processed_paths[0]',self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        # return 'data.pt'
        return 'data.pkl'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):

        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = jt.array(df.iloc[:, 0].values).to(jt.int64)
        dst = jt.array(df.iloc[:, 1].values).to(jt.int64)
        dst += int(src.max()) + 1
        t = jt.array(df.iloc[:, 2].values).to(jt.int64)
        y = jt.array(df.iloc[:, 3].values).to(jt.int64)
        msg = jt.array(df.iloc[:, 4:].values).to(jt.float32)

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
