import os.path as osp
from typing import Callable, Optional
import numpy as np
import pandas as pd
from jittor_geometric.data import InMemoryDataset, TemporalData, download_url
import jittor as jt

class JODIEDataset(InMemoryDataset):
    r"""The temporal graph datasets from the paper 
    "JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks"
    <https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>.

    Dataset Details:
    
    - **Reddit Post Dataset**: This dataset consists of interactions between users and subreddits. 
      We selected the 1,000 most active subreddits and the 10,000 most active users, resulting in 
      over 672,447 interactions. Each post's text is represented as a feature vector using LIWC categories.
    - **Wikipedia Edits**: This dataset represents edits made by users on Wikipedia pages. 
      We selected the 1,000 most edited pages and users with at least 5 edits, totaling 8,227 users 
      and 157,474 interactions. Each edit is converted into a LIWC-feature vector.
    - **LastFM Song Listens**: This dataset records user-song interactions, with 1,000 users and 
      the 1,000 most listened-to songs, resulting in 1,293,103 interactions. Unlike other datasets, 
      interactions do not have features.
    - **MOOC Student Drop-Out**: This dataset captures student interactions (e.g., viewing videos, 
      submitting answers) on a MOOC online course. There are 7,047 users interacting with 98 items 
      (videos, answers, etc.), generating over 411,749 interactions, including 4,066 drop-out events.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset, options include:
            - :obj:`"Reddit"`
            - :obj:`"Wikipedia"`
            - :obj:`"LastFM"`
            - :obj:`"MOOC"`
        transform (callable, optional): A function/transform that takes in a 
            :obj:`Data` object and returns a transformed version. The data object 
            will be transformed on each access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a 
            :obj:`Data` object and returns a transformed version. The data object 
            will be transformed before being saved to disk. (default: :obj:`None`)
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
        edge_ids = jt.arange(0, len(src))+1
        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y, edge_ids=edge_ids)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
    


class TemporalDataLoader:
    def __init__(self, data, batch_size=1, neg_sampling_ratio=0.0, drop_last=False):
        self.data = data
        self.batch_size = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio

        if neg_sampling_ratio > 0:
            self.min_dst = int(data.dst.min())
            self.max_dst = int(data.dst.max())

        data_len = len(data.src)
        self.arange = np.arange(0, data_len, batch_size)
        if drop_last and data_len % batch_size != 0:
            self.arange = self.arange[:-1]

    def __len__(self):
        return len(self.arange)

    def __iter__(self):
        for start in self.arange:
            end = start + self.batch_size
            batch = self.data[start:end]
            n_ids = [batch.src, batch.dst]

            if self.neg_sampling_ratio > 0:
                neg_dst_size = round(self.neg_sampling_ratio * len(batch.dst))
                neg_dst = jt.randint(low=self.min_dst, high=self.max_dst + 1, shape=(neg_dst_size,), dtype=batch.dst.dtype)
                batch.neg_dst = neg_dst
                n_ids += [batch.neg_dst]

            batch.n_id = jt.concat(n_ids).unique()
            yield batch
