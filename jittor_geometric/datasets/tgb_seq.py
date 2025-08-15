import os.path as osp
from typing import Callable, Optional

import pandas as pd
from jittor_geometric.data import InMemoryDataset, TemporalData, download_url
import os
import numpy as np
from jittor_geometric.data import InMemoryDataset, download_url
import jittor as jt
from huggingface_hub import hf_hub_download


class TGBSeqDataset(InMemoryDataset):
    r"""The temporal graph datasets from the paper 
    "TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics"
    <https://openreview.net/pdf?id=8e2LirwiJT>.

    ! IF YOU MEET NETWORK ERROR, PLEASE TRY TO RUN THE COMMAND BELOW:
    `export HF_ENDPOINT=https://hf-mirror.com`,
    TO USE THE MIRROR PROVIDED BY Hugging Face.

    Dataset Details:

    - **ML-20M**: This dataset contains movie rating data, where each record includes the rating recode from a user to a specific movie along with the timestamp of the rating. The rating is omitted in the dataset to represent the implicit feedback. 
    - **Taobao**: This dataset is a user behavior dataset derived from the e-commerce platform Taobao. It contains user click data on products from November 25, 2017, to December 3, 2017. The dataset is a bipartite graph where users and products are nodes, and an edge represents a user's click on a product at a given time.
    - **Yelp**: This dataset is a business review dataset sourced from Yelp, a prominent platform for business recommendations, including restaurants, bars, and beauty salons. It contains user reviews of businesses from 2018 to 2022. The dataset is a bipartite graph where users and businesses are nodes, and an edge represents a user's review of a business at a given time.
    - **GoogleLocal**: This dataset is a business review dataset derived from Google Maps, containing user reviews and ratings of local businesses. The GoogleLocal dataset is a bipartite graph where users and businesses are nodes, and an edge indicates a user's review of a business at a given time.
    - **Flickr**: This dataset is a ``Who-To-Follow'' social network dataset derived from Flickr, a photo-sharing platform with social networking features. The dataset was crawled daily from November 2 to December 3, 2006, and from February 3 to March 18, 2007. It is estimated to represent 25% of the entire Flickr network. The Flickr dataset is a non-bipartite graph where users are nodes, and an edge represents the friendship established between users at a given time.
    - **YouTube**: This dataset is a ``Who-To-Follow'' social network dataset derived from YouTube, a video-sharing platform that includes a user subscription network. Similar to Flickr, the YouTube dataset is a non-bipartite graph where users are nodes, and an edge indicates the subscription of a user to another user at a given time.
    - **Patent**: This dataset is a citation network dataset of U.S. patents, capturing the citation relationships between patents from 1963 to 1999. The dataset is organized as a non-bipartite graph where patents are nodes, and an edge represents a citation made by one patent to another at the time of publication.
    - **WikiLink**: This dataset is a web link network dataset derived from Wikipedia, containing the hyperlink relationships between Wikipedia pages. This dataset is a non-bipartite graph, where pages are nodes and edges indicate hyperlinks established from one page to another at a given time. 

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset, options include:
            - :obj:`"ML-20M"`
            - :obj:`"Taobao"`
            - :obj:`"Yelp"`
            - :obj:`"GoogleLocal"`
            - :obj:`"Flickr"`
            - :obj:`"YouTube"`
            - :obj:`"Patent"`
            - :obj:`"WikiLink"`
        transform (callable, optional): A function/transform that takes in a 
            :obj:`Data` object and returns a transformed version. The data object 
            will be transformed on each access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a 
            :obj:`Data` object and returns a transformed version. The data object 
            will be transformed before being saved to disk. (default: :obj:`None`)
    """

    names = ['ML-20M', 'Taobao', 'Yelp', 'GoogleLocal',
             'Flickr', 'YouTube', 'WikiLink', 'wikipedia']

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name
        self.root = root
        self.downloaded_name = f"{self.name}"
        self.file_dir = os.path.join(self.root, self.name)
        assert self.name in self.names
        self.edgelist_path = os.path.join(
            self.root, self.name, f'{self.name}.csv')
        self.test_ns_path = os.path.join(
            self.root, self.name, f'{self.name}_test_ns.npy')
        self.edge_feat_path = os.path.join(
            self.root, self.name, f'{self.name}_edge_feat.npy')
        self.node_feat_path = os.path.join(
            self.root, self.name, f'{self.name}_node_feat.npy')
        super().__init__(root, transform, pre_transform)
        self._load_file()

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    def _load_file(self):
        if os.path.exists(self.file_dir):
            if not os.path.exists(self.edgelist_path):
                raise FileNotFoundError(f'Local dataset file {self.edgelist_path} not found')
            if not os.path.exists(self.test_ns_path):
                print(f'Warning: Local test negative samples file {self.test_ns_path} not found. We will generate it when testing.')
        else:
            os.makedirs(self.file_dir)
            self._download()
        self.edgelist_df = pd.read_csv(self.edgelist_path)
        self.edge_feat, self.node_feat, self.val_ns, self.test_ns = None, None, None, None
        if os.path.exists(self.test_ns_path):
            self.test_ns = np.load(self.test_ns_path)
        if os.path.exists(self.edge_feat_path):
            self.edge_feat = np.load(self.edge_feat_path)
        if os.path.exists(self.node_feat_path):
            self.node_feat = np.load(self.node_feat_path)
        self.src_node_ids = self.edgelist_df['src'].values.astype(np.longlong)
        self.dst_node_ids = self.edgelist_df['dst'].values.astype(np.longlong)
        self.time = self.edgelist_df['time'].values.astype(np.float64)
        self.train_mask = self.edgelist_df['split'] == 0
        self.val_mask = self.edgelist_df['split'] == 1
        self.test_mask = self.edgelist_df['split'] == 2

    def _download(self):
        if not self.name in self.names:
            raise ValueError(f'Dataset {self.name} not supported by TGB-Seq.')
        print(f"Download started, this might take a while . . . ")
        if not osp.isdir(self.root):
            os.makedirs(self.root)
        print(f"Dataset {self.name} will be downloaded in ", self.root)
        try:
            hf_hub_download(repo_id=f"TGB-Seq/{self.name}", filename=f"{self.downloaded_name}.csv", local_dir=self.file_dir, repo_type="dataset")
            hf_hub_download(repo_id=f"TGB-Seq/{self.name}", filename=f"{self.downloaded_name}_test_ns.npy", local_dir=self.file_dir, repo_type="dataset")
        except Exception as e:
            print(f"Error: {e}")
            exit()

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'

    @property
    def num_nodes(self):
        return int(max(self.src_node_ids.max(), self.dst_node_ids.max())) # node index starts from 1

    @property
    def num_edges(self):
        return len(self.src_node_ids) # note that edge index starts from 1