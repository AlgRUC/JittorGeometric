import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os.path as osp
from typing import Any, Callable, Dict, List, Optional
import pickle
from huggingface_hub import hf_hub_download
import jittor as jt
from tqdm import tqdm

from jittor_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from jittor_geometric.utils import from_smiles as _from_smiles


class PCQM4Mv2(InMemoryDataset):
    r"""The PCQM4Mv2 dataset from the `"OGB-LSC: A Large-Scale Challenge for
    Machine Learning on Graphs" <https://arxiv.org/abs/2103.09430>`_ paper.
    :class:`PCQM4Mv2` is a quantum chemistry dataset originally curated under
    the `PubChemQC project
    <https://pubs.acs.org/doi/10.1021/acs.jcim.7b00083>`_.
    The task is to predict the DFT-calculated HOMO-LUMO energy gap of molecules
    given their 2D molecular graphs.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            If :obj:`"holdout"`, loads the holdout dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`jittor_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        from_smiles (callable, optional): A custom function that takes a SMILES
            string and outputs a :obj:`~jittor_geometric.data.Data` object.
            If not set, defaults to :meth:`~jittor_geometric.utils.from_smiles`.
            (default: :obj:`None`)
    """
    url = ('https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/'
           'pcqm4m-v2.zip')

    split_mapping = {
        'train': 'train',
        'val': 'valid',
        'test': 'test-dev',
        'holdout': 'test-challenge',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        from_smiles: Optional[Callable] = None,
    ) -> None:
        assert split in ['train', 'val', 'test', 'holdout']

        schema = {
            'x': dict(dtype=jt.int64, size=(-1, 9)),
            'edge_index': dict(dtype=jt.int64, size=(2, -1)),
            'edge_attr': dict(dtype=jt.int64, size=(-1, 3)),
            'smiles': str,
            'y': float,
        }

        self.split = split
        self.from_smiles = from_smiles or _from_smiles
        super().__init__(root, transform)
        with open(self.raw_paths[1], 'rb') as f:
            split_idx = pickle.load(f)
        self._indices = split_idx[self.split_mapping[split]].tolist()
        self.data, self.slices = jt.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('pcqm4m-v2', 'raw', 'data.csv.gz'),
            osp.join('pcqm4m-v2', 'split_dict.pkl'),
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        hf_hub_download(repo_id=f"Drug-Data/PCQM4Mv2", filename=f"split_dict.pkl", local_dir=osp.join(self.raw_dir,'pcqm4m-v2'), repo_type="dataset")

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        data_list: List[Data] = []
        iterator = enumerate(zip(df['smiles'], df['homolumogap']))
        for i, (smiles, y) in tqdm(iterator, total=len(df)):
            try:
                data = self.from_smiles(smiles)
                data.y = float(y)
                data.smiles = smiles
                data_list.append(data)
            except Exception as e:
                print(f"Warning: Failed to process SMILES '{smiles}': {e}")
                continue

        jt.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)}, split="{self.split}")'