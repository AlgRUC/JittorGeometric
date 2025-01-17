import os
import os.path as osp
import re
import warnings
from typing import Callable, Dict, Optional, Tuple, Union
import jittor as jt
from jittor_geometric.data import InMemoryDataset, download_url, extract_gz
from huggingface_hub import hf_hub_download


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the :ogb:`null`
    `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
            :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
            :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
            :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`jittor_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`jittor_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`jittor_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        from_smiles (callable, optional): A custom function that takes a SMILES
            string and outputs a :obj:`~jittor_geometric.data.Data` object.
            If not set, defaults to :meth:`~jittor_geometric.utils.from_smiles`.
            (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - ESOL
          - 1,128
          - ~13.3
          - ~27.4
          - 9
          - 1
        * - FreeSolv
          - 642
          - ~8.7
          - ~16.8
          - 9
          - 1
        * - ClinTox
          - 1,484
          - ~26.1
          - ~55.5
          - 9
          - 2
    """

    # url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: (display_name, url_name, csv_name, smiles_idx, y_idx)
    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        'esol': ('ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2),
        'freesolv': ('FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2),
        'clintox': ('ClinTox', 'clintox.csv.gz', 'clintox', 0, slice(1, 3)),
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = jt.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self) -> str:
        return f'{self.name}.pkl'

    def download(self) -> None:
        hf_hub_download(repo_id=f"TGB-Seq/MoleculeNet", filename=f"{self.name}.pkl", local_dir=self.processed_dir, repo_type="dataset")

    def process(self) -> None:
        pass

    def __repr__(self) -> str:
        return f'{self.name}(len={len(self)})'
