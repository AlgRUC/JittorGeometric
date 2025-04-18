import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import jittor as jt
from jittor import dataset, Var
from tqdm import tqdm
import numpy as np
from jittor_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from jittor_geometric.utils import one_hot, scatter
from huggingface_hub import hf_hub_download

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = jt.array([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class QM9(InMemoryDataset):
    r"""
    #     ! IF YOU MEET NETWORK ERROR, PLEASE TRY TO RUN THE COMMAND BELOW:
    # `export HF_ENDPOINT=https://hf-mirror.com`,
    # TO USE THE MIRROR PROVIDED BY Hugging Face.

    The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    .. note::

        We also provide a pre-processed version of the dataset in case
        :class:`rdkit` is not installed. The pre-processed version matches with
        the manually processed version as outlined in :meth:`process`.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`jt_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`jt_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`jt_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """  # noqa: E501

    # raw_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    # raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = jt.load(self.processed_paths[0])
        self.num_data = len(self.data['idx'])

    def mean(self, target: int) -> float:
        y = jt.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = jt.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Var]:
        if target in atomrefs:
            out = jt.zeros(100)
            out[jt.Var([1, 6, 7, 8, 9])] = jt.Var(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        return ['qm9.pkl']

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self) -> None:
        hf_hub_download(repo_id=f"Drug-Data/QM9", filename=f"qm9.pkl", local_dir=self.raw_dir, repo_type="dataset")

    def process(self) -> None:
        WITH_RDKIT = False

        if not WITH_RDKIT:
            print(("Using a pre-processed version of the dataset."),
                  file=sys.stderr)
            data, slices = jt.load(self.raw_paths[0])
            # if self.pre_filter is not None:
            #     data_list = [d for d in data_list if self.pre_filter(d)]

            # if self.pre_transform is not None:
            #     data_list = [self.pre_transform(d) for d in data_list]
            
            jt.save((data, slices), self.processed_paths[0])
            return

        # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        # bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # with open(self.raw_paths[1]) as f:
        #     target = [[float(x) for x in line.split(',')[1:20]]
        #               for line in f.read().split('\n')[1:-1]]
        #     y = jt.array(target)
        #     y = jt.cat([y[:, 3:], y[:, :3]], dim=-1)
        #     y = y * conversion.view(1, -1)

        # with open(self.raw_paths[2]) as f:
        #     skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        # suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
        #                            sanitize=False)

        # data_list = []
        # for i, mol in enumerate(tqdm(suppl)):
        #     if i in skip:
        #         continue

        #     N = mol.GetNumAtoms()

        #     conf = mol.GetConformer()
        #     pos = conf.GetPositions()
        #     pos = jt.Var(pos)

        #     type_idx = []
        #     atomic_number = []
        #     aromatic = []
        #     sp = []
        #     sp2 = []
        #     sp3 = []
        #     num_hs = []
        #     for atom in mol.GetAtoms():
        #         type_idx.append(types[atom.GetSymbol()])
        #         atomic_number.append(atom.GetAtomicNum())
        #         aromatic.append(1 if atom.GetIsAromatic() else 0)
        #         hybridization = atom.GetHybridization()
        #         sp.append(1 if hybridization == HybridizationType.SP else 0)
        #         sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        #         sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        #     z = jt.Var(atomic_number)

        #     rows, cols, edge_types = [], [], []
        #     for bond in mol.GetBonds():
        #         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #         rows += [start, end]
        #         cols += [end, start]
        #         edge_types += 2 * [bonds[bond.GetBondType()]]

        #     edge_index = jt.Var([rows, cols])
        #     edge_type = jt.Var(edge_types)
        #     edge_attr = one_hot(edge_type, num_classes=len(bonds))
        #     perm = (edge_index[0] * N + edge_index[1]).argsort()[0]
        #     edge_index = edge_index[:, perm]
            
        #     edge_type = edge_type[perm]
        #     edge_attr = edge_attr[perm]

        #     row, col = edge_index
        #     hs = (z == 1).to(jt.float)
        #     num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

        #     x1 = one_hot(jt.Var(type_idx), num_classes=len(types))
        #     x2 = jt.Var([atomic_number, aromatic, sp, sp2, sp3, num_hs],
        #                       ).t().contiguous()
        #     x = jt.cat([x1, x2], dim=-1)

        #     name = mol.GetProp('_Name')
        #     smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        #     data = Data(
        #         x=x,
        #         z=z,
        #         pos=pos,
        #         edge_index=edge_index,
        #         smiles=smiles,
        #         edge_attr=edge_attr,
        #         y=y[i].unsqueeze(0),
        #         name=name,
        #         idx=i,
        #     )

        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)

        #     data_list.append(data)

        # jt.save(self.collate(data_list), self.processed_paths[0])

    def get_idx_split(self, frac_train: float = 0.8, frac_valid: float = 0.1, frac_test: float = 0.1, seed: int = 42):

            assert np.isclose(frac_train + frac_valid + frac_test, 1.0)

            if seed is not None:
                np.random.seed(seed)

            # random split
            num_data = self.num_data
            shuffled_indices = np.random.permutation(num_data)

            train_cutoff = int(frac_train * num_data)
            valid_cutoff = int((frac_train + frac_valid) * num_data)

            train_idx = jt.array(shuffled_indices[:train_cutoff])
            valid_idx = jt.array(shuffled_indices[train_cutoff:valid_cutoff])
            test_idx = jt.array(shuffled_indices[valid_cutoff:])

            split_dict = {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx
            }
            return split_dict