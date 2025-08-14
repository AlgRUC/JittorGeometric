import os.path as osp

from jittor_geometric.io import read_planetoid_data
from jittor_geometric.data import InMemoryDataset, download_url
import jittor as jt
from jittor import init


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.

    This class represents three widely-used citation network datasets: Cora, CiteSeer, and PubMed. Nodes correspond to documents, and edges represent citation links between them. The datasets are designed for semi-supervised learning tasks, where training, validation, and test splits are provided as binary masks.

    Dataset Details:
    
    - **Cora**: A citation network where nodes represent machine learning papers, and edges represent citations. The task is to classify papers into one of seven classes.
    - **CiteSeer**: A citation network of research papers in computer and information science. The task is to classify papers into one of six classes.
    - **PubMed**: A citation network of biomedical papers on diabetes. The task is to classify papers into one of three classes.

    Splitting Options:
    - **public**: The original fixed split from the paper `"Revisiting Semi-Supervised Learning with Graph Embeddings"`.
    - **full**: Uses all nodes except those in the validation and test sets for training, inspired by `"FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling"`.
    - **random**: Generates random splits for train, validation, and test sets based on the specified parameters.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (str): The type of dataset split (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            Default is :obj:`"public"`.
        num_train_per_class (int, optional): Number of training samples per class for :obj:`"random"` split. Default is 20.
        num_val (int, optional): Number of validation samples for :obj:`"random"` split. Default is 500.
        num_test (int, optional): Number of test samples for :obj:`"random"` split. Default is 1000.
        transform (callable, optional): A function/transform that takes in a :obj:`torch_geometric.data.Data` object and returns a transformed version. Default is :obj:`None`.
        pre_transform (callable, optional): A function/transform that takes in a :obj:`torch_geometric.data.Data` object and returns a transformed version before saving to disk. Default is :obj:`None`.

    Example:
        >>> dataset = Planetoid(root='/path/to/dataset', name='Cora', split='random')
        >>> data = dataset[0]  # Access the processed data object
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name, split="public", num_train_per_class=20,
                 num_val=500, num_test=1000, transform=None,
                 pre_transform=None):
        self.name = name

        super(Planetoid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])
        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            init(data.train_mask, True)
            data.train_mask[jt.logical_or(
                data.val_mask, data.test_mask)] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            init(data.train_mask, False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero().view(-1)
                idx = idx[jt.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = jt.logical_not(data.train_mask).nonzero().view(-1)
            remaining = remaining[jt.randperm(remaining.size(0))]

            init(data.val_mask, False)
            data.val_mask[remaining[:num_val]] = True

            init(data.test_mask, False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pkl'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

    def make_link_split(self, val_ratio=0.05, test_ratio=0.10):
        data = self.get(0)

        edge_index = data.edge_index.int32()  # [2, E]

        row = edge_index[0]
        col = edge_index[1]

        E_u = int(row.shape[0])
        n_val = int(math.floor(val_ratio * E_u))
        n_test = int(math.floor(test_ratio * E_u))

        # Shuffle
        perm = jt.randperm(E_u)
        row_u = row[perm]
        col_u = col[perm]

        # Slice into val / test / train
        def stack_edges(r, c):
            if int(r.shape[0]) == 0:
                return jt.zeros((2, 0), dtype="int32")
            return jt.stack([r, c], dim=0)

        val_pos = stack_edges(row_u[:n_val], col_u[:n_val])
        test_pos = stack_edges(row_u[n_val:n_val + n_test], col_u[n_val:n_val + n_test])
        train_pos = stack_edges(row_u[n_val + n_test:], col_u[n_val + n_test:])

        # Write back to data
        data.train_pos_edge_index = train_pos
        data.val_pos_edge_index = val_pos
        data.test_pos_edge_index = test_pos

        return data