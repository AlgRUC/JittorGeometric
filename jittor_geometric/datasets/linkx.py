import os.path as osp
import numpy as np
import pandas as pd
import csv
import json
import jittor as jt
from jittor_geometric.data import Data, InMemoryDataset, download_url
from jittor_geometric.datasets import OGBNodePropPredDataset
import scipy
from scipy.io import loadmat
from typing import Callable, List, Optional


def even_quantile_labels(vals, nclasses, verbose=True):
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features


def load_twitch(lang, root):
    filepath = root + "/twitch-de/raw"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int64)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0]  # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features


class LINKXDataset(InMemoryDataset):
    r"""A variety of non-homophilous graph datasets from the `"Large Scale
    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
    Methods" <https://arxiv.org/abs/2110.14446>`_ paper.
    """

    github_url = ('https://github.com/CUAI/Non-Homophily-Large-Scale/'
                  'raw/master/data')
    gdrive_url = 'https://drive.google.com/uc?confirm=t&'

    facebook_datasets = [
        'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55'
    ]

    datasets = {
        'penn94': {
            'Penn94.mat': f'{github_url}/facebook100/Penn94.mat'
        },
        'reed98': {
            'Reed98.mat': f'{github_url}/facebook100/Reed98.mat'
        },
        'amherst41': {
            'Amherst41.mat': f'{github_url}/facebook100/Amherst41.mat',
        },
        'cornell5': {
            'Cornell5.mat': f'{github_url}/facebook100/Cornell5.mat'
        },
        'johnshopkins55': {
            'Johns%20Hopkins55.mat': f'{github_url}/facebook100/Johns%20Hopkins55.mat'
        },
        'genius': {
            'genius.mat': f'{github_url}/genius.mat'
        },
        'snap-patents': {
            'snap-patents.mat': f'{github_url}/snap-patents.mat'
        },
        'twitch-de': {
            'musae_DE_edges.csv': f'{github_url}/twitch/DE/musae_DE_edges.csv',
            'musae_DE_features.json': f'{github_url}/twitch/DE/musae_DE_features.json',
            'musae_DE_target.csv': f'{github_url}/twitch/DE/musae_DE_target.csv'
        },
        'deezer-europe': {
            'deezer-europe.mat': f'{github_url}/deezer-europe.mat'
        },
        # TODO: fix wiki
        #'wiki': {
        #    'wiki_views2M.pt':
        #    f'{gdrive_url}id=1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',
        #    'wiki_edges2M.pt':
        #    f'{gdrive_url}id=14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',
        #    'wiki_features2M.pt':
        #    f'{gdrive_url}id=1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
        #},
        'wiki':{
            'wiki_edges2M.npy': '',
            'wiki_features2M.npy': '',
            'wiki_views2M.npy': '',
        },
        'twitch-gamer': {
            'twitch-gamer_feat.csv':
            f'{gdrive_url}id=1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
            'twitch-gamer_edges.csv':
            f'{gdrive_url}id=1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
        },
        'pokec': {
            'pokec.mat': f'{github_url}/pokec.mat'
        },
        'arxiv-year': {},
    }

    splits = {
        'penn94': f'{github_url}/splits/fb100-Penn94-splits.npy',
        'genius': f'{github_url}/splits/genius-splits.npy',
        'twitch-gamers': f'{github_url}/splits/twitch-gamers-splits.npy',
        'snap-patents': f'{github_url}/splits/snap-patents-splits.npy',
        'pokec': f'{github_url}/splits/pokec-splits.npy', 
        'twitch-de': f'{github_url}/splits/twitch-e-DE-splits.npy',
        'deezer-europe': f'{github_url}/splits/deezer-europe-splits.npy',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super(LINKXDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = list(self.datasets[self.name].keys())
        if self.name in self.splits:
            names += [self.splits[self.name].split('/')[-1]]
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pkl'

    def download(self):
        if self.name in self.datasets.keys():
            for filename, path in self.datasets[self.name].items():
                download_url(path, self.raw_dir)
        if self.name in self.splits:
            download_url(self.splits[self.name], self.raw_dir)

    def _process_wiki(self):
        paths = {x.split('/')[-1]: x for x in self.raw_paths}
        x = jt.Var(np.load(paths['wiki_features2M.npy']))
        edge_index = jt.Var(np.load(paths['wiki_edges2M.npy'])).transpose(1, 0)
        y = jt.Var(np.load(paths['wiki_views2M.npy']))

        return Data(x=x, edge_index=edge_index, y=y)

    def _process_gamer(self):
        task = 'mature'
        normalize = True
        paths = {x.split('/')[-1]: x for x in self.raw_paths}
        edges = pd.read_csv(paths['twitch-gamer_edges.csv'])
        nodes = pd.read_csv(paths['twitch-gamer_feat.csv'])
        edge_index = jt.Var(edges.to_numpy()).t().long()
        num_nodes = len(nodes)
        label, features = load_twitch_gamer(nodes, task)
        feat_std = np.std(features, axis=0, keepdims=True)
        node_feat = jt.Var(features).float()
        if normalize:
            node_feat = (node_feat - node_feat.mean(dim=0, keepdims=True)) / feat_std

        x = node_feat.float()
        y = jt.Var(label).long()
        edge_index = edge_index.long()
        data = Data(x=x, edge_index=edge_index, y=y)
        if self.name in self.splits:
            splits = np.load(paths['twitch-gamer-splits.npy'], allow_pickle=True)
            sizes = (data.num_nodes, len(splits))
            data.train_mask = jt.zeros(sizes, dtype='bool')
            data.val_mask = jt.zeros(sizes, dtype='bool')
            data.test_mask = jt.zeros(sizes, dtype='bool')

            for i, split in enumerate(splits):
                data.train_mask[:, i][jt.Var(split['train']).long()] = True
                data.val_mask[:, i][jt.Var(split['valid']).long()] = True
                data.test_mask[:, i][jt.Var(split['test']).long()] = True

        return data

    def _process_deezer_europe(self):
        mat = loadmat(self.raw_paths[0])
        A, label, features = mat['A'], mat['label'], mat['features']
        edge_index = jt.Var(np.array(A.nonzero())).long()
        tmp_x = np.array(features.todense())
        x = jt.Var(tmp_x).float32()
        y = jt.Var(label).long().squeeze()
        data = Data(x=x, edge_index=edge_index, y=y)
        splits = np.load(self.raw_paths[1], allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = jt.zeros(sizes, dtype='bool')
        data.val_mask = jt.zeros(sizes, dtype='bool')
        data.test_mask = jt.zeros(sizes, dtype='bool')
        for i, split in enumerate(splits):
            data.train_mask[:, i][jt.Var(split['train']).long()] = True
            data.val_mask[:, i][jt.Var(split['valid']).long()] = True
            data.test_mask[:, i][jt.Var(split['test']).long()] = True
        return data

    def _process_facebook(self):
        mat = loadmat(self.raw_paths[0])

        A = mat['A'].tocsr().tocoo()
        row = jt.Var(A.row).long()
        col = jt.Var(A.col).long()
        edge_index = jt.stack([row, col], dim=0)

        metadata = jt.Var(mat['local_info'].astype('int64'))

        xs = []
        y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        x = jt.concat([metadata[:, :1], metadata[:, 2:]], dim=-1)
        for i in range(x.size(1)):
            _, out = x[:, i].unique(return_inverse=True)
            xs.append(jt.nn.one_hot(out).float())
        x = jt.concat(xs, dim=-1)

        data = Data(x=x, edge_index=edge_index, y=y)
        if self.name in self.splits:
            splits = np.load(self.raw_paths[1], allow_pickle=True)
            sizes = (data.num_nodes, len(splits))
            data.train_mask = jt.zeros(sizes, dtype='bool')
            data.val_mask = jt.zeros(sizes, dtype='bool')
            data.test_mask = jt.zeros(sizes, dtype='bool')

            for i, split in enumerate(splits):
                data.train_mask[:, i][jt.Var(split['train']).long()] = True
                data.val_mask[:, i][jt.Var(split['valid']).long()] = True
                data.test_mask[:, i][jt.Var(split['test']).long()] = True

        return data

    def _process_mat(self):
        mat = loadmat(self.raw_paths[0])
        edge_index = jt.Var(mat['edge_index']).long()
        x = jt.Var(mat['node_feat']).float()
        y = jt.Var(mat['label']).squeeze().long()
        data = Data(x=x, edge_index=edge_index, y=y)
        splits = np.load(self.raw_paths[1], allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = jt.zeros(sizes, dtype='bool')
        data.val_mask = jt.zeros(sizes, dtype='bool')
        data.test_mask = jt.zeros(sizes, dtype='bool')
        for i, split in enumerate(splits):
            data.train_mask[:, i][jt.Var(split['train']).long()] = True
            data.val_mask[:, i][jt.Var(split['valid']).long()] = True
            data.test_mask[:, i][jt.Var(split['test']).long()] = True
        return data

    def _process_snap_patents(self):
        mat = loadmat(self.raw_paths[0])
        edge_index = jt.Var(mat['edge_index']).long()
        tmp_x = np.array(mat['node_feat'].todense())
        x = jt.Var(tmp_x).float()
        years = mat['years'].flatten()
        nclass = 5
        labels = even_quantile_labels(years, nclass, verbose=False)
        y = jt.Var(labels).long()

        data = Data(x=x, edge_index=edge_index, y=y)
        splits = np.load(self.raw_paths[1], allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = jt.zeros(sizes, dtype='bool')
        data.val_mask = jt.zeros(sizes, dtype='bool')
        data.test_mask = jt.zeros(sizes, dtype='bool')
        for i, split in enumerate(splits):
            data.train_mask[:, i][jt.Var(split['train']).long()] = True
            data.val_mask[:, i][jt.Var(split['valid']).long()] = True
            data.test_mask[:, i][jt.Var(split['test']).long()] = True
        return data

    def _process_twitch_de(self):
        lang = 'DE'
        A, label, features = load_twitch(lang, root=self.root)
        edge_index = jt.Var(np.array(A.nonzero())).long()
        node_feat = jt.Var(features).float()
        label = jt.Var(label).long()

        data = Data(x=node_feat, edge_index=edge_index, y=label)
        paths = {x.split('/')[-1]: x for x in self.raw_paths}
        splits = np.load(paths['twitch-e-DE-splits.npy'], allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = jt.zeros(sizes, dtype='bool')
        data.val_mask = jt.zeros(sizes, dtype='bool')
        data.test_mask = jt.zeros(sizes, dtype='bool')
        for i, split in enumerate(splits):
            data.train_mask[:, i][jt.Var(split['train']).long()] = True
            data.val_mask[:, i][jt.Var(split['valid']).long()] = True
            data.test_mask[:, i][jt.Var(split['test']).long()] = True
        return data

    def _process_arxiv_year(self):
        dataset = OGBNodePropPredDataset(root=self.root, name='ogbn-arxiv')
        split_path = self.root + "/arxiv-year/raw/arxiv-year-splits.npy"
        data = dataset[0]
        edge_index = jt.Var(data.edge_index)
        x = jt.Var(data.x)
        nclass = 5
        label = even_quantile_labels(
            data.node_year.flatten(), nclass, verbose=False)
        y = jt.Var(label).reshape(-1, 1)
        data = Data(x=x, edge_index=edge_index, y=y)
        splits = np.load(split_path, allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = jt.zeros(sizes, dtype='bool')
        data.val_mask = jt.zeros(sizes, dtype='bool')
        data.test_mask = jt.zeros(sizes, dtype='bool')
        for i, split in enumerate(splits):
            data.train_mask[:, i][jt.Var(split['train']).long()] = True
            data.val_mask[:, i][jt.Var(split['valid']).long()] = True
            data.test_mask[:, i][jt.Var(split['test']).long()] = True
        return data

    def process(self):
        if self.name in self.facebook_datasets:
            data = self._process_facebook()
        elif self.name in ['genius', 'pokec']:
            data = self._process_mat()
        elif self.name == 'deezer-europe':
            data = self._process_deezer_europe()
        elif self.name == 'twitch-gamer':
            data = self._process_gamer()
        elif self.name == 'wiki':
            data = self._process_wiki()
        elif self.name == 'arxiv-year':
            data = self._process_arxiv_year()
        elif self.name == 'snap-patents':
            data = self._process_snap_patents()
        elif self.name == 'twitch-de':
            data = self._process_twitch_de()
        else:
            raise NotImplementedError(
                f"chosen dataset '{self.name}' is not implemented")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}({len(self)})'