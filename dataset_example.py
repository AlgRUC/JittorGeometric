import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork,WebKB, OGBNodePropPredDataset, JODIEDataset
import jittor_geometric.transforms as T

jt.flags.use_cuda = 0

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

dataset = args.dataset
if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(path, dataset, geom_gcn_preprocess=False)
elif dataset in ['texas', 'wisconsin', 'cornell']:
    dataset = WebKB(path, dataset)
elif dataset in ['ogbn-arxiv']:
    dataset = OGBNodePropPredDataset(name=dataset, root=path)
elif dataset in ['reddit', 'wikipedia', 'mooc', 'lastfm']:
    dataset = JODIEDataset(path, name=dataset)


data = dataset[0]
print(data)