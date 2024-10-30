import os.path as osp
import argparse

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid # 利用Planetoid类可以处理三个数据集，分别为“Cora”、“CiteSeer”和“PubMed”
import jittor_geometric.transforms as T
from jittor_geometric.nn import GCNConv, ChebConv, SGConv, GCN2Conv
# add by lusz
import time

from jittor_geometric.jitgeo_loader import RandomNodeLoader

dataset = 'Cora'
# dataset='PubMed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

print(data.edge_index)
print(data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum())


DataLoader = RandomNodeLoader(dataset, num_parts=2, fixed=False)
print(data.edge_attr)


for j in range(0, 2):
    for i in DataLoader:
        print("----------")
        for key, item in i:
            print(key, item)
            if isinstance(item, jt.Var):
                print(item.size())


