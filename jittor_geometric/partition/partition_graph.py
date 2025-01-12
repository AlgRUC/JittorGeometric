'''
Author: lusz
Date: 2024-11-15 15:28:04
Description: Load and preprocess GNN datasets with relative paths.
'''
import os.path as osp
import argparse
import pickle
import os
import sys
import jittor as jt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#from jittor_geometric.data import GraphChunk,CSR
from jittor import Var
from jittor_geometric.partition.chunk_manager import ChunkManager
from jittor_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, OGBNodePropPredDataset, HeteroDataset, Reddit
import jittor_geometric.transforms as T
from pymetis import part_graph
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true', help='Use GDC preprocessing.')
parser.add_argument('--dataset', type=str, required=True, help='Name of the GNN dataset to load.')
parser.add_argument('--num_parts', type=int, required=True, help='partition number')
args = parser.parse_args()

script_dir = osp.dirname(osp.realpath(__file__))
path = osp.join(script_dir, '..', '..', 'data')
dataset_name = args.dataset


def edge_index_to_adj_list(edge_index, num_nodes):
    """Convert edge_index to adjacency list."""
    adj_list = [[] for _ in range(num_nodes)]
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src != dst:
            adj_list[src].append(dst)
    return adj_list

dataset=args.dataset
# Load dataset
if dataset in ['computers', 'photo']:
    dataset = Amazon(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
elif dataset in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(path, dataset, geom_gcn_preprocess=False)
elif dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
    dataset = OGBNodePropPredDataset(name=dataset, root=path)
elif dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
    dataset = HeteroDataset(path, dataset)
elif dataset in ['reddit']:
    dataset = Reddit(os.path.join(path, 'Reddit'))
data = dataset[0]
edge_index = data.edge_index.numpy()
num_nodes = data.x.shape[0]

# metis partition
reorder_dir = osp.join(path, "reorder", f"{dataset_name}_{args.num_parts}part")
chunk_manager = ChunkManager(output_dir=reorder_dir)
partition = chunk_manager.metis_partition(edge_index, num_nodes,args.num_parts)
partition = np.array(partition)
os.makedirs(reorder_dir, exist_ok=True)
binary_file_path = osp.join(reorder_dir, f"{dataset_name}_partition_{args.num_parts}.bin")
if not osp.exists(binary_file_path):
    with open(binary_file_path, 'wb') as f:
        pickle.dump(partition, f)
    print("Partition file saved.")
else:
    print("Partition file already exists. Skipping.")

