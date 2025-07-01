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


def partition_graph(dataset_name, num_parts, use_gdc=False):
    """Partition a graph dataset using METIS."""
    script_dir = osp.dirname(osp.realpath(__file__))
    path = osp.join(script_dir, '..', '..', 'data')

    # Load dataset
    if dataset_name in ['computers', 'photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, dataset_name, geom_gcn_preprocess=False)
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = OGBNodePropPredDataset(name=dataset_name, root=path)
    elif dataset_name in ['roman_empire', 'amazon_ratings', 'minesweeper', 'questions', 'tolokers']:
        dataset = HeteroDataset(path, dataset_name)
    elif dataset_name in ['reddit']:
        dataset = Reddit(os.path.join(path, 'Reddit'))
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    num_nodes = data.x.shape[0]

    # METIS partition
    reorder_dir = osp.join(path, "reorder", f"{dataset_name}_{num_parts}part")
    chunk_manager = ChunkManager(output_dir=reorder_dir)
    partition = chunk_manager.metis_partition(edge_index, num_nodes, num_parts)
    partition = np.array(partition)
    os.makedirs(reorder_dir, exist_ok=True)
    binary_file_path = osp.join(reorder_dir, f"{dataset_name}_partition_{num_parts}.bin")
    if not osp.exists(binary_file_path):
        with open(binary_file_path, 'wb') as f:
            pickle.dump(partition, f)
        print("Partition file saved.")
    else:
        print("Partition file already exists. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC preprocessing.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the GNN dataset to load.')
    parser.add_argument('--num_parts', type=int, required=True, help='Partition number.')
    args = parser.parse_args()
    partition_graph(args.dataset, args.num_parts, args.use_gdc)