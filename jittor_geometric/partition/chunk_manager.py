'''
Description: 
Author: lusz
Date: 2024-12-03 15:49:39
'''
import os
import os.path as osp
from typing import List, Optional
#from jittor_geometric.data import GraphChunk,CSR
#from jittor_geometric.ops import cootocsr
import pickle
import numpy as np
from pymetis import part_graph

class ChunkManager:
    def __init__(self, output_dir : Optional[str]=None, graph_data=None):
        """
        Initialize the ChunkManager.
        :param output_dir: Path for saving files.
        :param graph_data: Original graph data, optional (only needed for partitioning).
        """
        self.output_dir = output_dir
        self.graph_data = graph_data
        if self.output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def metis_partition(self, edge_index, num_nodes, num_parts):
        """
        Perform graph partitioning using Metis and save partition files.
        :param edge_index: Edge indices of the graph (Var or np.ndarray).
        :param num_nodes: Number of nodes in the graph.
        :param num_parts: Number of subgraphs to partition into.
        :return: Partition information.
        """
        adj_list = self._edge_index_to_adj_list(edge_index, num_nodes)
        _, partition = part_graph(nparts=num_parts, adjacency=adj_list)
        partition = np.array(partition)

        # Save partition file
        if self.output_dir is not None:
            partition_file = osp.join(self.output_dir, f"partition_{num_parts}.bin")
            with open(partition_file, 'wb') as f:
                pickle.dump(partition, f)
            print(f"Partition file saved to {partition_file}")
        return partition

    def partition_to_chunk(self, partition_file, edge_index, edge_weight, num_nodes, num_parts):
        return

    @staticmethod
    def _edge_index_to_adj_list(edge_index, num_nodes):
        """
        Convert edge indices to adjacency list.
        :param edge_index: Edge indices of the graph.
        :param num_nodes: Number of nodes in the graph.
        :return: Graph represented as an adjacency list.
        """
        adj_list = [[] for _ in range(num_nodes)]
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src != dst:  # Ignore self-loops
                adj_list[src].append(dst)
        return adj_list
