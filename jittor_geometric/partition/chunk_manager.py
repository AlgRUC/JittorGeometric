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
        初始化 ChunkManager。
        :param output_dir: 文件保存路径。
        :param graph_data: 原始图数据，可选（仅在分区时需要）。
        """
        self.output_dir = output_dir
        self.graph_data = graph_data
        if self.output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def metis_partition(self, edge_index, num_nodes, num_parts):
        """
        使用 Metis 对图进行划分，并保存分区文件。
        :param edge_index: 图的边索引 (Var 或 np.ndarray)。
        :param num_nodes: 图的节点数。
        :param num_parts: 划分的子图数量。
        :return: 分区信息。
        """
        adj_list = self._edge_index_to_adj_list(edge_index, num_nodes)
        _, partition = part_graph(nparts=num_parts, adjacency=adj_list)
        partition = np.array(partition)

        # 保存分区文件
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
        将边索引转换为邻接表。
        :param edge_index: 图的边索引。
        :param num_nodes: 节点数量。
        :return: 邻接表表示的图。
        """
        adj_list = [[] for _ in range(num_nodes)]
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src != dst:  # 忽略自环
                adj_list[src].append(dst)
        return adj_list
