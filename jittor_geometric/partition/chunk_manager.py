'''
Description: 
Author: lusz
Date: 2024-12-03 15:49:39
'''
import os
import os.path as osp
from typing import List, Optional
from jittor_geometric.data import GraphChunk,CSR
from jittor_geometric.ops import cootocsr
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
        """
        将图划分为多个 chunk，并为每个 chunk 创建 GraphChunk 对象。
        
        :param partition_file: 包含分区信息的文件路径
        :param edge_index: 边索引（两个数组分别表示边的起点和终点）
        :param edge_weight: 边的权重
        :param num_nodes: 图中顶点的总数
        :param num_parts: 图划分的 chunk 数量
        :return: GraphChunk 对象列表
        """
        with open(partition_file, 'rb') as f:
            partition = pickle.load(f)
        print(f"Partition file loaded from {partition_file}")
        
        # 按分区重新映射顶点编号
        sorted_indices = np.argsort(partition)
        remap = np.zeros(num_nodes, dtype=np.int64)
        remap[sorted_indices] = np.arange(num_nodes)
        edge_index[0] = remap[edge_index[0]]
        edge_index[1] = remap[edge_index[1]]
        
        # 计算每个分区的起始和结束位置
        partition_offset = [0]
        for part_id in range(num_parts):
            partition_offset.append(np.sum(partition == part_id) + partition_offset[-1])
        partition_offset = np.array(partition_offset)
        
        chunks = []
        
        for part_id in range(num_parts):
            start, end = partition_offset[part_id], partition_offset[part_id + 1]
            chunk_nodes = range(start, end)  # 当前 chunk 包含的节点
            chunk_edges = np.isin(edge_index[0], chunk_nodes) & np.isin(edge_index[1], chunk_nodes)
            
            # 提取边信息
            chunk_edge_index = edge_index[:, chunk_edges]
            chunk_edge_weight = edge_weight[chunk_edges] if edge_weight is not None else None
            
            # 创建 GraphChunk
            graph_chunk = GraphChunk(
                chunks=num_parts,
                chunk_id=part_id,
                v_num=end - start,
                global_v_num=num_nodes
            )
            graph_chunk.set_csr(
                column_indices=chunk_edge_index[1],
                row_offset=np.cumsum(np.bincount(chunk_edge_index[0], minlength=end - start)),
                edge_weight=chunk_edge_weight
            )
            chunks.append(graph_chunk)
        
        return chunks
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
