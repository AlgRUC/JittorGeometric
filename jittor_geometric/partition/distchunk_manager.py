import os
import sys
import os.path as osp
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
sys.path.append([str(Path(__file__).resolve().parent)])
from jittor_geometric.data import DistChunk,CSR
import pickle
from pymetis import part_graph

class DistChunkManager:
    
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
    
    
    def partition_to_chunk(self, partition_file, edge_index, num_nodes, num_parts, edge_weight = None,
                           global_feature = None, global_masks = None, global_label = None, num_classes = None):
        """
            Partition the graph into multiple chunks and create a GraphChunk object for each chunk.
            :param partition_file: File path containing partition information
            :param edge_index: Edge index (two arrays representing the start and end nodes of edges)
            :param edge_weight: Weights of the edges
            :param num_nodes: Total number of nodes in the graph
            :param num_parts: Number of chunks to partition the graph into
            :param global_feature
            :param global_masks
            :param global_label
            :param num_classes
            :return: List of GraphChunk objects
        """
        # load
        with open(partition_file, 'rb') as f:
            partition = pickle.load(f)
        print(f"Partition file loaded from {partition_file}")
        # Remap vertex indices according to the partition
        node_sorted_indices = np.argsort(partition)
        remap = np.zeros(num_nodes, dtype=np.int64)
        remap[node_sorted_indices] = np.arange(num_nodes)
        edge_index[0] = remap[edge_index[0]]
        edge_index[1] = remap[edge_index[1]]

        # Resort edge_index and edge_weight
        sorted_indices = np.lexsort((edge_index[1], edge_index[0]))
        edge_index = edge_index[:, sorted_indices]
        edge_weight = edge_weight[sorted_indices] if edge_weight is not None else None

        # other information
        global_feature = global_feature[node_sorted_indices] if global_feature is not None else None
        global_label = global_label[node_sorted_indices] if global_label is not None else None
        reordered_masks = [None, None, None]
        
        # mask_names = ['train', 'val', 'test']
        for i in range(min(len(global_masks), 3)):
                 mask = global_masks[i]
                 if mask is not None:
                    reordered_masks[i] = np.asarray(mask)[node_sorted_indices]

        # Calculate Partition Offsets
        partition_offset = [0]
        for part_id in range(num_parts):
            partition_offset.append(np.sum(partition == part_id) + partition_offset[-1])
        partition_offset = np.array(partition_offset)
        
        chunks = []
        for part_id in range(num_parts):
            start, end = partition_offset[part_id], partition_offset[part_id + 1]
            chunk_nodes = range(start, end)   # Nodes included in the current chunk
            # 目的顶点在节点，边就在，包括边和对应权重
            chunk_edges = np.isin(edge_index[1], chunk_nodes)
            # 挑出其中目标顶点在part_id的边
            chunk_edge_index = edge_index[:, chunk_edges] # 由于重编号后经过排序,edge_index按照先定点，后终点，抽取后仍成立
            chunk_edge_weight = edge_weight[chunk_edges] if edge_weight is not None else None # 边权也重排列过
            
            # 图拓扑存的是本机的edge_index（使用local_id）
            chunk_edge_index[1] = chunk_edge_index[1] - start # 目的顶点是本地化边的id

            # 处理feature信息，Mask信息，lebal信息 (利用offset进行切片)
            chunk_feature = global_feature[start: end] if global_feature is not None else None
            # chunk_feature = global_feature[remap[start:end]] if global_feature is not None else None
            chunk_feature = np.array(chunk_feature)
            chunk_label = global_label[start: end] if global_label is not None else None
            chunk_train_mask = reordered_masks[0][start:end] if reordered_masks[0] is not None else None
            chunk_val_mask   = reordered_masks[1][start:end] if reordered_masks[1] is not None else None
            chunk_test_mask  = reordered_masks[2][start:end] if reordered_masks[2] is not None else None
            
            # Create a DistChunk object 
            graph_chunk = DistChunk(
                chunks=num_parts,
                chunk_id=part_id,
                v_num=end - start,
                global_v_num=num_nodes,
                edge_index=chunk_edge_index,
                edge_weight=chunk_edge_weight,
                offset=partition_offset,
                local_train_mask=chunk_train_mask,
                local_val_mask=chunk_val_mask,
                local_test_mask=chunk_test_mask,
                local_label=chunk_label,
                local_feature=chunk_feature,
                num_classes=num_classes
            )

            chunk_csr = DistChunkManager._edge_index_weight_to_csr(num_nodes, 
                                                                   end - start, 
                                                                   chunk_edge_index, 
                                                                   chunk_edge_weight)
        
            graph_chunk.set_csr(
            column_indices=chunk_csr.indices,
            row_offset=chunk_csr.indptr,
            edge_weight=chunk_csr.data
            )
            
            chunk_csc = DistChunkManager._edge_index_weight_to_csc(num_nodes, 
                                                               end - start, 
                                                               chunk_edge_index, 
                                                               chunk_edge_weight)

            graph_chunk.set_csc(
                row_indices=chunk_csc.indices,
                column_offset=chunk_csc.indptr,
                edge_weight=chunk_csc.data
            )
            chunks.append(graph_chunk)

        return chunks

    @staticmethod
    def _edge_index_weight_to_csr(num_rows, num_cols, edge_index, edge_weight=None):
        adjacency_matrix = csr_matrix((edge_weight, (edge_index[0], edge_index[1])),
                                  shape=(num_rows, num_cols))
        return adjacency_matrix

    @staticmethod
    def _edge_index_weight_to_csc(num_rows, num_cols, edge_index, edge_weight=None):
        adjacency_matrix = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                                  shape=(num_rows, num_cols))
        return adjacency_matrix

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
    
    def chunks_save(self, chunks, num_parts):
        for part_id in range(num_parts):
            save_file = osp.join(self.output_dir, f"subgraph_{num_parts}_of_{part_id}")
            with open(save_file, 'wb') as f:
                pickle.dump(chunks[part_id], f)
