import pickle
import os.path as osp
import jittor as jt
from typing import Optional
from jittor_geometric.data import CSR, CSC

class DistChunk:
    def __init__(self, 
                 chunks: int,  
                 chunk_id: int, 
                 v_num: int, 
                 global_v_num: int, 
                 edge_index, 
                 offset, 
                 edge_weight = None, 
                 # local_masks: dict = None,
                 local_train_mask = None,
                 local_val_mask = None,
                 local_test_mask = None,
                 local_feature: object = None,  
                 local_label: object = None,
                 num_classes: int = None
                 ):
        
        self.chunks = chunks
        self.chunk_id = chunk_id 
        self.v_num = v_num
        self.global_v_num =  global_v_num

        self.edge_index = jt.array(edge_index) 
        self.edge_weight = jt.array(edge_weight) 
        self.offset = jt.array(offset) 

        self.CSC = None
        self.CSR = None

        self.chunk_CSC = []
        self.chunk_CSR = []

        self.local_masks = {}

        local_train_mask = jt.array(local_train_mask)
        local_val_mask = jt.array(local_val_mask)
        local_test_mask = jt.array(local_test_mask)
        self.local_masks['train'] = local_train_mask
        self.local_masks['val'] = local_val_mask
        self.local_masks['test'] = local_test_mask

        self.local_feature = jt.array(local_feature)
        self.local_label = jt.array(local_label)
        
        # 数据集的label范围
        self.num_classes = num_classes

        # add
        self.source = []
        """
        source[i] = [] -> 当前chunk需要从chunk i给我发的邻居顶点。当前chunk在chunk i的邻居id
        """

    def set_csr(self, column_indices, row_offset, edge_weight=None):
        """
        Set the CSR (Compressed Sparse Row) representation of the graph.
        :param column_indices: Column indices of the non-zero elements.
        :param row_offset: Row offsets for the CSR format.
        :param edge_weight: Optional edge weights.
        """
        column_indices = jt.array(column_indices)
        row_offset = jt.array(row_offset)
        edge_weight = jt.array(edge_weight)
        self.CSR = CSR(column_indices, row_offset, edge_weight)

    # 添加CSC存储
    def set_csc(self, row_indices, column_offset, edge_weight=None):
        """
        Set the CSC (Compressed Sparse Column) representation of the graph.
        :param row_indices: Row indices of the non-zero elements.
        :param column_offset: Column offsets for the CSR format.
        :param edge_weight: Optional edge weights.
        """
        row_indices = jt.array(row_indices)
        column_offset = jt.array(column_offset)
        edge_weight = jt.array(edge_weight)
        self.CSC = CSC(row_indices, column_offset, edge_weight)

    def save(self, file_path: str):
        """
        Save the GraphChunk instance as a binary file.
        :param file_path: Path to the file where the instance will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        """
        Load a GraphChunk instance from a binary file.
        :param file_path: Path to the file from which the instance will be loaded.
        :return: Loaded GraphChunk instance.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod 
    def distributed_file(rank : int, nparts : int,  file_dir : Optional[str]=None):
        """Save the current instance to a binary file."""
        if file_dir is not None:
            return osp.join(file_dir, f"subgraph_{nparts}_of_{rank}")
        else : return None