from typing import List
from jittor_geometric.data import CSR
import pickle

class GraphChunk:
    def __init__(self, 
                 chunks: int, 
                 chunk_id: int, 
                 v_num: int, 
                 global_v_num: int,
                 local_mask: dict = None,  
                 local_feature: object = None,  
                 local_label: object = None):  
        
        self.chunks = chunks
        self.chunk_id = chunk_id 
        self.v_num = v_num
        self.global_v_num =  global_v_num
        self.CSR = None
        self.local_mask = local_mask  
        self.local_feature = local_feature
        self.local_label = local_label 

    def set_csr(self, column_indices, row_offset, edge_weight=None):
        """
        Set the CSR (Compressed Sparse Row) representation of the graph.
        :param column_indices: Column indices of the non-zero elements.
        :param row_offset: Row offsets for the CSR format.
        :param edge_weight: Optional edge weights.
        """
        self.CSR = CSR(column_indices, row_offset, edge_weight)
        
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
