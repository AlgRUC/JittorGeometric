'''
Description: 
Author: lusz
Date: 2024-11-16 09:41:10
'''
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
        self.CSR = CSR(column_indices, row_offset, edge_weight)
        
    def save(self, file_path: str):
        """
        保存 GraphChunk 实例为二进制文件
        :param file_path: 保存文件路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        """
        从二进制文件加载 GraphChunk 实例
        :param file_path: 保存的文件路径
        :return: 加载的 GraphChunk 实例
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)
