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
                 chunk_offset: List[int], 
                 chunks: int, 
                 chunk_id: int,  # 初始化时传入 chunk_id
                 device_num: int,
                 device_id: int,
                 local_mask: dict = None,  # 分区内的 mask 信息
                 local_embedding: object = None,  # 分区内的顶点特征
                 local_label: object = None):  # 分区内的标签信息
        """
        :param chunk_offset: 子图顶点的偏移列表
        :param chunks: 子图的总数
        :param chunk_id: 当前子图的 ID
        :param device_num: GPU 的总数（或设备数）
        :param device_id: 当前子图分配到的设备
        :param local_mask: 本分区的 mask 信息
        :param local_embedding: 本分区的嵌入信息
        :param local_label: 本分区的标签信息
        """
        self.chunk_offset = chunk_offset
        self.chunks = chunks
        self.chunk_id = chunk_id 
        self.gpus = device_num
        self.gpu_id = device_id

        self.owned_vertices = chunk_offset[chunk_id + 1] - chunk_offset[chunk_id]
        self.global_vertices = chunk_offset[chunks]


        self.CSR_list = [None] * chunks
        self.local_mask = local_mask  
        self.local_embedding = local_embedding 
        self.local_label = local_label  #

    def set_csr(self, column_indices, row_offset, edge_weight=None, target_chunk_id=None):
        """
        设置指定子图的 CSR 结构
        :param column_indices: CSR 的列索引
        :param row_offset: CSR 的行偏移
        :param edge_weight: CSR 的边权重
        :param target_chunk_id: 目标子图的 ID,默认为当前子图
        """
        if target_chunk_id is None:
            target_chunk_id = self.chunk_id
        self.CSR_list[target_chunk_id] = CSR(column_indices, row_offset, edge_weight)

    def get_csr_by_chunk_id(self, chunk_id: int) -> CSR:
        """
        获取指定子图的 CSR 结构
        :param chunk_id: 目标子图的 ID
        :return: 对应子图的 CSR 结构
        """
        return self.CSR_list[chunk_id]

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
