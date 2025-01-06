import jittor as jt
from jittor import sparse
import copy
from .general_loader import GeneralLoader
from ..utils import induced_graph
from jittor_geometric.partition.chunk_manager import ChunkManager
import numpy as np


class GraphSAINTLoader(GeneralLoader):
    
    r'''
    NOT YET FINISHED!
    '''
    
    def __init__(self, dataset, num_parts: int, mini_splits: int, fixed: bool = False):
        self.data = copy.copy(dataset[0])
        self.N = self.data.num_nodes
        self.E = self.data.num_edges
        self.edge_index = copy.copy(self.data.edge_index)
        self.data.edge_index = None
        
        self.num_parts = num_parts
        self.mini_splits = mini_splits
        self.fixed = fixed
        
        self.itermax = num_parts
        
        self.parts = self.partition(self.edge_index, self.N, self.mini_splits)
        
        self.itercnt = 0
        self.n_splits = self.get_node_indices()
        for i in self.n_splits:
            print(i)
        
    def partition(self, edge_index, num_nodes, mini_splits):
        chunk_manager = ChunkManager(output_dir=None)
        partition = chunk_manager.metis_partition(edge_index, num_nodes, mini_splits)
        partition = jt.Var(partition)
        partition = partition.sort()
        
        parts = []
        part_begin = 0
        part_end = 1
        for i in range(self.mini_splits):
            part_end = part_begin + 1
            while part_end <= num_nodes - 1 and partition[0][part_end] == partition[0][part_end - 1]:
                part_end += 1
            if part_begin >= part_end:
                parts.append(jt.zeros(0, dtype='int'))
            elif part_end >= num_nodes:
                parts.append(partition[1][part_begin:])
            else:
                parts.append(partition[1][part_begin: part_end])
            part_begin = part_end
        return parts
    
    def get_node_indices(self):
        n_id = np.random.permutation(self.mini_splits) % self.num_parts
        n_ids = [jt.nonzero((n_id == i)).view(-1) for i in range(self.num_parts)]
        return n_ids

    def __reset__(self):
        self.itercnt = 0
        if self.fixed == False:
            self.n_ids = self.get_node_indices()
    
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.itercnt < self.itermax:
            
            node_id = jt.zeros(0, dtype='int')
            for i in self.n_splits[self.itercnt]:
                node_id = jt.concat([node_id, self.parts[i]])
                
            node_map, edge_id = induced_graph(self.edge_index, node_id, self.N)
            
            # node_mask = jt.zeros(self.N, dtype='bool')
            # node_mask[node_id] = True
            # edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
            # edge_id = jt.nonzero(edge_mask).view(-1)
            
            data = self.data.__class__()
            
            # node_map = jt.zeros(self.N, dtype='int32')
            # node_map[node_id] = jt.arange(0, node_id.size(0))
            data.edge_index = node_map[self.edge_index[:, edge_id]]
            
            for key, item in self.data:
                if key in ['num_nodes']:
                    data[key] = node_id.size(0)
                elif isinstance(item, jt.Var) and item.size(0) == self.N:
                    data[key] = item[node_id]
                elif isinstance(item, jt.Var) and item.size(0) == self.E:
                    data[key] = item[edge_id]
                else:
                    data[key] = item
                    
            self.itercnt += 1
            return data
        
        else:
            self.__reset__()
            raise StopIteration