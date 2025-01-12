
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import jittor as jt
from jittor import sparse
import copy
from .general_loader import GeneralLoader
from ..utils import neighbor_sampler

class NeighborLoader(GeneralLoader):
    
    r'''
    The graph dataset loader, samples the neighbors of source nodes for graph convolution iteratively and randomly.
    The dataset loader yields all the tuples of (source_node, sampled neighbor) as the graph edge.
    The yielded data block will contain the node map and node index.
    
    Args:
        dataset (InMemoryDataset): The original graph dataset.
        source_node (Var): The list of source_node. 
        num_neighbors (List[int]): Number of sampled neighbors per layer. For example, [2, 3] represents sample 2 neighbors, and 3 neighbors of each sampled ones.
        batch_size (int): Size of each mini-batch
        fixed (bool, optional): If set to 'True', the dataset loader will yield identical mini-batches every round.
    '''
    
    
    def __init__(
        self, 
        dataset, 
        source_node : jt.Var = None,
        num_neighbors : List[int] = None,
        batch_size : int = None,
        fixed : bool = False
    ):
        self.data = copy.copy(dataset[0])
        self.N = self.data.num_nodes
        self.E = self.data.num_edges
        self.edge_index = copy.copy(self.data.edge_index)
        
        self.source_node = source_node
        self.data.edge_index = None
        
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.fixed = fixed
        
        self.itercnt = 0
        self.itermax = int((source_node.size(0) - 1) / batch_size) + 1
        
        self.n_ids = self.get_node_indices()
        # print(self.n_ids)
        # print(self.edge_index[:,0:15])
        
        self.row_offsets = jt.zeros((self.N+1, ),dtype='int')
        for i in range(self.E - 1, -1, -1):
            self.row_offsets[self.edge_index[0,i]] = i
        self.row_offsets[-1] = self.E
        for i in range(self.N - 1, -1, -1):
            if self.row_offsets[i] == 0:
                self.row_offsets[i] = self.row_offsets[i+1]
        # print(self.row_offsets[0:5], self.row_offsets[-1])
        self.row_range = self.row_offsets[1:] - self.row_offsets[:-1]
        for i in range(self.N):
            if self.row_range[i] == 0:
                self.row_range[i] = self.E
                
        # testsubgraph = self.neighbor_sampling(self.n_ids[0])
        # print(testsubgraph)
        
    def get_node_indices(self):
        n_id = np.random.permutation(self.source_node.size(0))
        # print(n_id)
        n_ids = []
        for i in range(self.itermax):
            n_ids.append(self.source_node[n_id[i * self.batch_size : min((i+1) * self.batch_size, self.source_node.size(0))]])
        return n_ids

    def neighbor_sampling(self, source_nodes):
        results = jt.empty((2, 0), dtype=int)
        for i in self.num_neighbors:
            source_nodes = jt.repeat_interleave(source_nodes, i)
            target_nodes = neighbor_sampler(self.edge_index, self.row_offsets, self.row_range, source_nodes, self.N, self.E)
            # idx = jt.randint_like(source_nodes, 0, self.E)
            # idx = (idx % self.row_range[source_nodes] + self.row_offsets[source_nodes]) % self.E
            # target_nodes = self.edge_index[1][idx]
            edges = jt.stack([target_nodes, source_nodes])
            
            new_results = jt.empty((2, results.shape[1] + edges.shape[1]), dtype=int)
            if results.shape[1] != 0:
                new_results[:, :results.shape[1]] = results
            new_results[:, results.shape[1]:] = edges
            results = new_results
            source_nodes = target_nodes
            
            # print(results)              
            
        return results
        
    def __reset__(self):
        self.itercnt = 0
        if self.fixed == False:
            self.n_ids = self.get_node_indices()
            
    
    def __iter__(self):
        return self
        
    def unique(self, x):
        res = [x[0].item()]
        for i in range(1, x.shape[0]):
            if x[i].item() != x[i-1].item():
                res.append(x[i].item())
        res = jt.Var(res)
        return res
        
    def __next__(self):
        if self.itercnt < self.itermax:
            central_nodes = self.n_ids[self.itercnt]
            edges = self.neighbor_sampling(central_nodes)
            
            node_id = self.unique(jt.sort(jt.flatten(edges))[0])
            node_mask = jt.zeros(self.N, dtype='bool')
            node_mask[node_id] = True
            # edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
            # edge_id = jt.nonzero(edge_mask).view(-1)
            
            data = self.data.__class__()
            
            node_map = jt.zeros(self.N, dtype='int')
            node_map[node_id] = jt.arange(0, node_id.size(0))
            
            data.node_map = node_id
            data.edge_index = node_map[edges]
            data.central_nodes = node_map[central_nodes]
            
            for key, item in self.data:
                if key in ['num_nodes']:
                    data[key] = node_id.size(0)
                elif isinstance(item, jt.Var) and item.size(0) == self.N:
                    data[key] = item[node_id]
                elif isinstance(item, jt.Var) and item.size(0) == self.E:
                    data[key] = None
                else:
                    data[key] = item
                    
            self.itercnt += 1
            return data
        
        else:
            self.__reset__()
            raise StopIteration