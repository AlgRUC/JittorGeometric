import jittor as jt
from jittor import sparse
import copy
from .general_loader import GeneralLoader
from ..utils import induced_graph

class RandomNodeLoader(GeneralLoader):
    
    r'''
    The graph dataset loader, randomly split all of the nodes into 'num_parts' mini-batches.
    The dataset loader yields the induced graph of the selected nodes iteratively.
    
    Args:
        dataset (InMemoryDataset): The original graph dataset.
        num_parts (int): Number of expected mini-batches.
        fixed (bool, optional): If set to 'True', the dataset loader will yield identical mini-batches every round.
    '''
    
    
    def __init__(self, dataset, num_parts: int, fixed: bool = True):
        self.data = copy.copy(dataset[0])
        self.N = self.data.num_nodes
        self.E = self.data.num_edges
        self.edge_index = copy.copy(self.data.edge_index)
        self.data.edge_index = None
        
        self.num_parts = num_parts
        self.fixed = fixed
        
        self.itermax = num_parts
        
        self.itercnt = 0
        self.n_ids = self.get_node_indices()
        
    def get_node_indices(self):
        n_id = jt.randint(0, self.num_parts, (self.N, ), dtype="int32")
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
            
            node_id = None
            
            while True:
                node_id = self.n_ids[self.itercnt]    
                if node_id.size(0) != 0:
                    break
                else:
                    self.itercnt += 1
                    if self.itercnt >= self.itermax:
                        self.__reset__()
                        raise StopIteration
            
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