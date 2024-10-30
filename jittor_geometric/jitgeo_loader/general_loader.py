import jittor as jt
from jittor_geometric.data import InMemoryDataset
from typing import Optional

#class GeneralIterator

class GeneralLoader:
    dataset: InMemoryDataset
    itercnt: int
    itermax: int
    
    def __init__(self, 
                 dataset: InMemoryDataset, 
                 shuffle: Optional[bool] = None):
        self.itercnt = 0
        self.itercnt = 0
        self.dataset = dataset
    
    def __reset__(self):
        self.itercnt = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.itercnt == 0:
            self.itercnt += 1
            return self.dataset[0]
        else:
            self.__reset__()
            raise StopIteration
        
    
    # def __collate__(self):
    #     raise NotImplementedError