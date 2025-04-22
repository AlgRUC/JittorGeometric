import jittor as jt
import threading
from typing import List, Dict, Optional
from collections import defaultdict

class InitialEmb:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.initial_embs = []
        self.num_nodes = 0
        self.hidden_dim = 0
        self.device = jt.flags.amp_reg
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = InitialEmb()
        return cls._instance
    
    def initialize(self, num_nodes, hidden_dim, device, initial_embs):
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.device = device
        self.initial_embs = [emb.clone().detach().to(device) for emb in initial_embs]
    
    def clear(self):
        self.initial_embs.clear()
        self.num_nodes = 0
        self.hidden_dim = 0

class DyNodeEmbedding:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.rw_lock = threading.RLock()
        self.m_device = jt.flags.amp_reg
        self.m_hidden_dim = 0
        self.m_num_nodes = 0
        self.m_embedding = []
        self.m_update_count = []
        self.m_update_emb = []
        self.is_merge = False
        self.window_initial = None
        self.window_step = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DyNodeEmbedding()
        return cls._instance

    def inits(self, embs: List[jt.Var]):
        with self.rw_lock:
            self.m_device = jt.flags.amp_reg
            self.m_embedding = [emb.detach().to(self.m_device) for emb in embs]
            self.m_num_nodes = self.m_embedding[0].shape[0]
            self.m_hidden_dim = self.m_embedding[0].shape[1]
            
            self.m_update_count = [defaultdict(int) for _ in range(len(embs))]
            self.m_update_emb = [defaultdict(dict) for _ in range(len(embs))]
            
            initial_emb = InitialEmb.get_instance()
            initial_emb.initialize(self.m_num_nodes, self.m_hidden_dim, 
                                  self.m_device, self.m_embedding)

    def index(self, uid: int, dim: int = 0) -> jt.Var:
        with self.rw_lock:
            update_times = self._get_version_id(dim, uid)
            if update_times != 0:
                return self._index_update(update_times-1, uid, dim)
            else:
                return self.m_embedding[dim][uid].clone()

    def update(self, uid: int, embedding: jt.Var, dim: int = 0):
        with self.rw_lock:
            if not self.is_merge:
                self.is_merge = True
            update_times = self._get_version_id(dim, uid)
            self._update_emb(update_times, uid, dim, embedding.detach())

    def merge(self):
        with self.rw_lock:
            for dim_id in range(len(self.m_embedding)):
                for uid in range(self.m_num_nodes):
                    update_times = self._get_version_id(dim_id, uid)
                    if update_times != 0:
                        self.m_embedding[dim_id][uid] = (
                            self.m_update_emb[dim_id][update_times-1][uid].detach()
                        )
                self.m_update_emb[dim_id].clear()
                self.m_update_count[dim_id].clear()
            self.is_merge = False

    def back_to_initial(self):
        initial_emb = InitialEmb.get_instance()
        self.m_num_nodes = initial_emb.num_nodes
        self.m_hidden_dim = initial_emb.hidden_dim
        self.m_device = initial_emb.device
        self.m_embedding = [emb.clone() for emb in initial_emb.initial_embs]
        self._clear_updates()

    def _get_version_id(self, dim: int, index: int) -> int:
        return self.m_update_count[dim].get(index, 0)

    def _index_update(self, vid: int, nid: int, dim_id: int) -> jt.Var:
        return self.m_update_emb[dim_id][vid][nid].clone()

    def _update_emb(self, vid: int, nid: int, dim_id: int, emb: jt.Var):
        if vid not in self.m_update_emb[dim_id]:
            self.m_update_emb[dim_id][vid] = {}
        self.m_update_emb[dim_id][vid][nid] = emb.to(self.m_device)
        self.m_update_count[dim_id][nid] += 1

    def _clear_updates(self):
        self.m_update_count = [defaultdict(int) for _ in self.m_update_count]
        self.m_update_emb = [defaultdict(dict) for _ in self.m_update_emb]
        self.is_merge = False

if __name__ == "__main__":
    initial_embs = [jt.randn(100, 64) for _ in range(3)]
    print(initial_embs)
    emb_manager = DyNodeEmbedding.get_instance()
    emb_manager.inits(initial_embs)
    
    new_emb = jt.randn(64)
    emb_manager.update(10, new_emb, dim=0)
    
    result = emb_manager.index(10)
    print("Embedding shape:", result.shape)
    
    emb_manager.merge()
    
    emb_manager.back_to_initial()