import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

@dataclass
class EdgeData:
    value: float = 1.0

    def to_string(self) -> str:
        return str(self.value)

@dataclass
class TEdge:
    src: int
    dst: int
    edge_data: EdgeData = None
    time: datetime = None

    def __post_init__(self):
        if self.edge_data is None:
            self.edge_data = EdgeData()
        if self.time is None:
            self.time = datetime.now()

    def __repr__(self) -> str:
        return f"( {self.src}, {self.dst} )"

class DanceTEdge:
    def __init__(self, initial_size: int = 20):
        self.ids: List[int] = []
        self.edge_datas: List[EdgeData] = []
        self.offset: Dict[int, int] = {}

    def is_edge(self, nid: int) -> bool:
        return nid in self.offset

    def is_exist(self, nids: List[int]) -> bool:
        for nid in nids:
            if nid in self.offset:
                print(f"{nid} already exists! Insert error!")
                return True
        return False

    def set_map(self) -> None:
        self.offset = {nid: i for i, nid in enumerate(self.ids)}

    def add_neighbors(self, nids: List[int], edge_datas: Optional[List[EdgeData]] = None) -> None:
        if self.is_exist(nids):
            return
        if edge_datas is None:
            edge_datas = [EdgeData()] * len(nids)
        current_size = len(self.ids)
        self.ids.extend(nids)
        self.edge_datas.extend(edge_datas)
        for i, nid in enumerate(nids):
            self.offset[nid] = current_size + i

    def add_neighbor(self, nid: int, edge_data: Optional[EdgeData] = None) -> None:
        if nid in self.offset:
            return
        if edge_data is None:
            edge_data = EdgeData()
        self.offset[nid] = len(self.ids)
        self.ids.append(nid)
        self.edge_datas.append(edge_data)

    @property
    def size(self) -> int:
        return len(self.ids)

    def get_ids(self) -> List[int]:
        return self.ids

    def get_edge_datas(self) -> List[EdgeData]:
        return self.edge_datas

class AdjTableNodes:
    def __init__(self):
        self.in_neighbors = DanceTEdge()
        self.out_neighbors = DanceTEdge()

    def add_out_edge(self, dst_id: int, edge_data: Optional[EdgeData] = None) -> None:
        self.out_neighbors.add_neighbor(dst_id, edge_data)

    def add_in_edge(self, src_id: int, edge_data: Optional[EdgeData] = None) -> None:
        self.in_neighbors.add_neighbor(src_id, edge_data)

    def add_in_edges(self, src_ids: List[int], edge_datas: Optional[List[EdgeData]] = None) -> None:
        self.in_neighbors.add_neighbors(src_ids, edge_datas)

    def add_out_edges(self, dst_ids: List[int], edge_datas: Optional[List[EdgeData]] = None) -> None:
        self.out_neighbors.add_neighbors(dst_ids, edge_datas)

    def has_in_edge(self, src: int) -> bool:
        return self.in_neighbors.is_edge(src)

    def has_out_edge(self, dst: int) -> bool:
        return self.out_neighbors.is_edge(dst)

    def in_edges_data(self) -> List[EdgeData]:
        return self.in_neighbors.get_edge_datas()

    def in_neighbors_ids(self) -> List[int]:
        return self.in_neighbors.get_ids()

    def out_edges_data(self) -> List[EdgeData]:
        return self.out_neighbors.get_edge_datas()

    def out_neighbors_ids(self) -> List[int]:
        return self.out_neighbors.get_ids()

    def num_in_neighbors(self) -> int:
        return self.in_neighbors.size

    def num_out_neighbors(self) -> int:
        return self.out_neighbors.size

class DiAdjList:
    def __init__(self):
        self.vertex_lists: List[AdjTableNodes] = []
        self.lock = threading.Lock()

    def add_vertexes(self, num_vertexes: int) -> None:
        with self.lock:
            self.vertex_lists.extend([AdjTableNodes() for _ in range(num_vertexes)])

    def create_base_edges(self, edges: List[TEdge]) -> None:
        src_edges: Dict[int, Tuple[List[int], List[EdgeData]]] = {}
        dst_edges: Dict[int, Tuple[List[int], List[EdgeData]]] = {}
        srcs_set: Set[int] = set()
        dsts_set: Set[int] = set()

        for edge in edges:
            src, dst = edge.src, edge.dst
            srcs_set.add(src)
            dsts_set.add(dst)
            
            if src not in src_edges:
                src_edges[src] = ([], [])
            src_edges[src][0].append(dst)
            src_edges[src][1].append(edge.edge_data)
            
            if dst not in dst_edges:
                dst_edges[dst] = ([], [])
            dst_edges[dst][0].append(src)
            dst_edges[dst][1].append(edge.edge_data)

        all_nodes = srcs_set.union(dsts_set)
        max_node = max(all_nodes) if all_nodes else 0
        current_len = len(self.vertex_lists)
        if max_node >= current_len:
            with self.lock:
                current_len = len(self.vertex_lists)
                if max_node >= current_len:
                    self.add_vertexes(max_node - current_len + 1)

        with ThreadPoolExecutor() as executor:
            futures = []
            for src, (dsts, datas) in src_edges.items():
                adj_node = self.vertex_lists[src]
                futures.append(executor.submit(adj_node.add_out_edges, dsts, datas))
            for dst, (srcs, datas) in dst_edges.items():
                adj_node = self.vertex_lists[dst]
                futures.append(executor.submit(adj_node.add_in_edges, srcs, datas))
            for future in futures:
                future.result()

    def add_edge(self, edge: TEdge) -> None:
        max_id = max(edge.src, edge.dst)
        with self.lock:
            current_len = len(self.vertex_lists)
            if max_id >= current_len:
                self.add_vertexes(max_id - current_len + 1)
        
        with ThreadPoolExecutor() as executor:
            future_out = executor.submit(self.vertex_lists[edge.src].add_out_edge, edge.dst, edge.edge_data)
            future_in = executor.submit(self.vertex_lists[edge.dst].add_in_edge, edge.src, edge.edge_data)
            future_out.result()
            future_in.result()

    def out_edges(self, n_id: int) -> List[TEdge]:
        adj_node = self.vertex_lists[n_id]
        return [TEdge(n_id, dst, data) for dst, data in zip(adj_node.out_neighbors_ids(), adj_node.out_edges_data())]

    def in_edges(self, n_id: int) -> List[TEdge]:
        adj_node = self.vertex_lists[n_id]
        return [TEdge(src, n_id, data) for src, data in zip(adj_node.in_neighbors_ids(), adj_node.in_edges_data())]

    def edges(self, n_id: int) -> List[TEdge]:
        return self.in_edges(n_id) + self.out_edges(n_id)

    def out_neighbors(self, n_id: int) -> List[int]:
        return self.vertex_lists[n_id].out_neighbors_ids()

    def in_neighbors(self, n_id: int) -> List[int]:
        return self.vertex_lists[n_id].in_neighbors_ids()

    def has_edge(self, sid: int, did: int) -> bool:
        if sid >= len(self.vertex_lists) or did >= len(self.vertex_lists):
            return False
        return self.vertex_lists[sid].has_out_edge(did) and self.vertex_lists[did].has_in_edge(sid)

    def degree(self, uid: int) -> int:
        return self.in_degree(uid) + self.out_degree(uid)

    def in_degree(self, uid: int) -> int:
        return self.vertex_lists[uid].num_in_neighbors()

    def out_degree(self, uid: int) -> int:
        return self.vertex_lists[uid].num_out_neighbors()

    def clear(self) -> None:
        self.vertex_lists.clear()
        self.vertex_lists = []