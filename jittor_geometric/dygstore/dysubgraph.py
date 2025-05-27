from datetime import datetime
from typing import Set, List, Dict
from concurrent.futures import ThreadPoolExecutor
from .adjlist import DiAdjList, TEdge, EdgeData

class DySubGraph:
    def __init__(self, 
                 u: int, 
                 v: int, 
                 event_type: int, 
                 timestamp: datetime,
                 di_adjlist: DiAdjList,
                 hop: int = 1):
        """
        Dynamic subgraph constructor
        
        Parameters:
        u: Source node ID
        v: Target node ID
        event_type: Event type (used for EdgeData)
        timestamp: Event timestamp
        di_adjlist: Reference to the global adjacency list
        hop: Number of hops for neighbor expansion
        """
        # Basic attributes
        self.u = u
        self.v = v
        self.event_type = event_type
        self.timestamp = timestamp
        self.di_adjlist = di_adjlist
        self.hop = hop
        
        # Subgraph metadata
        self.node_set: Set[int] = {u, v}
        self.time_map: Dict[int, datetime] = {u: timestamp, v: timestamp}
        
        # Initialize graph structure
        self._add_base_edges()
        self._expand_subgraph()

    def _add_base_edges(self):
        """Add base edges and update the global adjacency list"""
        # Add bidirectional edges
        for src, dst in [(self.u, self.v), (self.v, self.u)]:
            edge = TEdge(
                src=src,
                dst=dst,
                edge_data=EdgeData(value=1.0),
                time=self.timestamp
            )
            self.di_adjlist.add_edge(edge)

    def _expand_subgraph(self):
        """Recursively expand the subgraph structure"""
        current_frontier = {self.u, self.v}
        for _ in range(self.hop):
            new_frontier = set()
            with ThreadPoolExecutor() as executor:
                # Parallel neighbor retrieval
                futures = []
                for node in current_frontier:
                    futures.append(executor.submit(self._get_neighbors, node))
                
                # Process neighbors
                for future in futures:
                    out_nbrs, in_nbrs = future.result()
                    for nbr in out_nbrs + in_nbrs:
                        if nbr not in self.node_set:
                            new_frontier.add(nbr)
                            self.node_set.add(nbr)
                            self.time_map[nbr] = self.timestamp
            current_frontier = new_frontier

    def _get_neighbors(self, node: int) -> tuple[List[int], List[int]]:
        """Thread-safe method to retrieve neighbors"""
        return (
            self.di_adjlist.out_neighbors(node),
            self.di_adjlist.in_neighbors(node)
        )

    # Core access interfaces
    @property
    def nodes(self) -> List[int]:
        return list(self.node_set)
    
    @property
    def update_nodes(self) -> Set[int]:
        return {self.u, self.v}

    def get_neighbors(self, node: int) -> List[int]:
        """Retrieve all neighbors (outgoing and incoming) of a node"""
        return list(set(
            self.di_adjlist.out_neighbors(node) + 
            self.di_adjlist.in_neighbors(node)
        ))

    def get_last_active(self, node: int) -> datetime:
        """Retrieve the last active time of a node"""
        return self.time_map.get(node, datetime.min)

    # Compatibility with the original Event interface
    @property
    def event(self) -> dict:
        return {
            "src_id": self.u,
            "dst_id": self.v,
            "event_type": self.event_type,
            "time_point": self.timestamp
        }

    # Dynamic update methods
    def update_timestamp(self, node: int, new_time: datetime):
        """Update the timestamp of a node (with thread lock)"""
        with self.di_adjlist.lock:
            if node in self.time_map:
                self.time_map[node] = max(self.time_map[node], new_time)

    # Statistical information
    @property
    def edge_count(self) -> int:
        """Calculate the number of edges in the subgraph"""
        count = 0
        for node in self.node_set:
            count += self.di_adjlist.out_degree(node)
        return count

    def density(self) -> float:
        """Calculate the density of the subgraph"""
        n = len(self.node_set)
        if n < 2:
            return 0.0
        return self.edge_count / (n * (n - 1))

if __name__ == '__main__':
    # Initialize the global adjacency list
    global_adj = DiAdjList()
    global_adj.add_vertexes(1000)  # Preallocate node space

    # Create a dynamic subgraph
    event_subg = DySubGraph(
        u=123,
        v=456,
        event_type=1,
        timestamp=datetime.now(),
        di_adjlist=global_adj,
        hop=2
    )

    # Query subgraph information
    print("Nodes included:", event_subg.nodes)
    print("Neighbors of node 123:", event_subg.get_neighbors(123))
    print("Last active time of node 456:", event_subg.get_last_active(456))

    # Update the timestamp of a node
    event_subg.update_timestamp(123, datetime.now())

    # Verify global impact
    print("Does the global adjacency list contain edge (123,456):", 
        global_adj.has_edge(123, 456))  # Should return True