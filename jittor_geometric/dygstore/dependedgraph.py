from collections import defaultdict, deque
from typing import List, Dict, Set, Optional, Deque
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
from .dysubgraph import DySubGraph

class DependedGraph:
    def __init__(self, total_events: int, start_id: int = 0):
        """
        Dependency graph analysis system
        
        Parameters:
        total_events: Total number of events (for memory pre-allocation)
        start_id: Starting event ID (typically 0)
        """
        # Graph structure storage
        self.graph: Dict[int, Dict[str, Set[int]]] = defaultdict(
            lambda: {'deps': set(), 'deped': set()}
        )
        self.start_id = start_id
        self.total_events = total_events
        
        # Execution layer management
        self.execution_layers: List[List[int]] = []
        self.current_layer = 0
        self.lock = threading.RLock()
        
        # Cache optimization
        self.dependency_cache: Dict[tuple, bool] = {}
        
        # Initialize super node
        self._init_super_node()

    def _init_super_node(self):
        """Initialize virtual super node"""
        super_node_id = -1
        self.graph[super_node_id] = {'deps': set(), 'deped': set()}
        for i in range(self.total_events):
            self.add_dependency(super_node_id, i)

    def add_dependency(self, src: int, dst: int):
        """Add event dependency relationship (thread-safe)"""
        with self.lock:
            self.graph[src]['deped'].add(dst)
            self.graph[dst]['deps'].add(src)
            # Clear cache
            self.dependency_cache.clear()

    def analyze_batch(self, subgraphs: List[DySubGraph], max_workers: int = 10):
        """Batch analyze dependencies (parallel optimized version)"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx in range(0, len(subgraphs)):
                for idx1 in range(idx + 1, len(subgraphs)):
                    futures.append(
                        executor.submit(
                            self._analyze_single, 
                            subgraphs[idx], 
                            subgraphs[idx1], 
                            idx,
                            idx1
                        )
                    )
            for future in futures:
                future.result()
        self._build_execution_layers()

    def _analyze_single(self, prev: DySubGraph, current: DySubGraph, check_id: int, check_id1: int):
        """Single event dependency analysis (with cache optimization)"""
        cache_key = (id(prev), id(current))
        if cache_key in self.dependency_cache:
            if self.dependency_cache[cache_key]:
                self.add_dependency(check_id, check_id1)
            return

        # Temporal order check
        if prev.timestamp >= current.timestamp:
            return
        # Node conflict detection
        update_nodes = prev.update_nodes
        affected_nodes = current.nodes
        has_conflict = not update_nodes.isdisjoint(affected_nodes)

        with self.lock:
            self.dependency_cache[cache_key] = has_conflict
            if has_conflict:
                self.add_dependency(check_id, check_id1)

    def _build_execution_layers(self):
        """Build topological execution layers (based on Kahn's algorithm)"""
        in_degree = defaultdict(int)
        queue: Deque[int] = deque()
        layers = []

        # Initialize in-degree
        for node in self.graph:
            in_degree[node] = len(self.graph[node]['deps'])
            if in_degree[node] == 0:
                queue.append(node)

        # Layer processing
        while queue:
            layer_size = len(queue)
            current_layer = []
            for _ in range(layer_size):
                node = queue.popleft()
                current_layer.append(node)
                for neighbor in self.graph[node]['deped']:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            layers.append(current_layer)

        self.execution_layers = layers

    def get_parallel_batch(self) -> Optional[List[int]]:
        """Get next batch of parallel-executable event IDs (thread-safe)"""
        with self.lock:
            if self.current_layer >= len(self.execution_layers):
                return None
            layer = self.execution_layers[self.current_layer]
            self.current_layer += 1
            return [event_id for event_id in layer if event_id >= 0]

    def visualize(self):
        """Visualize dependency relationships (for debugging)"""
        print("===== Dependency Graph Structure =====")
        for node in sorted(self.graph.keys()):
            if node < 0:  # Filter system nodes
                continue
            deps = sorted([d for d in self.graph[node]['deps'] if d >= 0])
            deped = sorted([d for d in self.graph[node]['deped'] if d >= 0])
            print(f"Event {node}:")
            print(f"  Depends on: {deps}")
            print(f"  Required by: {deped}")
        print("===== Execution Layers =====")
        for i, layer in enumerate(self.execution_layers):
            print(f"Layer {i}: {sorted(layer)}")

if __name__ == '__main__':
    # To make methods called in _analyze_single work properly,
    # we add two helper methods to DySubGraph:
    # get_update_vertices: returns node set of current subgraph (for conflict detection)
    # get_all_nodes: returns all nodes of current subgraph
    if not hasattr(DySubGraph, "get_update_vertices"):
        DySubGraph.get_update_vertices = lambda self: set(self.nodes)
    if not hasattr(DySubGraph, "get_all_nodes"):
        DySubGraph.get_all_nodes = lambda self: set(self.nodes)

    # Create global adjacency list (all events share same graph structure)
    from adjlist import DiAdjList  # Ensure adjlist.py is in correct directory
    global_adj = DiAdjList()
    global_adj.add_vertexes(1000)

    # Create dynamic subgraphs to simulate different events
    base_time = datetime.now()
    subgraphs = []
    # Event 0: nodes 0 and 1
    subgraphs.append(DySubGraph(
        u=0, v=1, event_type=1,
        timestamp=base_time,
        di_adjlist=global_adj,
        hop=1
    ))
    # Event 1: nodes 1 and 2 (shares node 1 with Event 0 - should create dependency)
    subgraphs.append(DySubGraph(
        u=1, v=2, event_type=1,
        timestamp=base_time + timedelta(seconds=1),
        di_adjlist=global_adj,
        hop=1
    ))
    # Event 2: nodes 2 and 3 (shares node 2 with Event 1 - should create dependency)
    subgraphs.append(DySubGraph(
        u=2, v=3, event_type=1,
        timestamp=base_time + timedelta(seconds=2),
        di_adjlist=global_adj,
        hop=1
    ))
    # Event 3: nodes 4 and 5 (no overlap with previous events - no dependency)
    subgraphs.append(DySubGraph(
        u=4, v=5, event_type=1,
        timestamp=base_time + timedelta(seconds=3),
        di_adjlist=global_adj,
        hop=1
    ))
    # Event 4: nodes 3 and 1 (shares node 3 with Event 2 - should create dependency)
    subgraphs.append(DySubGraph(
        u=3, v=1, event_type=1,
        timestamp=base_time + timedelta(seconds=4),
        di_adjlist=global_adj,
        hop=1
    ))

    total_events = len(subgraphs)
    dep_graph = DependedGraph(total_events=total_events, start_id=0)
    # Batch dependency analysis: detect dependencies between consecutive event subgraphs and build execution layers
    dep_graph.analyze_batch(subgraphs, max_workers=4)

    # Output dependency graph visualization
    dep_graph.visualize()

    # Output parallel execution batches sequentially
    print("\n===== Parallel Execution Batches =====")
    batch = dep_graph.get_parallel_batch()
    batch_num = 0
    while batch is not None:
        print(f"Batch {batch_num}: {sorted(batch)}")
        batch = dep_graph.get_parallel_batch()
        batch_num += 1