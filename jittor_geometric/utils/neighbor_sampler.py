import numpy as np
from typing import Optional, Tuple
import jittor as jt
import copy

def neighbor_sampler(
    neighbor_list: jt.Var,
    neighbor_offs: jt.Var,
    neighbor_nums: jt.Var,
    source_node: jt.Var,
    max_nodes : Optional[int] = None,
    max_edges : Optional[int] = None
) -> jt.Var:
    r"""Samples the neighbor of all the source nodes, in the given graph 
    described by neighbor_list, neighbor_hook, and neighbor_offs.
    
    Args:
        neighbor_list (jittor.Var): An ordered list of neighbors.
        neighbor_offs (jittor.Var): For each node i, the neighbors of i
            are neighbor_list[neighbor_offs[i], neighbor_offs[i+1]]. There is
            neighbor_offs[max_nodes], which should equals to negihbor_list.size(0).
        neighbor_nums (jittor.Var): For each node i, the number of its 
            neighbors is neighbor_nums[i] == neighbor_offs[i+1] - 
            neighbor_offs[i].For neighbor_nums[i] == 0, should be preprocessed 
            (usually assigned max_edges).
        source_node (jittor,Var): The source node of sampling.
        max_nodes (int, optional): The total number of nodes, assert to be (the 
            length of neighbor_offs) - 1 and (the length of neighbor_nums) .
        max_edges (int, optional): The total number of edges, assert to be the 
            length of neighbor_list.
    """
    
    if max_nodes is None:
        max_nodes = neighbor_nums.size(0)
    if max_edges is None:
        max_edges = neighbor_list.size(0)
    idx = jt.randint_like(source_node, 0, max_edges)
    idx = (idx % neighbor_nums[source_node] + neighbor_offs[source_node]) % max_edges
    dst = neighbor_list[1,idx]
    
    return dst
    
    
def randomwalk_sampler(
    neighbor_list: jt.Var,
    neighbor_offs: jt.Var,
    neighbor_nums: jt.Var,
    source_node: jt.Var,
    walk_length: int,
    max_nodes : Optional[int] = None,
    max_edges : Optional[int] = None
) -> jt.Var:
    r"""Samples the random_walk of all the source nodes with length 'walk_length', 
    in the given graph described by neighbor_list, neighbor_hook, and neighbor_offs.
    
    Args:
        neighbor_list (jittor.Var): An ordered list of neighbors.
        neighbor_offs (jittor.Var): For each node i, the neighbors of i
            are neighbor_list[neighbor_offs[i], neighbor_offs[i+1]]. There is
            neighbor_offs[max_nodes], which should equals to negihbor_list.size(0).
        neighbor_nums (jittor.Var): For each node i, the number of its 
            neighbors is neighbor_nums[i] == neighbor_offs[i+1] - 
            neighbor_offs[i].For neighbor_nums[i] == 0, should be preprocessed 
            (usually assigned max_edges).
        source_node (jittor,Var): The source node of sampling.
        walk_length (int): The length of random walk.
        max_nodes (int, optional): The total number of nodes, assert to be (the 
            length of neighbor_offs) - 1 and (the length of neighbor_nums) .
        max_edges (int, optional): The total number of edges, assert to be the 
            length of neighbor_list.
    """
    if max_nodes is None:
        max_nodes = neighbor_nums.size(0)
    if max_edges is None:
        max_edges = neighbor_list.size(0)
    dst = jt.zeros([source_node.size(0), walk_length + 1], dtype = 'int')
    dst[:, 0] = source_node
    source = copy.copy(source_node)
    for i in range(walk_length):
        target = neighbor_sampler(neighbor_list, neighbor_offs, neighbor_nums, source, max_nodes, max_edges)
        dst[:, i+1] = target
        source = copy.copy(target)
    return dst
    