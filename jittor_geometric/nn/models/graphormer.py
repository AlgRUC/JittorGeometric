from __future__ import annotations  
import sys
from typing import Tuple, Dict, List, Union  
# Remove torch.multiprocessing import as it's not needed for the core functionality  
  
import networkx as nx
import jittor as jt
from jittor import nn
from jittor_geometric.data import Data, Batch
from jittor_geometric.utils import degree
from jittor_geometric.utils.convert import to_networkx

def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x
  
def floyd_warshall_source_to_all(G, source, cutoff=None):  
    if source not in G:  
        raise nx.NodeNotFound("Source {} not in G".format(source))  
  
    edges = {edge: i for i, edge in enumerate(G.edges())}  
  
    level = 0  # the current level  
    nextlevel = {source: 1}  # list of nodes to check at next level  
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)  
    edge_paths = {source: []}  
  
    while nextlevel:  
        thislevel = nextlevel  
        nextlevel = {}  
        for v in thislevel:  
            for w in G[v]:  
                if w not in node_paths:  
                    node_paths[w] = node_paths[v] + [w]  
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]  
                    nextlevel[w] = 1  
  
        level = level + 1  
  
        if (cutoff is not None and cutoff <= level):  
            break  
  
    return node_paths, edge_paths  

def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:  
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}  
    node_paths = {n: paths[n][0] for n in paths}  
    edge_paths = {n: paths[n][1] for n in paths}  
    return node_paths, edge_paths  

def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:  
    G = to_networkx(data)  # Use custom conversion function  
    node_paths, edge_paths = all_pairs_shortest_path(G)  
    return node_paths, edge_paths  
  
  
def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:  
    # Note: You'll need to implement to_data_list() for JittorGeometric Batch  
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]  
    relabeled_graphs = []  
    shift = 0  
    for i in range(len(graphs)):  
        num_nodes = graphs[i].number_of_nodes()  
        relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))  
        shift += num_nodes  
  
    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]  
    node_paths = {}  
    edge_paths = {}  
  
    for path in paths:  
        for k, v in path[0].items():  
            node_paths[k] = v  
        for k, v in path[1].items():  
            edge_paths[k] = v  
  
    return node_paths, edge_paths

class CentralityEncoding(nn.Module):  
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):  
        """  
        :param max_in_degree: max in degree of nodes  
        :param max_out_degree: max in degree of nodes  
        :param node_dim: hidden dimensions of node features  
        """  
        super().__init__()  
        self.max_in_degree = max_in_degree  
        self.max_out_degree = max_out_degree  
        self.node_dim = node_dim  
        self.z_in = nn.Parameter(jt.randn((max_in_degree, node_dim)))  
        self.z_out = nn.Parameter(jt.randn((max_out_degree, node_dim)))  
  
    def execute(self, x: jt.Var, edge_index: jt.Var) -> jt.Var:  
        """  
        :param x: node feature matrix  
        :param edge_index: edge_index of graph (adjacency list)  
        :return: jt.Var, node embeddings after Centrality encoding  
        """  
        num_nodes = x.shape[0]  
  
        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes, dtype=jt.int32).int32(),  
                                          self.max_in_degree - 1)  
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes, dtype=jt.int32).int32(),  
                                           self.max_out_degree - 1)  
  
        x += self.z_in[in_degree] + self.z_out[out_degree]  
  
        return x  
  
  
class SpatialEncoding(nn.Module):  
    def __init__(self, max_path_distance: int):  
        """  
        :param max_path_distance: max pairwise distance between nodes  
        """  
        super().__init__()  
        self.max_path_distance = max_path_distance  
  
        self.b = nn.Parameter(jt.randn(self.max_path_distance))  
  
    def execute(self, x: jt.Var, paths) -> jt.Var:  
        """  
        :param x: node feature matrix  
        :param paths: pairwise node paths  
        :return: jt.Var, spatial Encoding matrix  
        """  
        spatial_matrix = jt.zeros((x.shape[0], x.shape[0]))  
        for src in paths:  
            for dst in paths[src]:  
                spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1]  
  
        return spatial_matrix  
  
  
def dot_product(x1, x2) -> jt.Var:  
    return (x1 * x2).sum(dim=1)
  
  
class EdgeEncoding(nn.Module):  
    def __init__(self, edge_dim: int, max_path_distance: int):  
        """  
        :param edge_dim: edge feature matrix number of dimension  
        """  
        super().__init__()  
        self.edge_dim = edge_dim  
        self.max_path_distance = max_path_distance  
        self.edge_vector = nn.Parameter(jt.randn(self.max_path_distance, self.edge_dim))  

    def execute(self, x: jt.Var, edge_attr: jt.Var, edge_paths) -> jt.Var:  
        cij = jt.zeros((x.shape[0], x.shape[0]))  
  
        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()
        cij = jt.where(jt.isnan(cij), jt.array(0.0), cij)

        return cij
    
class GraphormerAttentionHead(nn.Module):  
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):  
        """  
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension  
        """  
        super().__init__()  
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)  
  
        self.q = nn.Linear(dim_in, dim_q)  
        self.k = nn.Linear(dim_in, dim_k)  
        self.v = nn.Linear(dim_in, dim_k)  
  
    def execute(self,  
                x: jt.Var,  
                edge_attr: jt.Var,  
                b: jt.Var,  
                edge_paths,  
                ptr=None) -> jt.Var:  
        """  
        :param x: node feature matrix  
        :param edge_attr: edge feature matrix  
        :param b: spatial Encoding matrix  
        :param edge_paths: pairwise node paths in edge indexes  
        :param ptr: batch pointer that shows graph indexes in batch of graphs  
        :return: jt.Var, node embeddings after attention operation  
        """  
        batch_mask_neg_inf = jt.full((x.shape[0], x.shape[0]), -1e6)  
        batch_mask_zeros = jt.zeros((x.shape[0], x.shape[0]))  
  
        if ptr is None:  
            batch_mask_neg_inf = jt.ones((x.shape[0], x.shape[0]))  
            batch_mask_zeros += 1  
        else:  
            for i in range(len(ptr) - 1):  
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1  
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1  
  
        query = self.q(x)  
        key = self.k(x)  
        value = self.v(x)  
  
        c = self.edge_encoding(x, edge_attr, edge_paths)  
        a = self.compute_a(key, query, ptr)  
        a = (a + b + c) * batch_mask_neg_inf  
        softmax = jt.nn.softmax(a, dim=-1) * batch_mask_zeros  
        x = jt.matmul(softmax, value)  
        return x  
  
    def compute_a(self, key, query, ptr=None):  
        if ptr is None:  
            a = jt.matmul(query, key.transpose(0, 1)) / (query.size(-1) ** 0.5)  
        else:  
            a = jt.zeros((query.shape[0], query.shape[0]))  
            for i in range(len(ptr) - 1):  
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = jt.matmul(  
                    query[ptr[i]:ptr[i + 1]],   
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)  
                ) / (query.size(-1) ** 0.5)  
  
        return a  
  
  
class GraphormerMultiHeadAttention(nn.Module):  
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):  
        """  
        :param num_heads: number of attention heads  
        :param dim_in: node feature matrix input number of dimension  
        :param dim_q: query node feature matrix input number dimension  
        :param dim_k: key node feature matrix input number of dimension  
        :param edge_dim: edge feature matrix number of dimension  
        """  
        super().__init__()  
        self.heads = nn.ModuleList(  
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]  
        )  
        self.linear = nn.Linear(num_heads * dim_k, dim_in)  
  
    def execute(self,  
                x: jt.Var,  
                edge_attr: jt.Var,  
                b: jt.Var,  
                edge_paths,  
                ptr) -> jt.Var:  
        """  
        :param x: node feature matrix  
        :param edge_attr: edge feature matrix  
        :param b: spatial Encoding matrix  
        :param edge_paths: pairwise node paths in edge indexes  
        :param ptr: batch pointer that shows graph indexes in batch of graphs  
        :return: jt.Var, node embeddings after all attention heads  
        """  
        return self.linear(  
            jt.cat([  
                attention_head(x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads  
            ], dim=-1)  
        )  
  
  
class GraphormerEncoderLayer(nn.Module):  
    def __init__(self, node_dim, edge_dim, n_heads, ff_dim, max_path_distance):  
        """  
        :param node_dim: node feature matrix input number of dimension  
        :param edge_dim: edge feature matrix input number of dimension  
        :param n_heads: number of attention heads  
        """  
        super().__init__()  
  
        self.node_dim = node_dim  
        self.edge_dim = edge_dim  
        self.n_heads = n_heads  
        self.ff_dim = ff_dim  
  
        self.attention = GraphormerMultiHeadAttention(  
            dim_in=node_dim,  
            dim_k=node_dim,  
            dim_q=node_dim,  
            num_heads=n_heads,  
            edge_dim=edge_dim,  
            max_path_distance=max_path_distance,  
        )  
        self.ln_1 = nn.LayerNorm(self.node_dim)  
        self.ln_2 = nn.LayerNorm(self.node_dim)  
        self.ff = nn.Sequential(  
                    nn.Linear(self.node_dim, self.ff_dim),  
                    nn.GELU(),  
                    nn.Linear(self.ff_dim, self.node_dim)  
        )  
  
    def execute(self,  
                x: jt.Var,  
                edge_attr: jt.Var,  
                b: jt.Var,  
                edge_paths,  
                ptr) -> jt.Var:  
        """  
        h'(l) = MHA(LN(h(l-1))) + h(l-1)  
        h(l) = FFN(LN(h'(l))) + h'(l)  
  
        :param x: node feature matrix  
        :param edge_attr: edge feature matrix  
        :param b: spatial Encoding matrix  
        :param edge_paths: pairwise node paths in edge indexes  
        :param ptr: batch pointer that shows graph indexes in batch of graphs  
        :return: jt.Var, node embeddings after Graphormer layer operations  
        """  
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x  
        x_new = self.ff(self.ln_2(x_prime)) + x_prime  
  
        return x_new



class Graphormer(nn.Module):  
    def __init__(self,  
                 num_layers: int,  
                 input_node_dim: int,  
                 node_dim: int,  
                 input_edge_dim: int,  
                 edge_dim: int,  
                 output_dim: int,  
                 n_heads: int,  
                 ff_dim: int,  
                 max_in_degree: int,  
                 max_out_degree: int,  
                 max_path_distance: int):  
        """  
        :param num_layers: number of Graphormer layers  
        :param input_node_dim: input dimension of node features  
        :param node_dim: hidden dimensions of node features  
        :param input_edge_dim: input dimension of edge features  
        :param edge_dim: hidden dimensions of edge features  
        :param output_dim: number of output node features  
        :param n_heads: number of attention heads  
        :param max_in_degree: max in degree of nodes  
        :param max_out_degree: max in degree of nodes  
        :param max_path_distance: max pairwise distance between two nodes  
        """  
        super().__init__()  
  
        self.num_layers = num_layers  
        self.input_node_dim = input_node_dim  
        self.node_dim = node_dim  
        self.input_edge_dim = input_edge_dim  
        self.edge_dim = edge_dim  
        self.output_dim = output_dim  
        self.n_heads = n_heads  
        self.ff_dim = ff_dim  
        self.max_in_degree = max_in_degree  
        self.max_out_degree = max_out_degree  
        self.max_path_distance = max_path_distance  
  
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)  
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)  

        self.centrality_encoding = CentralityEncoding(  
            max_in_degree=self.max_in_degree,  
            max_out_degree=self.max_out_degree,  
            node_dim=self.node_dim  
        )  
  
        self.spatial_encoding = SpatialEncoding(  
            max_path_distance=max_path_distance,  
        )  
  
        self.layers = nn.ModuleList([  
            GraphormerEncoderLayer(  
                node_dim=self.node_dim,  
                edge_dim=self.edge_dim,  
                n_heads=self.n_heads,  
                ff_dim=self.ff_dim,  
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)  
        ])  
  
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)  
  
    def execute(self, data: Union[Data]) -> jt.Var:  
        """  
        :param data: input graph of batch of graphs  
        :return: jt.Var, output node embeddings  
        """  
        x = data.x.float()  
        edge_index = data.edge_index.int32()  # JittorGeometric uses int32 for edge indices  
        edge_attr = data.edge_attr.float() 
  
        if type(data) == Data:  
            ptr = None  
            node_paths, edge_paths = shortest_path_distance(data)  
        else:
            ptr = None  # Set to None since JittorGeometric doesn't use ptr  
            # node_paths, edge_paths = batched_shortest_path_distance(data)
            node_paths, edge_paths = shortest_path_distance(data)
  
        x = self.node_in_lin(x)  
        edge_attr = self.edge_in_lin(edge_attr)  
  
        x = self.centrality_encoding(x, edge_index)  
        b = self.spatial_encoding(x, node_paths)  
  
        for layer in self.layers:  
            x = layer(x, edge_attr, b, edge_paths, ptr)  
  
        x = self.node_out_lin(x)  
  
        return x