import numpy as np  
import networkx as nx
import jittor as jt  
from jittor import nn  
from jittor_geometric.data import Data
from jittor_geometric.nn.models.graphormer import Graphormer
from typing import Tuple, Dict, List, Union
from jittor_geometric.utils.convert import to_networkx

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

class TransformerM(Graphormer):  
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
                 max_path_distance: int,  
                 # Transformer-M specific parameters  
                 add_3d: bool = False,  
                 num_3d_bias_kernel: int = 128,  
                 no_2d: bool = False,  
                 mode_prob: str = "0.2,0.2,0.6"):  
          
        # 初始化父类Graphormer  
        super().__init__(  
            num_layers=num_layers,  
            input_node_dim=input_node_dim,  
            node_dim=node_dim,  
            input_edge_dim=input_edge_dim,  
            edge_dim=edge_dim,  
            output_dim=output_dim,  
            n_heads=n_heads,  
            ff_dim=ff_dim,  
            max_in_degree=max_in_degree,  
            max_out_degree=max_out_degree,  
            max_path_distance=max_path_distance  
        )  
          
        # Transformer-M特有参数  
        self.add_3d = add_3d  
        self.no_2d = no_2d  
        self.num_3d_bias_kernel = num_3d_bias_kernel  
          
        # 解析多模态概率  
        try:  
            mode_prob_list = [float(x) for x in mode_prob.split(',')]  
            assert len(mode_prob_list) == 3  
            assert abs(sum(mode_prob_list) - 1.0) < 1e-6  
        except:  
            mode_prob_list = [0.2, 0.2, 0.6]  
        self.mode_prob = mode_prob_list  
          
        # 添加3D分子偏置组件  
        if self.add_3d:  
            self.molecule_3d_bias = Molecule3DBias(  
                num_heads=n_heads,  
                embed_dim=node_dim,  
                num_kernel=num_3d_bias_kernel  
            )
        else:  
            self.molecule_3d_bias = None  
      
    def _get_modal_masks(self, mask_choice: int):  
        """根据掩码选择返回2D和3D掩码"""  
        mask_dict = {0: [1, 1], 1: [1, 0], 2: [0, 1]}  # {2D+3D, 2D only, 3D only}  
        mask = mask_dict[mask_choice]  
        return mask[0], mask[1]  
      
    def execute(self, data) -> jt.Var:  
        """  
        Transformer-M的前向传播，支持多模态训练  
        """  
        x = data.x.float()  
        edge_index = data.edge_index.int()  
        edge_attr = data.edge_attr.float()  
          
        # 多模态掩码逻辑  
        mask_2d = mask_3d = None  
        if self.training:  
            n_mol = x.shape[0] if len(x.shape) == 2 else 1  
            mask_choice = np.random.choice(3, p=self.mode_prob)  
            mask_2d, mask_3d = self._get_modal_masks(mask_choice)  
              
            # 转换为张量用于后续计算  
            mask_2d = jt.array([mask_2d])  
            mask_3d = jt.array([mask_3d])  
          
        # 使用父类的基础处理  
        if type(data) == Data:  
            ptr = None  
            node_paths, edge_paths = shortest_path_distance(data)  
        else:  
            ptr = None  
            node_paths, edge_paths = shortest_path_distance(data)  
          
        # 节点和边特征线性变换  
        x = self.node_in_lin(x)  
        edge_attr = self.edge_in_lin(edge_attr)  
          
        # 中心性编码（继承自Graphormer）  
        x = self.centrality_encoding(x, edge_index)  
          
        # 空间编码（2D图结构）  
        b = self.spatial_encoding(x, node_paths)  
          
        # 应用2D掩码  
        if mask_2d is not None and not self.no_2d:  
            b = b * mask_2d  
          
        # 3D分子偏置处理  
        attn_bias_3d = None  
        if self.molecule_3d_bias is not None and hasattr(data, 'pos'):  
            attn_bias_3d = self.molecule_3d_bias(data.pos)  
              
            # 应用3D掩码  
            if mask_3d is not None:  
                attn_bias_3d = attn_bias_3d * mask_3d  
              
            # 将3D偏置与2D空间编码结合  
            b = b + attn_bias_3d  
          
        # 通过Transformer层  
        for layer in self.layers:  
            x = layer(x, edge_attr, b, edge_paths, ptr)  
            jt.gc()  
          
        # 输出投影  
        x = self.node_out_lin(x)  
          
        return x  
  
# 需要实现的3D偏置组件（简化版本）  
class Molecule3DBias(nn.Module):  
    def __init__(self, num_heads: int, embed_dim: int, num_kernel: int = 128):  
        super().__init__()  
        self.num_heads = num_heads  
        self.embed_dim = embed_dim  
        self.num_kernel = num_kernel  
          
        # 高斯基函数参数  
        self.gaussian_weights = nn.Parameter(jt.randn(num_kernel, num_heads))  
        self.gaussian_centers = nn.Parameter(jt.randn(num_kernel))  
        self.gaussian_widths = nn.Parameter(jt.ones(num_kernel))  
      
    def execute(self, pos: jt.Var) -> jt.Var:  
        """  
        计算基于3D坐标的注意力偏置  
        :param pos: 3D坐标 [n_nodes, 3]  
        :return: 3D注意力偏置 [n_nodes, n_nodes]  
        """  
        n_nodes = pos.shape[0]  
          
        # 计算原子间距离  
        pos_i = pos.unsqueeze(1)  # [n_nodes, 1, 3]  
        pos_j = pos.unsqueeze(0)  # [1, n_nodes, 3]  
        dist = jt.norm(pos_i - pos_j, dim=-1)  # [n_nodes, n_nodes]  
          
        # 使用高斯基函数编码距离  
        dist_expanded = dist.unsqueeze(-1)  # [n_nodes, n_nodes, 1]  
        centers_expanded = self.gaussian_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_kernel]  
        widths_expanded = self.gaussian_widths.unsqueeze(0).unsqueeze(0)  # [1, 1, num_kernel]  
          
        # 高斯基函数  
        gaussian_features = jt.exp(-0.5 * ((dist_expanded - centers_expanded) / widths_expanded) ** 2)  
          
        # 投影到注意力偏置  
        attn_bias = jt.matmul(gaussian_features, self.gaussian_weights)  # [n_nodes, n_nodes, num_heads]  
          
        return attn_bias.mean(dim=-1)  # 平均所有头的偏置