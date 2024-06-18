## JittorGeometric需要的算子，明确输入和输出

#### 1.to_undirected(edge_index: Var, edge_attr: Union[Var,None,List[Var]], num_nodes: Optional[int])→ Union[Var,Tuple[Var,List[Var]]]
Converts the graph given by edge_index (Var, edge indices) to an undirected graph such that for every edge.

**PARAMETERS**: 
edge_index (Var) – The edge indices.
edge_attr (Var or List[Var], optional) – Edge weights or multi-dimensional edge features.(default: None)
num_nodes (int, optional) – The number of nodes, i.e. max(edge_index) + 1. (default: None)

**RETURN TYPE**:
Var if edge_attr is not passed, else (Var, Optional[Val] or List[Var]])

