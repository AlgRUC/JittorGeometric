<!--
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-22 20:01:47
-->
## JittorGeometric需要的算子，明确输入和输出

#### 1.to_undirected(edge_index: Var, edge_attr: Union[Var,None,List[Var]], num_nodes: Optional[int])→ Union[Var,Tuple[Var,List[Var]]]
Converts the graph given by edge_index (Var, edge indices) to an undirected graph such that for every edge.

**Parameters**:  
edge_index (Var) – The edge indices.  
edge_attr (Var or List[Var], optional) – Edge weights or multi-dimensional edge features. (default: None)  
num_nodes (int, optional) – The number of nodes, i.e. max(edge_index) + 1. (default: None)

**Return**:  
Var if edge_attr is not passed, else (Var, Optional[Val] or List[Var]])

<strong><span style="background-color: yellow;">Already completed, please refer to  jittor_geometric.ops.toundirected <span></strong>


#### 2. SparseTensor.random_walk() / random_walk(src: SparseTensor, start: torch.Tensor, walk_length: int) -> torch.Tensor
Generates the random walks on graph **src**, starting from **start**, yielding a len(start) * (walk_length + 1) tensor.
I explain the operator in torch version, which can be easily converted into jittor version.

**Parameters**:
src (torch_sparse.tensor.SparseTensor) - The graph.
start (torch.tensor) - the starting nodes of the random walks.
walk_length (int) - the length of random walk.

**Return**
a len(start) * (walk_length + 1) tensor, representing the generated random walks.

#### 3. torch.multinomial(input: Tensor, num_samples: _int, replacement: _bool=False, *, generator: Optional[Generator]=None, out: Optional[Tensor]=None) -> Tensor: ...
Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.

**Parameters**
input (Tensor) – the input tensor containing probabilities
num_samples (int) – number of samples to draw
replacement (bool, optional) – whether to draw with replacement or not
generator (torch.Generator, optional) – a pseudorandom number generator for sampling
out (Tensor, optional) – the output tensor.

**Return**
A tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
