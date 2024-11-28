<!--
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-22 19:37:27
-->
## Operator Files
- `cpp/xxxx_op.h`: Operator header file
- `cpp/xxxx_op.cc`: Specific implementation of the operator
- `xxxx.py`: Python program wrapping the operator

## Invocation Method
```python
from jittor_geometric.ops import xxxx
```

### Example
```python
# python api
from jittor_geometric.ops import cootocsr
data.csr = cootocsr(edge_index, edge_weight, v_num)
```
## Usage of Each Operator
### 1. `cootocsr`
Converts a graph from COO (Coordinate) format to CSR (Compressed Sparse Row) format.

#### Inputs
- **`edge_index` (Var)**: The indices of the edges in the COO format. It is expected to be a 2D Var where each column represents an edge, with the first row containing source nodes and the second row containing destination nodes.
- **`edge_weight` (Var)**: The weights of the edges in the COO format. It is a 1D Var where each element represents the weight of the corresponding  edge. If `edge_weight` is empty, weights do not need to be computed
- **`v_num` (int)**: The number of vertices in the graph.

#### Outputs
Returns a CSR representation of the graph, which includes column indices, row offsets, and edge weights.

### 2. `cootocsc`
Converts a graph from COO (Coordinate) format to CSC (Compressed Sparse Column) format.

#### Inputs
- **`edge_index` (Var)**: The indices of the edges in the COO format. It is expected to be a 2D Var where each column represents an edge, with the first row containing source nodes and the second row containing destination nodes.

- **`edge_weight` (Var)**: The weights of the edges in the COO format. It is a 1D Var where each element represents the weight of the corresponding edge.If `edge_weight` is empty, weights do not need to be computed
- **`v_num` (int)**: The number of vertices in the graph.

#### Outputs
Returns a CSC representation of the graph, which includes column indices, row offsets, and edge weights.

### 3. `aggregateWithWeight`
This function performs aggregation on the vertex embedding matrix using CSC (Compressed Sparse Column) and CSR (Compressed Sparse Row) representations of the graph.

#### Inputs
- **`x` (Var)**: The vertex embedding matrix of shape `(v_num, dim)`, where `v_num` is the number of vertices and `dim` is the dimension of the embeddings.
- **`csc` (CSC)**: The CSC representation of the graph, used for the forward pass.
  - `csc.edge_weight` (jt.Var): The edge weights in CSC format.
  - `csc.row_indices` (jt.Var): The row indices of non-zero entries in CSC format.
  - `csc.column_offset` (jt.Var): The column offsets in CSC format.
- **`csr` (CSR)**: The CSR representation of the graph, used for the backward pass.
  - `csr.edge_weight` (jt.Var): The edge weights in CSR format.
  - `csr.column_indices` (jt.Var): The column indices of non-zero entries in CSR format.
  - `csr.row_offset` (jt.Var): The row offsets in CSR format.

#### Outputs
Returns the aggregated vertex embeddings of the same shape as the input Var `x`.

### 4. `scatterToEdge`
Projects vertex features onto the edges of the graph, enabling operations that require edge-level computations. The function supports both "source-to-edge" (`src`) and "destination-to-edge" (`dst`) flows.

#### Inputs
- **`x` (Var)**: The vertex embedding matrix of shape `(v_num, dim)`, where `v_num` is the number of vertices, and `dim` is the dimension of the embeddings.
- **`csc` (CSC)**: The Compressed Sparse Column representation of the graph. Contains:
  - `csc.row_indices` (jt.Var): The row indices of non-zero entries in CSC format.
  - `csc.column_offset` (jt.Var): The column offsets in CSC format.
- **`flow` (str)**: Specifies the flow direction. Can be either:
  - `"src"`: Projects features from source vertices to edges.
  - `"dst"`: Projects features from destination vertices to edges.

#### Outputs
- **`output` (Var)**: The resulting edge-level features of shape `(e_num, dim)`, where `e_num` is the number of edges.

#### Gradients
During backpropagation, the gradient of the vertex embeddings is computed based on the edge-level gradients.

#### Example
```python
from jittor_geometric.ops import scatterToEdge

# Example inputs
x = jt.Var([[1.0, 2.0], [3.0, 4.0]])  # Vertex embeddings of shape (2, 2)
csc = CSC(
    row_indices=jt.Var([0, 1, 1]),    # Row indices of non-zero entries
    column_offset=jt.Var([0, 1, 3])   # Column offsets
)
flow = "src"  # Source-to-edge projection

# Scatter vertex features to edges
output = scatterToEdge(x, csc, flow)

# Outputs
# output: Edge-level features of shape (e_num, dim)
```
### 5. `scatterToVertex`
Projects vertex features onto the edges of the graph, enabling operations that require edge-level computations. The function supports both "source-to-edge" (`src`) and "destination-to-edge" (`dst`) flows.

#### Inputs
- **`x` (Var)**: The vertex embedding matrix of shape `(v_num, dim)`, where `v_num` is the number of vertices, and `dim` is the dimension of the embeddings.
- **`csc` (CSC)**: The Compressed Sparse Column representation of the graph. Contains:
  - `csc.row_indices` (jt.Var): The row indices of non-zero entries in CSC format.
  - `csc.column_offset` (jt.Var): The column offsets in CSC format.
- **`flow` (str)**: Specifies the flow direction. Can be either:
  - `"src"`: Projects features from source vertices to edges.
  - `"dst"`: Projects features from destination vertices to edges.

#### Outputs
- **`output` (Var)**: The resulting edge-level features of shape `(e_num, dim)`, where `e_num` is the number of edges.

#### Gradients
During backpropagation, the gradient of the vertex embeddings is computed based on the edge-level gradients.

#### Example
```python
from jittor_geometric.ops import scatterToEdge

# Example inputs
x = jt.Var([[1.0, 2.0], [3.0, 4.0]])  # Vertex embeddings of shape (2, 2)
csc = CSC(
    row_indices=jt.Var([0, 1, 1]),    # Row indices of non-zero entries
    column_offset=jt.Var([0, 1, 3])   # Column offsets
)
flow = "src"  # Source-to-edge projection

# Scatter vertex features to edges
output = scatterToEdge(x, csc, flow)

# Outputs
# output: Edge-level features of shape (e_num, dim)
```
### 6. `edgesoftmax`

Applies a softmax normalization to edge features based on the neighbors of each vertex, using the CSC (Compressed Sparse Column) format.

#### Inputs
- **`x` (Var)**: The input edge feature matrix of shape `(e_num,)`, where `e_num` is the number of edges.
- **`csc` (CSC)**: The graph represented in CSC format with the following attributes:
  - **`row_indices` (Var)**: Row indices of non-zero entries in CSC format, representing the source vertices of edges.
  - **`column_offset` (Var)**: Column offsets in CSC format, indicating the start of each vertex's edge list.

#### Outputs
- **`output` (Var)**: The normalized edge features of shape `(e_num,)`, where features connected to the same vertex are normalized using the softmax function.

#### Gradients
During backpropagation, the gradient of the input edge features (`x`) is computed based on the gradient of the output and the CSC structure.

#### Example
```python
from jittor_geometric.ops import EdgeSoftmax
from jittor_geometric.data import CSC

# Example inputs
x = jt.Var([0.5, 1.5, 2.0])  # Edge features of shape (3,)
csc = CSC(
    row_indices=jt.Var([0, 1, 1]),    # Row indices in CSC format
    column_offset=jt.Var([0, 1, 3])   # Column offsets
)

# Apply edge softmax
output = EdgeSoftmax(x, csc)

# Outputs
# output: Normalized edge features of shape (3,)
```

### 7. `spmmcoo`
Performs Sparse Matrix Multiplication (SpMM) using the COO (Coordinate) representation of the graph. This function supports both the forward and backward pass.

#### Inputs
- **`x` (Var)**: The vertex embedding matrix of shape `(v_num, dim)`, where `v_num` is the number of vertices, and `dim` is the dimension of the embeddings.
- **`edge_index` (Var)**: A 2D Var where each column represents an edge in COO format. The first row contains the source nodes, and the second row contains the destination nodes.
- **`edge_weight` (Var)**: The weights of the edges in COO format. It is a 1D Var where each element corresponds to the weight of the respective edge.

#### Outputs
- **`output` (Var)**: The result of the sparse matrix multiplication, with the same shape as the input Var `x` (`(v_num, dim)`).

#### Gradients
During backpropagation, the gradient of the input `x` is computed using the COO representation and the gradient of the output.

#### Example
```python
from jittor_geometric.ops import spmmcoo

# Example inputs
x = jt.Var([[1.0, 2.0], [3.0, 4.0]])  # Vertex embeddings of shape (2, 2)
edge_index = jt.Var([[0, 1], [1, 0]])  # Directed edges in COO format
edge_weight = jt.Var([0.5, 0.8])  # Edge weights

# Sparse matrix multiplication
output = spmmcoo(x, edge_index, edge_weight)

# Outputs
# output: Result of SpMM
```
### 8. `spmmcsr`
Performs Sparse Matrix Multiplication (SpMM) using the CSR (Compressed Sparse Row) representation of the graph. This function supports both the forward and backward pass.

#### Inputs
- **`x` (Var)**: The vertex embedding matrix of shape `(v_num, dim)`, where `v_num` is the number of vertices, and `dim` is the dimension of the embeddings.
- **`csr` (CSR)**: The CSR representation of the graph.
  - `csr.column_indices` (jt.Var): The column indices of non-zero entries in CSR format.
  - `csr.edge_weight` (jt.Var): The edge weights in CSR format.
  - `csr.row_offset` (jt.Var): The row offsets in CSR format.

#### Outputs
- **`output` (Var)**: The result of the sparse matrix multiplication, with the same shape as the input Var `x` (`(v_num, dim)`).

#### Gradients
During backpropagation, the gradient of the input `x` is computed using the CSR representation and the gradient of the output.

#### Example
```python
from jittor_geometric.ops import spmmcsr
x = jt.Var([[1.0, 2.0], [3.0, 4.0]])  # Vertex embeddings of shape (2, 2)
csr = CSR(
    column_indices=jt.Var([0, 1]),
    edge_weight=jt.Var([0.5, 0.8]),
    row_offset=jt.Var([0, 1, 2])
)
output = spmmcsr(x, csr)
```
### 9. `toundirected`
Converts a directed graph into an undirected graph by adding reverse edges.

#### Inputs
- **`edge_index` (Var)**: The indices of the edges in the directed graph. It is a 2D Var where each column represents an edge, with the first row containing source nodes and the second row containing destination nodes.
- **`edge_attr` (Var)**: The attributes (e.g., weights) of the edges in the directed graph. It is a 1D Var where each element corresponds to the attribute of the respective edge.
- **`num_nodes` (int, optional)**: The number of vertices in the graph. If not provided, it will be inferred from `edge_index`.

#### Outputs
Returns a tuple:
1. **`new_edge_index` (Var)**: The indices of the edges in the undirected graph. It includes all original edges and their reverse edges.
2. **`new_edge_attr` (Var)**: The attributes of the edges in the undirected graph. It includes the original attributes and their corresponding values for the reverse edges.

#### Example
```python
from jittor_geometric.ops import toUndirected

# Example inputs
edge_index = jt.Var([[0, 1], [1, 2]])  # Directed edges: 0->1, 1->2
edge_attr = jt.Var([0.5, 0.8])  # Attributes: [0.5, 0.8]
num_nodes = 3

# Convert to undirected
new_edge_index, new_edge_attr = toUndirected(edge_index, edge_attr, num_nodes)

# Outputs
# new_edge_index: [[0, 1, 1, 2], [1, 0, 2, 1]]
# new_edge_attr: [0.5, 0.5, 0.8, 0.8]