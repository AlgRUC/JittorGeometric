<!--
 * @Description: 
 * @Author: lusz
 * @Date: 2024-06-22 19:37:27
-->
## Operator Files
- `xxxx_op.h`: Operator header file
- `xxxx_op.cc`: Specific implementation of the operator
- `xxxx.py`: Python program wrapping the operator

## Invocation Method
```python
from jittor_geometric.ops import xxxx
```

### Example
```python
from jittor_geometric.ops import cootocsr

data.csr = cootocsr.cootocsr(edge_index, edge_weight, v_num)
```
## Usage of Each Operator

### 1. `cootocsr`
Converts a graph from COO (Coordinate) format to CSR (Compressed Sparse Row) format.

#### Inputs
- **`edge_index` (Var)**: The indices of the edges in the COO format. It is expected to be a 2D Var where each column represents an edge, with the first row containing source nodes and the second row containing destination nodes.
- **`edge_weight` (Var)**: The weights of the edges in the COO format. It is a 1D Var where each element represents the weight of the corresponding  edge. if `edge_weight` is empty, weights do not need to be computed
- **`v_num` (int)**: The number of vertices in the graph.

#### Outputs
Returns a CSR representation of the graph, which includes column indices, row offsets, and edge weights.

### 2. `cootocsc`
Converts a graph from COO (Coordinate) format to CSC (Compressed Sparse Column) format.

#### Inputs
- **`edge_index` (Var)**: The indices of the edges in the COO format. Each tuple represents an edge, with the first element being the source node and the second element being the destination node.
- **`edge_weight` (Var)**: The weights of the edges in the COO format. Each element represents the weight of the corresponding edge.if `edge_weight` is empty, weights do not need to be computed
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