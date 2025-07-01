'''
Description: 
Author: lusz
Date: 2025-01-11 13:40:29
'''
import jittor as jt
from jittor_geometric.data import CSC, CSR

def from_nodes(csc, nodes):
    """
    Given a CSC structure and a set of input nodes, find all the neighbor nodes.

    Parameters:
        csc (CSC): Compressed Sparse Column structure.
        nodes (Var): Input node IDs (Var type).

    Returns:
        Var: A Var containing all neighbor nodes corresponding to the input nodes.
    """
    if csc is not None:
        # Use CSC (column-based) to find neighbors
        column_offset = csc.column_offset
        row_indices = csc.row_indices

        # Extract neighbors for each input node
        neighbors = []
        for i in range(nodes.shape[0]):
            node = nodes[i]
            start = column_offset[node]
            end = column_offset[node + 1]
            neighbors.append(row_indices[start:end])

    else:
        raise ValueError("CSC structure must be provided.")

    # Flatten the list of neighbors and remove duplicates
    neighbors = jt.Var(jt.contrib.concat([n for n in neighbors]))
    # unique_neighbors = jt.unique(neighbors)

    return neighbors


def to_nodes(csr, nodes):
    """
    Given a CSR structure and a set of input nodes, find all the neighbor nodes.

    Parameters:
        csr (CSR): Compressed Sparse Row structure.
        nodes (Var): Input node IDs (Var type).

    Returns:
        Var: A Var containing all neighbor nodes corresponding to the input nodes.
    """
    if csr is not None:
        # Use CSR to find neighbors
        row_offset = csr.row_offset
        column_indices = csr.column_indices
        neighbors = []
        for i in range(nodes.shape[0]):
            node = nodes[i]
            start = row_offset[node]
            end = row_offset[node + 1]
            neighbors.append(column_indices[start:end])

    else:
        raise ValueError("CSR structure must be provided.")

    # Flatten the list of neighbors and remove duplicates
    neighbors = jt.Var(jt.contrib.concat([n for n in neighbors]))
    # unique_neighbors = jt.unique(neighbors)

    return neighbors

