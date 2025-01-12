from typing import Optional

import jittor as jt
from jittor import Var
from jittor_geometric.utils.coalesce import coalesce


def to_undirected(edge_index,
                  edge_weight: Optional[Var] = None,
                  num_nodes: Optional[int] = None,
                  reduce: str = 'add'):

    row, col = edge_index[0], edge_index[1]
    row, col = jt.concat([row, col], dim=0), jt.concat([col, row], dim=0)
    edge_index = jt.stack([row, col], dim=0)
    edge_weight = jt.concat([edge_weight, edge_weight], dim=0) if edge_weight is not None else None

    return coalesce(edge_index, edge_weight, num_nodes, reduce)