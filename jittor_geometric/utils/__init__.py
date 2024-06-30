from .coalesce import coalesce
from .degree import degree
from .loop import (contains_self_loops, remove_self_loops,
                   segregate_self_loops, add_self_loops,
                   add_remaining_self_loops)
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from .get_laplacian import get_laplacian
from .undirected import to_undirected

__all__ = [
    'coalesce',
    'degree',
    'contains_self_loops',
    'remove_self_loops',
    'segregate_self_loops',
    'add_self_loops',
    'add_remaining_self_loops',
    'contains_isolated_nodes',
    'remove_isolated_nodes',
    'get_laplacian',
    'undirected'
]

classes = __all__
