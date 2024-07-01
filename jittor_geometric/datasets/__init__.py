
from .planetoid import Planetoid
from .amazon import Amazon
from .wikipedia_network import WikipediaNetwork
from .webkb import WebKB
from .ogb import OGBNodePropPredDataset

__all__ = [
    'Planetoid',
    'Amazon',
    'WikipediaNetwork',
    'WebKB',
    'OGBNodePropPredDataset'
]

classes = __all__
