
from .planetoid import Planetoid
from .amazon import Amazon
from .wikipedia_network import WikipediaNetwork
from .webkb import WebKB
from .ogb import OGBNodePropPredDataset
from .jodie import JODIEDataset

__all__ = [
    'Planetoid',
    'Amazon',
    'WikipediaNetwork',
    'WebKB',
    'OGBNodePropPredDataset'
    'JODIEDataset'
]

classes = __all__
