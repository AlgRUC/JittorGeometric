from .planetoid import Planetoid
from .amazon import Amazon
from .wikipedia_network import WikipediaNetwork
from .ogb import OGBNodePropPredDataset
from .jodie import JODIEDataset
from .reddit import Reddit
__all__ = [
    'Planetoid',
    'Amazon',
    'WikipediaNetwork',
    'WebKB',
    'OGBNodePropPredDataset',
    'JODIEDataset'
]

classes = __all__