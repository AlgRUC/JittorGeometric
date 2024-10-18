
from .planetoid import Planetoid
from .amazon import Amazon
from .wikipedia_network import WikipediaNetwork
from .geomgcn import GeomGCN
from .ogb import OGBNodePropPredDataset
from .jodie import JODIEDataset
from .linkx import LINKXDataset

__all__ = [
    'Planetoid',
    'Amazon',
    'WikipediaNetwork',
    'GeomGCN',
    'LINKXDataset',
    'OGBNodePropPredDataset',
    'JODIEDataset'
]

classes = __all__
