from typing import Tuple, Optional, Union

from jittor import Var

Adj = Optional[Var]
OptVar = Optional[Var]
PairVar = Tuple[Var, Var]
OptPairVar = Tuple[Var, Optional[Var]]
PairOptVar = Tuple[Optional[Var], Optional[Var]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Var]

EdgeType = Tuple[str, str, str]
NodeType = str
SparseVar = Optional[Var]
jt_lib = object
jt_scatter = object