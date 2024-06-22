from typing import Tuple, Optional, Union

from jittor import Var

Adj = Optional[Var]
OptVar = Optional[Var]
PairVar = Tuple[Var, Var]
OptPairVar = Tuple[Var, Optional[Var]]
PairOptVar = Tuple[Optional[Var], Optional[Var]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Var]
