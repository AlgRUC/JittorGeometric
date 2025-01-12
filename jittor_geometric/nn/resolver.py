from typing import Any, Optional, Union
from jittor_geometric.resolver import resolver

def aggregation_resolver(query: Union[Any, str], *args, **kwargs):
    import jittor_geometric.nn.aggr as aggr
    if isinstance(query, (list, tuple)):
        return aggr.MultiAggregation(query, *args, **kwargs)

    base_cls = aggr.Aggregation
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    aggr_dict = {
        'add': aggr.SumAggregation,
    }
    return resolver(aggrs, aggr_dict, query, base_cls, None, *args, **kwargs)