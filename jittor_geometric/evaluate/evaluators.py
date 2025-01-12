import numpy as np
import jittor as jt

class MRR_Evaluator():
  def __init__(self) -> None:
    pass

  def eval(self, y_pred_pos, y_pred_neg):
    if jt is not None and isinstance(y_pred_pos, jt.Var):
        y_pred_pos = y_pred_pos.detach().cpu().numpy()
    if jt is not None and isinstance(y_pred_neg, jt.Var):
        y_pred_neg = y_pred_neg.detach().cpu().numpy()
    if not isinstance(y_pred_pos, np.ndarray) or not isinstance(y_pred_neg, np.ndarray):
        raise RuntimeError(
            "Arguments to Evaluator need to be either numpy ndarray or jittor Var!"
        )
    batch_size = y_pred_pos.shape[0]
    y_pred_pos = y_pred_pos.reshape(-1, 1)
    y_pred_neg = y_pred_neg.reshape(batch_size,-1)
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1./ranking_list.astype(np.float32)
    return mrr_list
  