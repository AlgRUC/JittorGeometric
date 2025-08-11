import jittor as jt
import jittor.nn as nn


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def execute(self, pos_score, neg_score, mask=None):
        loss = -jt.log(self.gamma + jt.sigmoid(pos_score - neg_score))
        if mask is not None:
            loss = loss[mask]
        return loss.mean()