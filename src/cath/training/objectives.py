"""Next-POI log-probability utility."""
import torch.nn.functional as F


def utility(scores, positive_index):
    return F.log_softmax(scores, dim=0)[positive_index]
