import torch as th

from .base import RetrievalMatrixTopKMetric


class RetrievalMatrixPrecision(RetrievalMatrixTopKMetric):
    """Computes precision@k for binary target data."""

    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        k = self.k if self.k is not None else target.size(1)
        topk_target = target.take_along_dim(preds.topk(k, dim=1)[1], dim=1)
        return topk_target.sum() / k
