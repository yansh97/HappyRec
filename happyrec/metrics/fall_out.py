import torch as th

from .base import RetrievalMatrixTopKMetric


class RetrievalMatrixFallOut(RetrievalMatrixTopKMetric):
    """Computes fall out@k for binary target data."""

    higher_is_better: bool = False

    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        k = self.k if self.k is not None else target.size(1)
        target = 1 - target
        topk_target = target.take_along_dim(preds.topk(k, dim=1)[1], dim=1)
        return (topk_target.sum(dim=1) / target.sum(dim=1)).sum()
