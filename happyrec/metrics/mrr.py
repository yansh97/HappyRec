import torch as th

from .base import RetrievalMatrixMetric


class RetrievalMatrixMRR(RetrievalMatrixMetric):
    """Computes mrr for binary target data."""

    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        sorted_target = target.take_along_dim(
            preds.argsort(dim=1, descending=True), dim=1
        )
        return (1 / (sorted_target.argmax(dim=1) + 1)).sum()
