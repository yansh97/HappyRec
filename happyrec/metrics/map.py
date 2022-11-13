import torch as th

from .base import RetrievalMatrixMetric


class RetrievalMatrixMAP(RetrievalMatrixMetric):
    """Computes map for binary target data."""

    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        sorted_target = target.take_along_dim(
            preds.argsort(dim=1, descending=True), dim=1
        )
        sum_precision = (
            sorted_target.cumsum(dim=1)
            * (sorted_target > 0).float()
            / th.arange(1, target.size(1) + 1)
        ).sum(dim=1)
        return (sum_precision / target.sum(dim=1)).sum()
