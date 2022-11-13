import torch as th

from .base import RetrievalMatrixMetric


class RetrievalMatrixRPrecision(RetrievalMatrixMetric):
    """Computes r-precision for binary target data."""

    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        sorted_target = target.take_along_dim(
            preds.argsort(dim=1, descending=True), dim=1
        )
        precision = sorted_target.cumsum(dim=1).float() / th.arange(
            1, target.size(1) + 1
        )
        return precision[th.arange(target.size(0)), target.sum(dim=1) - 1].sum()
