import torch as th

from .base import RetrievalMatrixTopKMetric


class RetrievalMatrixNormalizedDCG(RetrievalMatrixTopKMetric):
    """Computes ndcg@k for binary target data."""

    @staticmethod
    def _dcg(target: th.Tensor) -> th.Tensor:
        discount = th.log2(th.arange(target.size(1)).to(target) + 2.0).unsqueeze(0)
        return (target / discount).sum(dim=-1)

    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        k = self.k if self.k is not None else target.size(1)
        ideal_topk_target = target.topk(k, dim=1)[0]
        topk_target = target.take_along_dim(preds.topk(k, dim=1)[1], dim=1)
        return (self._dcg(topk_target) / self._dcg(ideal_topk_target)).sum()
