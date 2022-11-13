from abc import ABC, abstractmethod
from typing import Literal

import torch as th
from torchmetrics import Metric


class RetrievalMatrixMetric(Metric, ABC):
    """Retrieval metric with binary target matrix and float prediction matrix."""

    is_differentiable: bool = False
    """Whether the metric is differentiable or not."""

    higher_is_better: bool = True
    """Whether a higher metric value is better or not."""

    full_state_update: bool = False
    """Whether the metric value of one update step is related to the existing states."""

    def __init__(
        self,
        empty_target_action: Literal["neg", "pos", "skip", "error"] = "neg",
        **kwargs,
    ) -> None:
        """Initialize the metric.

        :param empty_target_action: Specify what to do with queries that do not have
          at least a positive or negative (depend on metric) target. Defaults:
          ``"neg"``.
        :raises ValueError: If ``empty_target_action`` is not one of ``neg``, ``pos``,
          ``skip`` or ``error``.
        """
        super().__init__(**kwargs)

        empty_target_action_options = ("neg", "pos", "skip", "error")
        if empty_target_action not in empty_target_action_options:
            raise ValueError(
                "Argument empty_target_action received a wrong value"
                f" {empty_target_action}."
            )

        self.empty_target_action: Literal[
            "neg", "pos", "skip", "error"
        ] = empty_target_action

        self.value_name: str = f"{self.__class__.__name__}_value"
        self.query_num_name: str = f"{self.__class__.__name__}_query_num"
        # Different child classes have different state names, since the same state
        # name will be reused for all child classes in the MetricCollection.

        self.add_state(self.value_name, default=th.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(self.query_num_name, default=th.tensor(0), dist_reduce_fx="sum")

    def _check_update_input(
        self, preds: th.Tensor, target: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        if preds.shape != target.shape:
            raise ValueError("preds and target must be of the same shape")

        if not preds.numel() or preds.ndim != 2:
            raise ValueError(
                "preds and target must be non-empty and two-dimensional tensors",
            )

        if not preds.is_floating_point():
            raise ValueError("preds must be a tensor of floats")

        if target.dtype not in (th.bool, th.long, th.int):
            raise ValueError("target must be a tensor of booleans, integers.")

        if target.max() > 1 or target.min() < 0:
            raise ValueError("target must contain `binary` values")

        return preds.float(), target.long()

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        """Check shape, check and convert dtypes and update states."""
        preds, target = self._check_update_input(preds, target)

        value: th.Tensor = getattr(self, self.value_name)
        query_num: th.Tensor = getattr(self, self.query_num_name)

        if self.higher_is_better:
            non_empty_queries = target.sum(dim=1) > 0
        else:
            non_empty_queries = (1 - target).sum(dim=1) > 0
        non_empty_query_num = non_empty_queries.long().sum()

        if non_empty_query_num == 0:
            value += 0.0
        else:
            non_empty_preds = preds[non_empty_queries]
            non_empty_target = target[non_empty_queries]
            value += self._metric(non_empty_preds, non_empty_target).to(value)

        match self.empty_target_action:
            case "neg":
                query_num += target.size(0)
            case "pos":
                query_num += target.size(0)
                value += (target.size(0) - non_empty_query_num).to(value)
            case "skip":
                query_num += non_empty_query_num.to(query_num)
            case "error":
                if self.higher_is_better:
                    raise ValueError(
                        "All queries must have at least one positive target."
                    )
                else:
                    raise ValueError(
                        "All queries must have at least one negative target."
                    )

        setattr(self, self.value_name, value)
        setattr(self, self.query_num_name, query_num)

    def compute(self) -> th.Tensor:
        """Compute the metric based on the states accumulated in update."""
        value: th.Tensor = getattr(self, self.value_name)
        total_num: th.Tensor = getattr(self, self.query_num_name)
        return value / total_num if total_num > 0 else th.tensor(0.0).to(value)

    @abstractmethod
    def _metric(self, preds: th.Tensor, target: th.Tensor) -> th.Tensor:
        """Compute a metric over a predictions and target of one update step.
        All queries must have at least one positive target.

        :param preds: A tensor of predictions of shape (num_queries, num_entities).
        :param target: A tensor of targets of shape (num_queries, num_entities).
        :return: A scalar tensor of the metric value.
        """
        raise NotImplementedError


class RetrievalMatrixTopKMetric(RetrievalMatrixMetric):
    def __init__(
        self,
        k: int | None = None,
        empty_target_action: Literal["neg", "pos", "skip", "error"] = "neg",
        **kwargs,
    ) -> None:
        super().__init__(empty_target_action=empty_target_action, **kwargs)
        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")
        self.k = k
