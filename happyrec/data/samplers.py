import operator
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum, unique
from typing import Any, Callable

import numpy as np

from ..constants import DEFAULT_SEED
from ..utils.asserts import assert_type
from ..utils.logger import logger
from .core import Data, Field, Frame, Phase, Source
from .field_types import category, list_
from .predefined_fields import IID, UID
from .transforms import assert_no_eval_negative_samples


@unique
class SampleDistribution(Enum):
    UNIFORM = "uniform"
    POPULARITY = "popularity"

    def __repr__(self) -> str:
        return str(self)


@dataclass(slots=True)
class Sampler:
    def setup(self, data: Data, phase: Phase) -> None:
        raise NotImplementedError

    def sample_iids(self, size: int) -> np.ndarray:
        raise NotImplementedError

    def sample_iids_by_uids(self, uids: np.ndarray, size: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class RecSampler(Sampler):
    _: KW_ONLY
    distribution: SampleDistribution = SampleDistribution.UNIFORM
    repeatable: bool = False
    seed: int = DEFAULT_SEED

    rng: np.random.Generator = field(init=False, repr=False, hash=False, compare=False)

    iids: np.ndarray | None = field(init=False, repr=False, hash=False, compare=False)
    item_probs: np.ndarray | None = field(
        init=False, repr=False, hash=False, compare=False
    )
    uid_to_index: dict[Any, int] | None = field(
        init=False, repr=False, hash=False, compare=False
    )
    user_iids_set: np.ndarray | None = field(
        init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self) -> None:
        assert_type(self.distribution, SampleDistribution)
        assert_type(self.repeatable, bool)
        assert_type(self.seed, int)
        self.rng = np.random.default_rng(self.seed)
        self.iids = None
        self.item_probs = None
        self.uid_to_index = None
        self.user_iids_set = None

    def setup(self, data: Data, phase: Phase) -> None:
        interaction_frame = data[Source.INTERACTION]
        interaction_mask = np.full(interaction_frame.num_elements, False)
        for p in Phase:
            interaction_mask |= interaction_frame[phase.mask_field]
            if p == phase:
                break
        interaction_frame = interaction_frame.loc_fields[[UID, IID]]
        interaction_frame = interaction_frame.loc_elements[interaction_mask]

        user_frame = data[Source.USER]
        item_frame = data[Source.ITEM]

        self.iids = item_frame[IID].value

        if self.distribution == SampleDistribution.POPULARITY:
            item_count_dict = defaultdict(int)
            unique_iids, counts = np.unique(
                interaction_frame[IID].value, return_counts=True
            )
            item_count_dict.update(zip(unique_iids, counts))
            item_counts = np.vectorize(operator.getitem)(
                item_count_dict, item_frame[IID].value
            )
            self.item_probs = item_counts / interaction_frame.num_elements

        if not self.repeatable:
            self.uid_to_index = dict(
                zip(user_frame[UID].value, range(user_frame.num_elements))
            )
            iids_set_dict: dict[Any, set] = defaultdict(set)
            for uid, frame in interaction_frame.groupby(by=UID):
                iids_set = set(frame[IID].value)
                if len(iids_set) == len(self.iids):
                    raise ValueError
                iids_set_dict[uid] = iids_set
            self.user_iids_set = np.vectorize(operator.getitem)(
                iids_set_dict, user_frame[UID].value
            )

    def sample_iids(self, size: int) -> np.ndarray:
        if self.iids is None:
            raise ValueError
        return self.rng.choice(self.iids, size=size, replace=True, p=self.item_probs)

    def sample_iids_by_uids(self, uids: np.ndarray, size: int) -> np.ndarray:
        if self.iids is None or uids.ndim != 1:
            raise ValueError

        if self.repeatable:
            return self.sample_iids(len(uids) * size).reshape(-1, size)  # (U, S)

        if self.uid_to_index is None or self.user_iids_set is None:
            raise ValueError

        _array_contains: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.vectorize(
            operator.contains
        )

        uid_indices = np.vectorize(operator.getitem)(self.uid_to_index, uids)  # (U,)
        uid_indices = uid_indices.repeat(size)  # (U*S,)

        iids_set: np.ndarray = self.user_iids_set[uid_indices]  # (U*S,)
        illegal_indices = np.arange(len(iids_set))
        iids = self.sample_iids(len(iids_set))  # (U*S,)
        illegal_indices = illegal_indices[_array_contains(iids_set, iids)]  # (N,)
        while len(illegal_indices) > 0:
            iids[illegal_indices] = self.sample_iids(len(illegal_indices))  # (N,)
            illegal_mask = _array_contains(
                iids_set[illegal_indices], iids[illegal_indices]
            )  # (N,)
            illegal_indices = illegal_indices[illegal_mask]
        iids = iids.reshape(-1, size)  # (U, S)
        return iids


@dataclass(frozen=True, slots=True)
class EvalNegativeSampler:
    size: int
    sampler: Sampler

    def __post_init__(self) -> None:
        assert_type(self.size, int)
        if self.size <= 0:
            raise ValueError
        assert_type(self.sampler, Sampler)

    def __call__(self, data: Data) -> Data:
        logger.info("Sampling evaluation negative items by %s ...", repr(self))
        assert_no_eval_negative_samples(data)
        data = self.sample(data)
        data.validate()
        return data

    def sample(self, data: Data) -> Data:
        user_frame = data[Source.USER]

        neg_iids_fields: dict[str, Field] = {}
        for phase in (Phase.VALIDATION, Phase.TEST):
            self.sampler.setup(data, phase)
            neg_iids_field = Field(
                list_(category(), self.size),
                self.sampler.sample_iids_by_uids(user_frame[UID].value, self.size),
            )
            neg_iids_fields[phase.neg_iids_field] = neg_iids_field

        user_frame = Frame({**user_frame, **neg_iids_fields})

        return Data.from_frames(data[Source.INTERACTION], user_frame, data[Source.ITEM])
