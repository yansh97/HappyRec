from dataclasses import KW_ONLY, dataclass, field

import numpy as np

from ..constants import DEFAULT_SEED, LABEL, TIMESTAMP, UID
from ..utils.asserts import assert_type
from ..utils.logger import logger
from .core import Data, Field, Frame, Phase, Source
from .field_types import bool_, int_
from .transforms import assert_no_eval_negative_samples, assert_not_splitted


@dataclass(frozen=True, slots=True)
class Splitter:
    def __call__(self, data: Data) -> Data:
        logger.info("Splitting data by %s ...", repr(self))
        assert_not_splitted(data)
        assert_no_eval_negative_samples(data)
        data = self.split(data)
        data.validate()
        return data

    def split(self, data: Data) -> Data:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RecSplitter(Splitter):
    _: KW_ONLY
    ignore_negative_interactions: bool
    order_by_time: bool
    group_by_uid: bool
    seed: int = DEFAULT_SEED
    rng: np.random.Generator = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self) -> None:
        assert_type(self.ignore_negative_interactions, bool)
        assert_type(self.order_by_time, bool)
        assert_type(self.group_by_uid, bool)
        assert_type(self.seed, int)
        object.__setattr__(self, "rng", np.random.default_rng(self.seed))

    def generate_indices(self, index_array: np.ndarray) -> dict[Phase, np.ndarray]:
        raise NotImplementedError

    def split(self, data: Data) -> Data:
        interaction_frame = data[Source.INTERACTION]

        base_fields = data.base_fields[Source.INTERACTION]
        index_field = Field(int_(), np.arange(interaction_frame.num_elements))
        index_frame = Frame(
            {"index": index_field, **interaction_frame.loc_fields[base_fields]}
        )

        if self.ignore_negative_interactions:
            index_mask = index_frame[LABEL].value > 0
            index_frame = index_frame.loc_elements[index_mask]

        if self.order_by_time:
            index_frame = index_frame.sort_values(by=TIMESTAMP, kind="stable")
        else:
            perm_indices = self.rng.permutation(index_frame.num_elements)
            index_frame = index_frame.loc_elements[perm_indices]

        if self.group_by_uid:
            index_groups: list[np.ndarray] = [
                group["index"].value for _, group in index_frame.groupby(by=UID)
            ]
        else:
            index_groups = [index_frame["index"].value]

        group_indices = [self.generate_indices(group) for group in index_groups]
        indices = {
            phase: np.concatenate([group[phase] for group in group_indices])
            for phase in Phase
        }

        mask_fields: dict[str, Field] = {}
        for phase in Phase:
            if len(indices[phase]) == 0:
                raise ValueError
            mask = np.full(interaction_frame.num_elements, False)
            mask[indices[phase]] = True
            mask_fields[phase.mask_field] = Field(bool_(), mask)
        interaction_frame = Frame({**interaction_frame, **mask_fields})

        logger.debug("  # raw interactions: %d", interaction_frame.num_elements)
        num_training = len(indices[Phase.TRAINING])
        num_validation = len(indices[Phase.VALIDATION])
        num_test = len(indices[Phase.TEST])
        total_num = num_training + num_validation + num_test
        logger.debug("  # total interactions: %d", total_num)
        logger.debug("  # training interactions: %d", num_training)
        logger.debug("  # validation interactions: %d", num_validation)
        logger.debug("  # test interactions: %d", num_test)

        return Data.from_frames(interaction_frame, data[Source.USER], data[Source.ITEM])


@dataclass(frozen=True, slots=True)
class RatioSplitter(RecSplitter):
    ratio: tuple[float, float, float]
    _: KW_ONLY
    ignore_negative_interactions: bool = False
    order_by_time: bool = False
    group_by_uid: bool = False

    def __post_init__(self) -> None:
        RecSplitter.__post_init__(self)
        assert_type(self.ratio, tuple)
        if len(self.ratio) != 3:
            raise ValueError
        if any(r <= 0 for r in self.ratio):
            raise ValueError
        normalized_ratio = tuple(r / sum(self.ratio) for r in self.ratio)
        object.__setattr__(self, "ratio", normalized_ratio)

    def generate_indices(self, index_array: np.ndarray) -> dict[Phase, np.ndarray]:
        total_num = len(index_array)
        val_num = int(np.floor(self.ratio[1] * total_num))
        test_num = int(np.floor(self.ratio[2] * total_num))
        return {
            Phase.TRAINING: index_array[: -val_num - test_num],
            Phase.VALIDATION: index_array[-val_num - test_num : -test_num],
            Phase.TEST: index_array[-test_num:],
        }


@dataclass(frozen=True, slots=True)
class LeaveOneOutSplitter(RecSplitter):
    _: KW_ONLY
    ignore_negative_interactions: bool = True
    order_by_time: bool = True
    group_by_uid: bool = True

    def generate_indices(self, index_array: np.ndarray) -> dict[Phase, np.ndarray]:
        total_num = len(index_array)
        val_num = 1 if total_num >= 3 else 0
        test_num = 1 if total_num >= 3 else 0
        return {
            Phase.TRAINING: index_array[: -val_num - test_num],
            Phase.VALIDATION: index_array[-val_num - test_num : -test_num],
            Phase.TEST: index_array[-test_num:],
        }
