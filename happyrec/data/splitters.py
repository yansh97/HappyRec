from dataclasses import KW_ONLY, dataclass, field

import numpy as np

from ..constants import DEFAULT_SEED
from ..utils.asserts import assert_type
from ..utils.logger import logger
from .core import Data, Field, Frame, Partition, Source
from .field_types import bool_, int_
from .predefined_fields import LABEL, TIMESTAMP, UID
from .transforms import assert_no_eval_negative_samples, assert_not_splitted


@dataclass(frozen=True, slots=True)
class Splitter:
    def __call__(self, data: Data) -> Data:
        logger.debug("Splitting data by %s ...", repr(self))
        assert_not_splitted(data)
        assert_no_eval_negative_samples(data)
        data = self.split(data)
        data.validate()
        return data

    def split(self, data: Data) -> Data:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RecSplitter:
    _: KW_ONLY
    ignore_negative_interactions: bool = False
    order_by_time: bool = False
    group_by_uid: bool = False
    seed: int = DEFAULT_SEED
    rng: np.random.Generator = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self) -> None:
        assert_type(self.ignore_negative_interactions, bool)
        assert_type(self.order_by_time, bool)
        assert_type(self.group_by_uid, bool)
        assert_type(self.seed, int)
        object.__setattr__(self, "rng", np.random.default_rng(self.seed))

    def generate_indices(self, index_array: np.ndarray) -> dict[Partition, np.ndarray]:
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
            index_groups = [
                group["index"].value for _, group in index_frame.groupby(by=UID)
            ]
        else:
            index_groups = [index_frame["index"].value]

        group_indices = [self.generate_indices(group) for group in index_groups]
        indices = {
            partition: np.concatenate([group[partition] for group in group_indices])
            for partition in Partition
        }

        mask_fields: dict[str, Field] = {}
        for partition in Partition:
            if len(indices[partition]) == 0:
                raise ValueError
            mask = np.full(interaction_frame.num_elements, False)
            mask[indices[partition]] = True
            mask_fields[partition.mask_field] = Field(bool_(), mask)
        interaction_frame = Frame({**interaction_frame, **mask_fields})

        return Data.from_frames(interaction_frame, data[Source.USER], data[Source.ITEM])


@dataclass(frozen=True, slots=True)
class RatioSplitter(RecSplitter):
    ratio: tuple[float, float, float]

    def __post_init__(self) -> None:
        RecSplitter.__post_init__(self)
        assert_type(self.ratio, tuple)
        if len(self.ratio) != 3:
            raise ValueError
        if any(r <= 0 for r in self.ratio):
            raise ValueError
        normalized_ratio = tuple(r / sum(self.ratio) for r in self.ratio)
        object.__setattr__(self, "ratio", normalized_ratio)

    def generate_indices(self, index_array: np.ndarray) -> dict[Partition, np.ndarray]:
        total_num = len(index_array)
        val_num = int(np.floor(self.ratio[1] * total_num))
        test_num = int(np.floor(self.ratio[2] * total_num))
        return {
            Partition.TRAINING: index_array[: -val_num - test_num],
            Partition.VALIDATION: index_array[-val_num - test_num : -test_num],
            Partition.TEST: index_array[-test_num:],
        }


@dataclass(frozen=True, slots=True)
class LeaveOneOutSplitter(RecSplitter):
    def generate_indices(self, index_array: np.ndarray) -> dict[Partition, np.ndarray]:
        total_num = len(index_array)
        val_num = 1 if total_num >= 3 else 0
        test_num = 1 if total_num >= 3 else 0
        return {
            Partition.TRAINING: index_array[: -val_num - test_num],
            Partition.VALIDATION: index_array[-val_num - test_num : -test_num],
            Partition.TEST: index_array[-test_num:],
        }
