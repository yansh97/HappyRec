import copy
import operator
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..utils.asserts import assert_never_type, assert_type, assert_typed_list
from ..utils.logger import logger
from .core import Data, Field, Frame, Source
from .field_types import (
    CategoryEtype,
    FixedSizeListFtype,
    ListFtype,
    ScalarFtype,
    category,
)
from .predefined_fields import IID, LABEL, UID


def _assert_not_splitted(data: Data) -> None:
    if data.is_splitted:
        raise ValueError


def _assert_no_eval_negative_samples(data: Data) -> None:
    if data.has_eval_negative_samples:
        raise ValueError


_array_contains: Callable[[set, np.ndarray], np.ndarray] = np.vectorize(
    operator.contains
)


@dataclass(frozen=True, slots=True)
class DataTransform:
    def __call__(self, data: Data) -> Data:
        logger.debug("Transforming data by %s ...", repr(self))
        _assert_not_splitted(data)
        _assert_no_eval_negative_samples(data)
        data = self.transform(data)
        data.validate()
        return data

    def transform(self, data: Data) -> Data:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Compose(DataTransform):
    transforms: list[DataTransform]

    def __post_init__(self) -> None:
        assert_typed_list(self.transforms, DataTransform)

    def transform(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform.transform(data)
        return data


@dataclass(frozen=True, slots=True)
class RemoveNegativeInteractions(DataTransform):
    def transform(self, data: Data) -> Data:
        interaction_frame = data.frames[Source.INTERACTION]
        user_frame = data.frames[Source.USER]
        item_frame = data.frames[Source.ITEM]

        interaction_mask = interaction_frame[LABEL].value > 0
        interaction_frame = interaction_frame.loc_elements[interaction_mask]
        logger.debug(
            "  Filtered interactions: %d -> %d",
            len(interaction_mask),
            interaction_frame.num_elements,
        )

        return Data(interaction_frame, user_frame, item_frame)


@dataclass(frozen=True, slots=True)
class FilterByUnusedUIDsAndIIDs(DataTransform):
    def transform(self, data: Data) -> Data:
        interaction_frame = data.frames[Source.INTERACTION]
        user_frame = data.frames[Source.USER]
        item_frame = data.frames[Source.ITEM]

        uid_set = set(user_frame[UID].value)
        uid_mask = _array_contains(uid_set, interaction_frame[UID].value)
        iid_set = set(item_frame[IID].value)
        iid_mask = _array_contains(iid_set, interaction_frame[IID].value)
        interaction_mask = uid_mask & iid_mask
        interaction_frame = interaction_frame.loc_elements[interaction_mask]
        logger.debug(
            "  Filtered interactions: %d -> %d",
            len(interaction_mask),
            interaction_frame.num_elements,
        )

        uid_set = set(interaction_frame[UID].value)
        user_mask = _array_contains(uid_set, user_frame[UID].value)
        user_frame = user_frame.loc_elements[user_mask]
        logger.debug(
            "  Filtered users: %d -> %d", len(user_mask), user_frame.num_elements
        )

        iid_set = set(interaction_frame[IID].value)
        item_mask = _array_contains(iid_set, item_frame[IID].value)
        item_frame = item_frame.loc_elements[item_mask]
        logger.debug(
            "  Filtered items: %d -> %d", len(item_mask), item_frame.num_elements
        )

        return Data(interaction_frame, user_frame, item_frame)


@dataclass(frozen=True, slots=True)
class FilterKCoreInteractions(DataTransform):
    k: int
    recursion: bool = True
    ignore_negative_interactions: bool = True

    def __post_init__(self) -> None:
        assert_type(self.k, int)
        if self.k <= 0:
            raise ValueError
        assert_type(self.recursion, bool)
        assert_type(self.ignore_negative_interactions, bool)

    def transform(self, data: Data) -> Data:
        interaction_frame = data.frames[Source.INTERACTION]
        user_frame = data.frames[Source.USER]
        item_frame = data.frames[Source.ITEM]

        uids = interaction_frame[UID].value
        iids = interaction_frame[IID].value
        if self.ignore_negative_interactions:
            positive_mask = interaction_frame[LABEL].value > 0
            uids = uids[positive_mask]
            iids = iids[positive_mask]
        while True:
            uid_counter = Counter(uids)
            uid_set = set(uid for uid, count in uid_counter.items() if count >= self.k)
            uid_mask = _array_contains(uid_set, uids)
            iid_counter = Counter(iids)
            iid_set = set(iid for iid, count in iid_counter.items() if count >= self.k)
            iid_mask = _array_contains(iid_set, iids)
            interaction_mask = uid_mask & iid_mask
            uids = uids[interaction_mask]
            iids = iids[interaction_mask]
            logger.debug(
                "  Filtered %s interactions: %d -> %d",
                "positive" if self.ignore_negative_interactions else "all",
                len(interaction_mask),
                len(uids),
            )
            if not self.recursion or len(uids) == len(interaction_mask):
                break

        uid_mask = _array_contains(set(uids), interaction_frame[UID].value)
        iid_mask = _array_contains(set(iids), interaction_frame[IID].value)
        interaction_mask = uid_mask & iid_mask
        interaction_frame = interaction_frame.loc_elements[interaction_mask]
        logger.debug(
            "  Filtered interactions: %d -> %d",
            len(interaction_mask),
            interaction_frame.num_elements,
        )

        return Data(interaction_frame, user_frame, item_frame)


@dataclass(frozen=True, slots=True)
class DowncastDtype(DataTransform):
    def transform(self, data: Data) -> Data:
        return data.downcast()


@dataclass(frozen=True, slots=True)
class FactorizeCategoricalFields(DataTransform):
    def transform(self, data: Data) -> Data:
        def _factorize_array(array: np.ndarray) -> np.ndarray:
            return np.unique(array, return_inverse=True)[1]

        def _factorize_field(field: Field) -> Field:
            if not isinstance(field.ftype.element_type, CategoryEtype):
                return copy.copy(field)

            if isinstance(field.ftype, ScalarFtype):
                value = _factorize_array(field.value)
            elif isinstance(field.ftype, FixedSizeListFtype):
                value = _factorize_array(field.value.reshape(-1)).reshape(
                    field.value.shape
                )
            elif isinstance(field.ftype, ListFtype):
                value = np.fromiter(
                    np.split(
                        _factorize_array(np.concatenate(field.value)),
                        np.vectorize(len)(field.value).cumsum()[:-1].tolist(),
                    ),
                    dtype=object,
                    count=len(field.value),
                )
            else:
                assert_never_type(field.ftype)
            return Field(field.ftype, value)

        interaction_frame = data.frames[Source.INTERACTION]
        user_frame = data.frames[Source.USER]
        item_frame = data.frames[Source.ITEM]

        interaction_fields = {}
        user_fields = {}
        item_fields = {}

        # convert uid fields and iid fields
        logger.debug("  Factorizing UIDs ...")
        uid_value = np.concatenate(
            [interaction_frame[UID].value, user_frame[UID].value]
        )
        uid_field = Field(category(), uid_value)
        factorized_uid_field = _factorize_field(uid_field)
        interaction_fields[UID] = factorized_uid_field.loc[
            : interaction_frame.num_elements
        ]
        user_fields[UID] = factorized_uid_field.loc[interaction_frame.num_elements :]

        logger.debug("  Factorizing IIDs ...")
        iid_value = np.concatenate(
            [interaction_frame[IID].value, item_frame[IID].value]
        )
        iid_field = Field(category(), iid_value)
        factorized_iid_field = _factorize_field(iid_field)
        interaction_fields[IID] = factorized_iid_field.loc[
            : interaction_frame.num_elements
        ]
        item_fields[IID] = factorized_iid_field.loc[interaction_frame.num_elements :]

        # convert other categorical fields
        for frame, fields in zip(
            (interaction_frame, user_frame, item_frame),
            (interaction_fields, user_fields, item_fields),
        ):
            for name, field in frame.items():
                if name in (UID, IID):
                    continue
                logger.debug("  Factorizing field %s ...", name)
                fields[name] = _factorize_field(field)

        return Data(Frame(interaction_fields), Frame(user_fields), Frame(item_fields))
