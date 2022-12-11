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


def assert_not_splitted(data: Data) -> None:
    if data.is_splitted:
        raise ValueError


def assert_no_eval_negative_samples(data: Data) -> None:
    if data.has_eval_negative_samples:
        raise ValueError


_array_contains: Callable[[set, np.ndarray], np.ndarray] = np.vectorize(
    operator.contains
)


@dataclass(frozen=True, slots=True)
class DataTransform:
    def __call__(self, data: Data) -> Data:
        logger.debug("Transforming data by %s ...", repr(self))
        assert_not_splitted(data)
        assert_no_eval_negative_samples(data)
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
        interaction_frame = data[Source.INTERACTION]

        interaction_mask = interaction_frame[LABEL].value > 0
        interaction_frame = interaction_frame.loc_elements[interaction_mask]
        logger.debug(
            "  Filtered interactions: %d -> %d",
            len(interaction_mask),
            interaction_frame.num_elements,
        )

        return Data.from_frames(interaction_frame, data[Source.USER], data[Source.ITEM])


@dataclass(frozen=True, slots=True)
class FilterByUnusedUIDsAndIIDs(DataTransform):
    def transform(self, data: Data) -> Data:
        interaction_frame = data[Source.INTERACTION]
        user_frame = data[Source.USER]
        item_frame = data[Source.ITEM]

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

        return Data.from_frames(interaction_frame, user_frame, item_frame)


@dataclass(frozen=True, slots=True)
class FilterKCoreInteractions(DataTransform):
    k: int
    recursion: bool = True
    ignore_negative_interactions: bool = True
    filter_by_users: bool = True
    filter_by_items: bool = True

    def __post_init__(self) -> None:
        assert_type(self.k, int)
        if self.k <= 0:
            raise ValueError
        assert_type(self.recursion, bool)
        assert_type(self.ignore_negative_interactions, bool)
        assert_type(self.filter_by_users, bool)
        assert_type(self.filter_by_items, bool)
        if not self.filter_by_users and not self.filter_by_items:
            raise ValueError

    def transform(self, data: Data) -> Data:
        interaction_frame = data[Source.INTERACTION]

        uids = interaction_frame[UID].value
        iids = interaction_frame[IID].value
        if self.ignore_negative_interactions:
            positive_mask = interaction_frame[LABEL].value > 0
            uids = uids[positive_mask]
            iids = iids[positive_mask]
        while True:
            interaction_mask = np.full(len(uids), True)
            if self.filter_by_users:
                uid_set = set(
                    uid for uid, count in Counter(uids).items() if count >= self.k
                )
                interaction_mask &= _array_contains(uid_set, uids)
            if self.filter_by_items:
                iid_set = set(
                    iid for iid, count in Counter(iids).items() if count >= self.k
                )
                interaction_mask &= _array_contains(iid_set, iids)
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

        return Data.from_frames(interaction_frame, data[Source.USER], data[Source.ITEM])


@dataclass(frozen=True, slots=True)
class DowncastDtype(DataTransform):
    def transform(self, data: Data) -> Data:
        return data.downcast()


@dataclass(frozen=True, slots=True)
class FactorizeCategoryFields(DataTransform):
    def transform(self, data: Data) -> Data:
        def _factorize_array(array: np.ndarray) -> np.ndarray:
            return np.unique(array, return_inverse=True)[1]

        def _factorize_field(field: Field) -> Field:
            if not isinstance(field.ftype.element_type, CategoryEtype):
                return field

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

        interaction_frame = data[Source.INTERACTION]
        user_frame = data[Source.USER]
        item_frame = data[Source.ITEM]

        factorized_interaction_fields = {}
        factorized_user_fields = {}
        factorized_item_fields = {}

        # convert uid fields and iid fields
        logger.debug("  Factorizing UIDs ...")
        uid_field = Field(
            category(),
            np.concatenate([interaction_frame[UID].value, user_frame[UID].value]),
        )
        factorized_uid_field = _factorize_field(uid_field)
        factorized_interaction_fields[UID] = factorized_uid_field.loc[
            : interaction_frame.num_elements
        ]
        factorized_user_fields[UID] = factorized_uid_field.loc[
            interaction_frame.num_elements :
        ]

        logger.debug("  Factorizing IIDs ...")
        iid_field = Field(
            category(),
            np.concatenate([interaction_frame[IID].value, item_frame[IID].value]),
        )
        factorized_iid_field = _factorize_field(iid_field)
        factorized_interaction_fields[IID] = factorized_iid_field.loc[
            : interaction_frame.num_elements
        ]
        factorized_item_fields[IID] = factorized_iid_field.loc[
            interaction_frame.num_elements :
        ]

        # convert other categorical fields
        for frame, factorized_frame in zip(
            (interaction_frame, user_frame, item_frame),
            (
                factorized_interaction_fields,
                factorized_user_fields,
                factorized_item_fields,
            ),
        ):
            for name, field in frame.items():
                if name in (UID, IID):
                    continue
                logger.debug("  Factorizing field %s ...", name)
                factorized_frame[name] = _factorize_field(field)

        return Data.from_frames(
            Frame(factorized_interaction_fields),
            Frame(factorized_user_fields),
            Frame(factorized_item_fields),
        )
