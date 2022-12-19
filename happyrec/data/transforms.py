import operator
from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np

from ..constants import IID, LABEL, UID
from ..utils.asserts import assert_never_type, assert_type, is_typed_list
from ..utils.logger import logger
from .core import Data, Field, Frame, Source
from .data_info import CategoryInfo
from .field_types import (
    CategoryEtype,
    FieldType,
    FixedSizeListFtype,
    ImageFtype,
    ListFtype,
    ScalarFtype,
    TextFtype,
    category,
)

_array_contains: Callable[[set, np.ndarray], np.ndarray] = np.vectorize(
    operator.contains
)


@dataclass(frozen=True, slots=True)
class DataTransform:
    def __call__(self, data: Data) -> Data:
        logger.info("Transforming data by %s ...", repr(self))
        if data.has_phase_mask:
            raise ValueError
        if data.has_eval_negative_samples:
            raise ValueError
        data = self.transform(data)
        data.validate()
        return data

    def transform(self, data: Data) -> Data:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Compose(DataTransform):
    transforms: list[DataTransform]

    def __post_init__(self) -> None:
        is_typed_list(self.transforms, DataTransform)

    @classmethod
    def from_transforms(cls, *transforms: DataTransform) -> "Compose":
        return cls(list(transforms))

    def transform(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform.transform(data)
        return data


@dataclass(frozen=True, slots=True)
class DowncastDtype(DataTransform):
    def transform(self, data: Data) -> Data:
        return data.downcast()


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
class RemoveDuplicateInteractions(DataTransform):
    keep: Literal["first", "last"] = "first"

    def __post_init__(self) -> None:
        if self.keep not in {"first", "last"}:
            raise ValueError

    def transform(self, data: Data) -> Data:
        interaction_frame = data[Source.INTERACTION]

        uid_codes = np.unique(interaction_frame[UID].value, return_inverse=True)[1]
        iid_codes = np.unique(interaction_frame[IID].value, return_inverse=True)[1]
        codes = uid_codes.astype(object) * len(iid_codes) + iid_codes
        match self.keep:
            case "first":
                indices = np.sort(np.unique(codes, return_index=True)[1])
            case "last":
                indices = np.sort(
                    len(codes) - 1 - np.unique(codes[::-1], return_index=True)[1]
                )
        num_interactions = interaction_frame.num_elements
        interaction_frame = interaction_frame.loc_elements[indices]
        logger.debug(
            "  Filtered interactions: %d -> %d",
            num_interactions,
            interaction_frame.num_elements,
        )

        return Data.from_frames(interaction_frame, data[Source.USER], data[Source.ITEM])


@dataclass(frozen=True, slots=True)
class RemoveInteractionsWithInvalidUsersOrItems(DataTransform):
    def transform(self, data: Data) -> Data:
        interaction_frame = data[Source.INTERACTION]

        uid_set = set(data[Source.USER][UID].value)
        uid_mask = _array_contains(uid_set, interaction_frame[UID].value)
        iid_set = set(data[Source.ITEM][IID].value)
        iid_mask = _array_contains(iid_set, interaction_frame[IID].value)
        interaction_mask = uid_mask & iid_mask
        interaction_frame = interaction_frame.loc_elements[interaction_mask]
        logger.debug(
            "  Filtered interactions: %d -> %d",
            len(interaction_mask),
            interaction_frame.num_elements,
        )

        return Data.from_frames(interaction_frame, data[Source.USER], data[Source.ITEM])


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
                unique_uids, counts = np.unique(uids, return_counts=True)
                uid_set = set(unique_uids[counts >= self.k])
                interaction_mask &= _array_contains(uid_set, uids)
            if self.filter_by_items:
                unique_iids, counts = np.unique(iids, return_counts=True)
                iid_set = set(unique_iids[counts >= self.k])
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
class FactorizeCategoryFields(DataTransform):
    remove_unused_user_and_item: bool = True
    category_info: CategoryInfo | None = None

    def __post_init__(self) -> None:
        assert_type(self.remove_unused_user_and_item, bool)
        if self.category_info is not None:
            assert_type(self.category_info, CategoryInfo)

    def _remove_unused_user_and_item(self, data: Data) -> Data:
        user_frame = data[Source.USER]
        item_frame = data[Source.ITEM]

        uid_set = set(data[Source.INTERACTION][UID].value)
        user_mask = _array_contains(uid_set, user_frame[UID].value)
        user_frame = user_frame.loc_elements[user_mask]
        logger.debug(
            "  Filtered users: %d -> %d", len(user_mask), user_frame.num_elements
        )

        iid_set = set(data[Source.INTERACTION][IID].value)
        item_mask = _array_contains(iid_set, item_frame[IID].value)
        item_frame = item_frame.loc_elements[item_mask]
        logger.debug(
            "  Filtered items: %d -> %d", len(item_mask), item_frame.num_elements
        )

        return Data.from_frames(data[Source.INTERACTION], user_frame, item_frame)

    def _factorize_array(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unique_value, factorized_value = np.unique(array, return_inverse=True)
        return unique_value, factorized_value.astype(np.uint64)

    def _factorize_scalar_field(self, field_name: str, field: Field) -> Field:
        ftype = cast(ScalarFtype, field.ftype)
        if not isinstance(ftype.element_type, CategoryEtype):
            return field
        logger.debug("  Factorizing field %s ...", field_name)
        unique_value, factorized_value = self._factorize_array(field.value)
        if self.category_info is not None:
            self.category_info.update(field_name, unique_value)
        return Field(field.ftype, factorized_value)

    def _factorize_fixed_size_list_field(self, field_name: str, field: Field) -> Field:
        ftype = cast(FixedSizeListFtype, field.ftype)
        if not isinstance(ftype.element_type, CategoryEtype):
            return field
        logger.debug("  Factorizing field %s ...", field_name)
        flat_value = field.value.reshape(-1)
        unique_value, factorized_flat_value = self._factorize_array(flat_value)
        factorized_value = factorized_flat_value.reshape(field.value.shape)
        if self.category_info is not None:
            self.category_info.update(field_name, unique_value)
        return Field(field.ftype, factorized_value)

    def _factorize_list_field(self, field_name: str, field: Field) -> Field:
        ftype = cast(ListFtype, field.ftype)
        if not isinstance(ftype.element_type, CategoryEtype):
            return field
        logger.debug("  Factorizing field %s ...", field_name)
        flat_value = np.concatenate(field.value)
        unique_value, factorized_flat_value = self._factorize_array(flat_value)
        sections = np.vectorize(len)(field.value).cumsum()[:-1].tolist()
        factorized_value = np.fromiter(
            np.split(factorized_flat_value, sections),
            dtype=object,
            count=len(field.value),
        )
        if self.category_info is not None:
            self.category_info.update(field_name, unique_value)
        return Field(field.ftype, factorized_value)

    def _factorize_field(self, field_name: str, field: Field) -> Field:
        ftype: FieldType = field.ftype
        if isinstance(ftype, ScalarFtype):
            return self._factorize_scalar_field(field_name, field)
        elif isinstance(ftype, FixedSizeListFtype):
            return self._factorize_fixed_size_list_field(field_name, field)
        elif isinstance(ftype, ListFtype):
            return self._factorize_list_field(field_name, field)
        elif isinstance(ftype, TextFtype):
            return field
        elif isinstance(ftype, ImageFtype):
            return field
        else:
            assert_never_type(ftype)

    def _factorize_category_fields(self, data: Data) -> Data:
        interaction_frame = data[Source.INTERACTION]
        user_frame = data[Source.USER]
        item_frame = data[Source.ITEM]

        factorized_interaction_fields = {}
        factorized_user_fields = {}
        factorized_item_fields = {}

        # convert uid fields and iid fields
        uid_field = Field(
            category(),
            np.concatenate([interaction_frame[UID].value, user_frame[UID].value]),
        )
        factorized_uid_field = self._factorize_field(UID, uid_field)
        factorized_interaction_fields[UID] = factorized_uid_field.loc[
            : interaction_frame.num_elements
        ]
        factorized_user_fields[UID] = factorized_uid_field.loc[
            interaction_frame.num_elements :
        ]

        iid_field = Field(
            category(),
            np.concatenate([interaction_frame[IID].value, item_frame[IID].value]),
        )
        factorized_iid_field = self._factorize_field(IID, iid_field)
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
                factorized_frame[name] = self._factorize_field(name, field)

        return Data.from_frames(
            Frame(factorized_interaction_fields),
            Frame(factorized_user_fields),
            Frame(factorized_item_fields),
        )

    def transform(self, data: Data) -> Data:
        if self.remove_unused_user_and_item:
            data = self._remove_unused_user_and_item(data)
        data = self._factorize_category_fields(data)
        return data
