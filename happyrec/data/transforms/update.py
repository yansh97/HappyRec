import operator
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...utils.logger import logger
from ..data import Data, Source
from ..field import CategoricalType, Field, ItemType
from ..predefined_fields import IID, UID
from .base import DataTransform


class FilterUnusedUIDsAndIIDs(DataTransform):
    """Filter out unused user IDs and item IDs."""

    def check(self, data: Data) -> None:
        if data.is_splitted:
            raise ValueError("The data is already splitted.")
        if data.has_eval_negative_samples:
            raise ValueError("The data has already generated negative samples.")

    def transform(self, data: Data) -> Data:
        """Filter out unused user IDs and item IDs.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        array_in: Callable[[Any, np.ndarray], np.ndarray] = np.vectorize(
            operator.contains
        )

        interaction_frame = data.frames[Source.INTERACTION]
        user_frame = data.frames[Source.USER]
        item_frame = data.frames[Source.ITEM]

        interaction_mask = np.full(interaction_frame.num_elements, True)
        num_interactions = interaction_frame.num_elements
        uid_set = set(user_frame[UID].value)
        interaction_mask &= array_in(uid_set, interaction_frame[UID].value)
        iid_set = set(item_frame[IID].value)
        interaction_mask &= array_in(iid_set, interaction_frame[IID].value)
        interaction_frame = interaction_frame.loc_elements[interaction_mask]
        logger.debug(
            "Filtered interactions: %d -> %d",
            num_interactions,
            interaction_frame.num_elements,
        )

        num_users = user_frame.num_elements
        uid_set = set(interaction_frame[UID].value)
        user_frame = user_frame.loc_elements[array_in(uid_set, user_frame[UID].value)]
        logger.debug("Filtered users: %d -> %d", num_users, user_frame.num_elements)

        num_items = item_frame.num_elements
        iid_set = set(interaction_frame[IID].value)
        item_frame = item_frame.loc_elements[array_in(iid_set, item_frame[IID].value)]
        logger.debug("Filtered items: %d -> %d", num_items, item_frame.num_elements)

        data.frames[Source.INTERACTION] = interaction_frame
        data.frames[Source.USER] = user_frame
        data.frames[Source.ITEM] = item_frame

        return data


class FactorizeCategoricalFields(DataTransform):
    """Factorize categorical fields."""

    def check(self, data: Data) -> None:
        if data.is_splitted:
            raise ValueError("The data is already splitted.")
        if data.has_eval_negative_samples:
            raise ValueError("The data has already generated negative samples.")

    def transform(self, data: Data) -> Data:
        """Factorize categorical fields.

        :param data: The data to be transformed.
        :return: The transformed data.
        """

        def _factorize_1d_array(array: np.ndarray) -> np.ndarray:
            codes = pd.Series(array).factorize(sort=True)[0]
            if codes.min() == -1:
                raise ValueError("Found null value in the array.")
            return codes

        def _convert_categorical_field(field: Field) -> None:
            if not isinstance(field.ftype, CategoricalType):
                return
            match field.ftype.item_type:
                case ItemType.SCALAR:
                    field.value = _factorize_1d_array(field.value)
                case ItemType.ARRAY:
                    field.value = _factorize_1d_array(field.value.reshape(-1)).reshape(
                        field.value.shape
                    )
                case ItemType.SEQUENCE:
                    field.value = np.fromiter(
                        np.split(
                            _factorize_1d_array(np.concatenate(field.value)),
                            np.vectorize(len)(field.value).cumsum()[:-1].tolist(),
                        ),
                        dtype=np.object_,
                        count=len(field.value),
                    )

        # convert uid fields and iid fields
        interaction_frame = data.frames[Source.INTERACTION]
        user_frame = data.frames[Source.USER]
        item_frame = data.frames[Source.ITEM]

        uid_values = np.concatenate(
            [interaction_frame[UID].value, user_frame[UID].value]
        )
        uid_codes = _factorize_1d_array(uid_values)
        interaction_frame[UID].value = uid_codes[: interaction_frame.num_elements]
        user_frame[UID].value = uid_codes[interaction_frame.num_elements :]

        iid_values = np.concatenate(
            [interaction_frame[IID].value, item_frame[IID].value]
        )
        iid_codes = _factorize_1d_array(iid_values)
        interaction_frame[IID].value = iid_codes[: interaction_frame.num_elements]
        item_frame[IID].value = iid_codes[interaction_frame.num_elements :]

        # convert other categorical fields
        for frame in data.frames.values():
            for field in frame.values():
                _convert_categorical_field(field)

        return data


# def _get_num_info(
#     data: Data, col_id: str
# ) -> tuple[list[tuple[Frame, str]], int | float, int | float]:
#     frame_infos: list[tuple[Frame, str]] = []
#     min_value = np.inf
#     max_value = -np.inf
#     for frame in (data.inter_frame, data.user_frame, data.item_frame):
#         for name in frame:
#             _, id_, dtype = get_col_info(name)
#             if id_ == col_id:
#                 if dtype != ScalarType.NUM:
#                     raise ValueError(f"invalid num dtype of feature: {name}")
#                 frame_infos.append((frame, name))
#                 min_value = min(min_value, frame[name].min())
#                 max_value = max(max_value, frame[name].max())
#     return frame_infos, min_value, max_value


# def _cut_num_values(
#     data: Data,
#     frame_infos: list[tuple[Frame, str]],
#     bins: list[int | float],
# ) -> None:
#     for frame, name in frame_infos:
#         nums = frame[name]
#         del frame[name]
#         codes: np.ndarray = (
#             pd.cut(nums, bins=bins, right=False, include_lowest=True).codes + 1
#         )  # 0是padding值
#         if frame is data.user_frame or frame is data.item_frame:
#             codes[0] = 0  # 添加padding值
#         type_, id_, _ = get_col_info(name)
#         new_name = generate_col_name(type_, id_, ScalarType.CATE)
#         frame[new_name] = codes


# class CutNumValuesIntoBins(DataTransform):
#     """Cut the values of a number column into bins.

#     This transform will cut the values of number columns into bins, and then
#     replace the original columns with new category columns. All the number columns
#     with the same column id will be cut into the same bins.
#     """

#     # pylint: disable=too-few-public-methods

#     def __init__(self, col_id: str, bins: list[int | float]):
#         """Initialize the transform.

#         :param col_id: The column id of the number column
#         :param bins: The boundaries of the bins. The length of the list should be
#             equal to the number of bins plus 1. All the bins are left-closed and
#             right-open.
#         """
#         # [bins[0], bins[1]), [bins[1], bins[2]), ...
#         self.col_id = col_id
#         self.bins = bins

#     def transform(self, data: Data) -> Data:
#         logger.debug("cut num values of %s by bins", self.col_id)

#         frame_infos, min_value, max_value = _get_num_info(data, self.col_id)

#         if len(frame_infos) == 0:
#             raise ValueError(f"no num feature of {self.col_id}")
#         if min_value < self.bins[0]:
#             raise ValueError(f"min value of {self.col_id} is less than bins[0]")
#         if max_value >= self.bins[-1]:
#             raise ValueError(
#                 f"max value of {self.col_id} is greater than or equal to bins[-1]"
#             )

#         _cut_num_values(data, frame_infos, self.bins)

#         return data


# class CutNumValuesIntoExpBins(DataTransform):
#     """Cut the values of a number column into exponentially increasing bins.

#     This transform will cut the values of number columns into exponentially increasing
#     bins, and then replace the original columns with new category columns. All the
#     number columns with the same column id will be cut into the same bins.
#     """

#     # pylint: disable=too-few-public-methods

#     def __init__(self, col_id: str, left: int | float, start: int, base: int = 2):
#         """Initialize the transform.

#         Bins are exponentially increasing, and all the bins are left-closed and
#         right-open.

#             [left, start), [start, start * base), [start * base, start * base * base),
#             ...

#         :param col_id: The column id of the number column.
#         :param left: The left boundary of the first bin.
#         :param start: The left boundary of the second bin.
#         :param base: The base of the exponential. Default: ``2``.
#         :raises ValueError: If ``left`` is greater than or equal to ``start``.
#         :raises ValueError: If ``start`` is less than or equal to ``0``.
#         :raises ValueError: If ``base`` is less than or equal to ``1``.
#         """
#         # [left, start), [start, start * base), [start * base, start * base * base),
# ...
#         if left >= start:
#             raise ValueError("left must be less than start")
#         if start < 1:
#             raise ValueError(f"start must be greater than or equal to 1, got {start}")
#         if base < 2:
#             raise ValueError(f"base must be greater than or equal to 2, got {base}")
#         self.col_id = col_id
#         self.left = left
#         self.base = base
#         self.start = start

#     def transform(self, data: Data) -> Data:
#         logger.debug("cut num values of %s by exp bins", self.col_id)

#         feat_infos, min_value, max_value = _get_num_info(data, self.col_id)

#         if len(feat_infos) == 0:
#             raise ValueError(f"no num feature of {self.col_id}")

#         bins = self._get_bins(min_value, max_value)
#         _cut_num_values(data, feat_infos, bins)

#         return data

#     def _get_bins(self, min_value: int | float, max_value: int | float):
#         if min_value < self.left:
#             raise ValueError(f"min value of {self.col_id} is less than left")
#         start = self.start
#         bins = [self.left, start]
#         while start < max_value:
#             start *= self.base
#             bins.append(start)
#         return bins
