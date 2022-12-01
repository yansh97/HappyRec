import operator
from typing import Callable

import numpy as np
import pandas as pd

from ...utils.logger import logger
from ..data import Data, Source
from ..field import CategoricalType, Field, ItemType
from ..predefined_fields import IID, UID
from .base import DataTransform


class FilterUnusedUIDsAndIIDs(DataTransform):
    """Filter out unused user IDs and item IDs."""

    def transform(self, data: Data) -> Data:
        """Filter out unused user IDs and item IDs.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        array_in: Callable[[set, np.ndarray], np.ndarray] = np.vectorize(
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

        logger.debug("Factorizing user IDs ...")
        uid_values = np.concatenate(
            [interaction_frame[UID].value, user_frame[UID].value]
        )
        uid_codes = _factorize_1d_array(uid_values)
        interaction_frame[UID].value = uid_codes[: interaction_frame.num_elements]
        user_frame[UID].value = uid_codes[interaction_frame.num_elements :]

        logger.debug("Factorizing item IDs ...")
        iid_values = np.concatenate(
            [interaction_frame[IID].value, item_frame[IID].value]
        )
        iid_codes = _factorize_1d_array(iid_values)
        interaction_frame[IID].value = iid_codes[: interaction_frame.num_elements]
        item_frame[IID].value = iid_codes[interaction_frame.num_elements :]

        # convert other categorical fields
        for frame in data.frames.values():
            for name, field in frame.items():
                logger.debug("Factorizing field %s ...", name)
                _convert_categorical_field(field)

        return data
