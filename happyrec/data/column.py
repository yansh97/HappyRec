from enum import Enum, unique
from functools import partial
from typing import Final

import numpy as np
import pandas as pd

from ..config.constants import SEQ_SEP

_INT_DTYPES: Final[tuple[type[np.integer], ...]] = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)
"""Legal integer data types in HappyRec."""

_FLOAT_DTYPES: Final[tuple[type[np.floating], ...]] = (np.float16, np.float32)
"""Legal float data types in HappyRec."""


@unique
class ColType(Enum):
    """ColType (column type) defines how to operate on the column data."""

    CAT = "c"
    """Category"""
    CAT_ARR = "ca"
    """Category array of fixed length"""
    CAT_SEQ = "cs"
    """Category sequence of variable length"""
    NUM = "n"
    """Number"""
    NUM_ARR = "na"
    """Number array of fixed length"""
    BOOL = "b"
    """Boolean"""
    STR = "s"
    """String"""
    OBJ = "o"
    """Object (nullable)"""

    def is_cat_like(self) -> bool:
        """Check if the column data type is category-like.

        :return: A bool indicating if the column data type is category-like.
        """
        return self in (ColType.CAT, ColType.CAT_ARR, ColType.CAT_SEQ)

    def is_num_like(self) -> bool:
        """Check if the column data type is number-like.

        :return: A bool indicating if the column data type is number-like.
        """
        return self in (ColType.NUM, ColType.NUM_ARR)

    def check_value(self, series: pd.Series) -> bool:
        """Check if the value of the column is legal.

        :param value: The value of the column.
        :return: A bool indicating if the value of the column is legal.
        """

        def _is_legal_arr(item: np.ndarray, type_, shape) -> bool:
            return (
                pd.notna(item)  # type: ignore
                and isinstance(item, np.ndarray)
                and item.dtype.type == type_
                and item.shape == shape
            )

        def _is_legal_seq(item: np.ndarray, type_) -> bool:
            return (
                pd.notna(item)  # type: ignore
                and isinstance(item, np.ndarray)
                and item.dtype.type == type_
                and len(item.shape) == 1
                and item.shape[0] > 0
            )

        match self:
            case ColType.CAT:
                ret = (
                    series.size > 0
                    and series.dtype.type in _INT_DTYPES
                    and series.notna().all()
                )
            case ColType.CAT_ARR:
                ret = (
                    series.size > 0
                    and pd.notna(series[0])
                    and isinstance(series[0], np.ndarray)
                    and len(series[0].shape) == 1
                    and series[0].shape[0] > 0
                    and series[0].dtype.type in _INT_DTYPES
                    and series.map(
                        partial(
                            _is_legal_arr,
                            type_=series[0].dtype.type,
                            shape=series[0].shape,
                        )
                    ).all()
                )
            case ColType.CAT_SEQ:
                ret = (
                    series.size > 0
                    and pd.notna(series[0])
                    and isinstance(series[0], np.ndarray)
                    and series[0].dtype.type in _INT_DTYPES
                    and series.map(
                        partial(_is_legal_seq, type_=series[0].dtype.type)
                    ).all()
                )
            case ColType.NUM:
                ret = (
                    series.size > 0
                    and series.dtype.type in _INT_DTYPES + _FLOAT_DTYPES
                    and series.notna().all()
                )
            case ColType.NUM_ARR:
                ret = (
                    series.size > 0
                    and pd.notna(series[0])
                    and isinstance(series[0], np.ndarray)
                    and len(series[0].shape) == 1
                    and series[0].shape[0] > 0
                    and series[0].dtype.type in _INT_DTYPES + _FLOAT_DTYPES
                    and series.map(
                        partial(
                            _is_legal_arr,
                            type_=series[0].dtype.type,
                            shape=series[0].shape,
                        )
                    ).all()
                )
            case ColType.BOOL:
                ret = (
                    series.size > 0
                    and series.dtype.type == np.bool_
                    and series.notna().all()
                )
            case ColType.STR:
                ret = (
                    series.size > 0
                    and series.dtype.type == np.object_
                    and series.notna().all()
                    and series.map(lambda x: isinstance(x, str)).all()
                )
            case ColType.OBJ:
                ret = series.size > 0 and series.dtype.type == np.object_
        return ret  # type: ignore

    def downcast(self, series: pd.Series) -> pd.Series:
        """Downcast the numpy dtype of the value.

        :param value: The value of the column.
        :return: The downcasted value of the column.
        """

        def _downcast_integer_arr(value: np.ndarray) -> np.ndarray:
            min_value = value.min()
            max_value = value.max()
            for dtype in _INT_DTYPES:
                if np.iinfo(dtype).min <= min_value <= max_value <= np.iinfo(dtype).max:
                    return value.astype(dtype)
            return value

        match self:
            case ColType.CAT | ColType.NUM:
                if series.dtype.type in _INT_DTYPES:
                    ret = pd.Series(_downcast_integer_arr(series.to_numpy()))
                else:
                    ret = series
            case ColType.CAT_ARR | ColType.CAT_SEQ | ColType.NUM_ARR:
                if series.dtype.type in _INT_DTYPES:
                    flat_value: np.ndarray = np.concatenate(series)
                    sections: list[int] = series.map(len).cumsum()[:-1].tolist()
                    ret = pd.Series(
                        np.split(_downcast_integer_arr(flat_value), sections)
                    )
                else:
                    ret = series
            case ColType.BOOL | ColType.STR | ColType.OBJ:
                ret = series
        return ret

    def from_str_series(self, series: pd.Series) -> pd.Series:
        """Generate the value of the column from the string series.

        :param array: The string series.
        :return: The value of the column.
        """

        def _to_num_series(s: pd.Series) -> pd.Series:
            try:
                return s.astype(np.int64)
            except ValueError:
                return s.astype(np.float32)

        def _str_to_array(str_: str) -> np.ndarray:
            return _to_num_series(
                pd.Series(str_.split(SEQ_SEP), dtype=np.str_)
            ).to_numpy()

        match self:
            case ColType.CAT | ColType.NUM:
                ret = _to_num_series(series)
            case ColType.CAT_ARR | ColType.CAT_SEQ | ColType.NUM_ARR:
                ret = series.map(_str_to_array)
            case ColType.BOOL:
                ret = _to_num_series(series).astype(np.bool_)
            case ColType.STR:
                ret = series.astype(np.str_)
            case ColType.OBJ:
                try:
                    ret = series.map(eval).astype(np.object_)
                except (NameError, SyntaxError):
                    ret = series.astype(np.object_)
        return self.downcast(ret)

    def to_str_array(self, series: pd.Series) -> pd.Series:
        """Convert the value of the column to a string series.

        :param value: The value of the column.
        :return: The string series.
        """

        def _join_arr(arr: np.ndarray) -> str:
            return SEQ_SEP.join(arr.astype(str))

        match self:
            case ColType.CAT | ColType.NUM:
                ret = series.astype(np.str_)
            case ColType.CAT_ARR | ColType.CAT_SEQ | ColType.NUM_ARR:
                ret = series.map(_join_arr).astype(np.str_)
            case ColType.BOOL:
                ret = series.astype(np.int8).astype(np.str_)
            case ColType.STR:
                ret = series.astype(np.str_)
            case ColType.OBJ:
                ret = series.map(repr).astype(np.str_)
        return ret


def get_col_info(col_name: str) -> tuple[str, ColType]:
    """Get the feature name and column type from the column name.

    :param name: The column name.
    :return: The feature name and column type.
    """
    feat_name, col_type = col_name.split(":")
    return feat_name, ColType(col_type)


def get_col_type(col_name: str) -> ColType:
    """Get the column type from the column name.

    :param name: The column name.
    :return: The column type.
    """
    return get_col_info(col_name)[1]


def get_feat_name(col_name: str) -> str:
    """Get the feather name from the column name.

    :param name: The column name.
    :return: The feather name.
    """
    return get_col_info(col_name)[0]


def generate_col_name(feat_name: str, col_type: ColType) -> str:
    """Generate the column name.

    :param feat_name: The feature name.
    :param col_type: The column type.
    :return: The column name.
    """
    return f"{feat_name}:{col_type.value}"
