import operator
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pyarrow as pa

from ..constants import (
    IID,
    LABEL,
    SEQ_SEP,
    TEST_MASK,
    TEST_NEG_IIDS,
    TIMESTAMP,
    TRAIN_MASK,
    UID,
    VAL_MASK,
    VAL_NEG_IIDS,
)
from ..utils.asserts import assert_type
from .element_types import BoolEtype, CategoryEtype, ElementType, FloatEtype, IntEtype


@dataclass(frozen=True, slots=True)
class FieldType:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def _can_be_sorted(self) -> bool:
        raise NotImplementedError

    def _dtype(self, value: np.ndarray) -> str:
        raise NotImplementedError

    def _shape(self, value: np.ndarray) -> tuple:
        raise NotImplementedError

    def _non_null_count(self, value: np.ndarray) -> int:
        raise NotImplementedError

    def _mean(self, value: np.ndarray) -> Any:
        raise NotImplementedError

    def _min(self, value: np.ndarray) -> Any:
        raise NotImplementedError

    def _max(self, value: np.ndarray) -> Any:
        raise NotImplementedError

    def _check_value_shape(self, value: np.ndarray) -> None:
        raise NotImplementedError

    def _check_value_data(self, value: np.ndarray) -> None:
        raise NotImplementedError

    def _validate(self, value: np.ndarray) -> None:
        self._check_value_shape(value)
        self._check_value_data(value)

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _to_pa_array(self, value: np.ndarray) -> pa.Array:
        raise NotImplementedError

    def _from_pa_array(self, pa_array: pa.Array) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ScalarFtype(FieldType):
    element_type: ElementType

    def __post_init__(self) -> None:
        assert_type(self.element_type, ElementType)

    @property
    def name(self) -> str:
        return f"scalar<{self.element_type}>"

    def _can_be_sorted(self) -> bool:
        return True

    def _dtype(self, value: np.ndarray) -> str:
        return str(value.dtype)

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return len(value)

    def _mean(self, value: np.ndarray) -> Any:
        return value.mean().item()

    def _min(self, value: np.ndarray) -> Any:
        return value.min().item()

    def _max(self, value: np.ndarray) -> Any:
        return value.max().item()

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 1:
            raise ValueError

    def _check_value_data(self, value: np.ndarray) -> None:
        dtype = self._dtype(value)
        if dtype not in self.element_type._valid_dtypes():
            raise ValueError
        if _has_nan_value(value):
            raise ValueError

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        dtype = _get_downcast_dtype(self, value)
        if dtype is None:
            return value
        return value.astype(dtype)

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        return value.astype(str).astype(object)

    def _to_pa_array(self, value: np.ndarray) -> pa.Array:
        pa_type = self.element_type._convert_dtype_to_pa_type(self._dtype(value))
        return pa.array(value, type=pa_type)

    def _from_pa_array(self, pa_array: pa.Array) -> np.ndarray:
        return pa_array.to_numpy()


@dataclass(frozen=True, slots=True)
class FixedSizeListFtype(FieldType):
    element_type: ElementType
    length: int

    def __post_init__(self) -> None:
        assert_type(self.element_type, ElementType)
        assert_type(self.length, int)
        if self.length <= 0:
            raise ValueError

    @property
    def name(self) -> str:
        return f"list<{self.element_type}, {self.length}>"

    def _can_be_sorted(self) -> bool:
        return False

    def _dtype(self, value: np.ndarray) -> str:
        return str(value.dtype)

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return len(value)

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return value.reshape(-1).min().item()

    def _max(self, value: np.ndarray) -> Any:
        return value.reshape(-1).max().item()

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 2:
            raise ValueError
        if value.shape[1] != self.length:
            raise ValueError

    def _check_value_data(self, value: np.ndarray) -> None:
        dtype = self._dtype(value)
        if dtype not in self.element_type._valid_dtypes():
            raise ValueError
        if _has_nan_value(value):
            raise ValueError

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        dtype = _get_downcast_dtype(self, value)
        if dtype is None:
            return value
        return value.astype(dtype)

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        def _to_str(x: np.ndarray) -> str:
            return SEQ_SEP.join(x.astype(str).astype(object))

        return np.fromiter(map(_to_str, value), dtype=object, count=len(value))

    def _to_pa_array(self, value: np.ndarray) -> pa.Array:
        pa_etype = self.element_type._convert_dtype_to_pa_type(self._dtype(value))
        pa_type = pa.list_(pa_etype, self.length)
        return pa.FixedSizeListArray.from_arrays(value, type=pa_type)

    def _from_pa_array(self, pa_array: pa.Array) -> np.ndarray:
        array = pa_array.flatten().to_numpy()
        return array.reshape(-1, self.length)


@dataclass(frozen=True, slots=True)
class ListFtype(FieldType):
    element_type: ElementType

    def __post_init__(self) -> None:
        assert_type(self.element_type, ElementType)

    @property
    def name(self) -> str:
        return f"list<{self.element_type}>"

    def _can_be_sorted(self) -> bool:
        return False

    def _dtype(self, value: np.ndarray) -> str:
        dtypes: np.ndarray = np.vectorize(operator.attrgetter("dtype"))(value)
        if not (dtypes == dtypes[0]).all():
            raise ValueError
        return str(dtypes[0])

    def _shape(self, value: np.ndarray) -> tuple:
        seq_lens = np.vectorize(len)(value)
        min_len = int(seq_lens.min())
        max_len = int(seq_lens.max())
        return (len(value), (min_len, max_len))

    def _non_null_count(self, value: np.ndarray) -> int:
        return len(value)

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        mask: np.ndarray = np.vectorize(len)(value) > 0
        min_values = np.vectorize(operator.methodcaller("min"))(value)
        return min_values[mask].min().item()

    def _max(self, value: np.ndarray) -> Any:
        mask: np.ndarray = np.vectorize(len)(value) > 0
        max_values = np.vectorize(operator.methodcaller("max"))(value)
        return max_values[mask].max().item()

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 1:
            raise ValueError

    def _check_value_data(self, value: np.ndarray) -> None:
        if value.dtype != object:
            raise ValueError

        if not np.vectorize(isinstance)(value, np.ndarray).all():
            raise ValueError
        if (np.vectorize(operator.attrgetter("ndim"))(value) != 1).any():
            raise ValueError

        dtype = self._dtype(value)
        if dtype not in self.element_type._valid_dtypes():
            raise ValueError
        if np.vectorize(_has_nan_value)(value).any():
            raise ValueError

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        dtype = _get_downcast_dtype(self, value)
        if dtype is None:
            return value
        return np.fromiter(
            map(operator.methodcaller("astype", dtype), value),
            dtype=object,
            count=len(value),
        )

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        def _to_str(x: np.ndarray) -> str:
            return SEQ_SEP.join(x.astype(str).astype(object))

        return np.fromiter(map(_to_str, value), dtype=object, count=len(value))

    def _to_pa_array(self, value: np.ndarray) -> pa.Array:
        pa_etype = self.element_type._convert_dtype_to_pa_type(self._dtype(value))
        pa_type = pa.list_(pa_etype)
        values = np.concatenate(value)
        offsets = np.concatenate([np.array([0]), np.vectorize(len)(value).cumsum()])
        return pa.ListArray.from_arrays(offsets, values, type=pa_type)

    def _from_pa_array(self, pa_array: pa.Array) -> np.ndarray:
        return pa_array.to_numpy()


@dataclass(frozen=True, slots=True)
class TextFtype(FieldType):
    @property
    def name(self) -> str:
        return "text"

    def _can_be_sorted(self) -> bool:
        return False

    def _dtype(self, value: np.ndarray) -> str:
        return str(value.dtype)

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return (value != "").sum().item()

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return None

    def _max(self, value: np.ndarray) -> Any:
        return None

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 1:
            raise ValueError

    def _check_value_data(self, value: np.ndarray) -> None:
        dtype = self._dtype(value)
        if dtype not in ("object",):
            raise ValueError
        if not np.vectorize(isinstance)(value, str).all():
            raise ValueError

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        return value

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        return value

    def _to_pa_array(self, value: np.ndarray) -> pa.Array:
        return pa.array(value, type=pa.string())

    def _from_pa_array(self, pa_array: pa.Array) -> np.ndarray:
        return pa_array.to_numpy()


@dataclass(frozen=True, slots=True)
class ImageFtype(FieldType):
    @property
    def name(self) -> str:
        return "image"

    def _can_be_sorted(self) -> bool:
        return False

    def _dtype(self, value: np.ndarray) -> str:
        return str(value.dtype)

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return (value != b"").sum().item()

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return None

    def _max(self, value: np.ndarray) -> Any:
        return None

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 1:
            raise ValueError

    def _check_value_data(self, value: np.ndarray) -> None:
        dtype = self._dtype(value)
        if dtype not in ("object",):
            raise ValueError
        if not np.vectorize(isinstance)(value, bytes).all():
            raise ValueError

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        return value

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        image_sizes = np.vectorize(len)(value).astype(str).astype(object)
        ret = "image(" + np.vectorize(len)(value).astype(str).astype(object) + "B)"
        ret[image_sizes == "0"] = ""
        return ret

    def _to_pa_array(self, value: np.ndarray) -> pa.Array:
        return pa.array(value, type=pa.binary())

    def _from_pa_array(self, pa_array: pa.Array) -> np.ndarray:
        return pa_array.to_numpy()


def bool_() -> FieldType:
    return ScalarFtype(BoolEtype())


def int_() -> FieldType:
    return ScalarFtype(IntEtype())


def float_() -> FieldType:
    return ScalarFtype(FloatEtype())


def category() -> FieldType:
    return ScalarFtype(CategoryEtype())


def list_(type_: FieldType, length: int | None = None) -> FieldType:
    assert_type(type_, ScalarFtype)
    type_ = cast(ScalarFtype, type_)
    if length is None:
        return ListFtype(type_.element_type)
    return FixedSizeListFtype(type_.element_type, length)


def text() -> FieldType:
    return TextFtype()


def image() -> FieldType:
    return ImageFtype()


def _has_nan_value(array: np.ndarray) -> bool:
    return bool(np.isnan(array.reshape(-1)).any())


def _get_downcast_dtype(ftype: FieldType, value: np.ndarray) -> str | None:
    INT_DTYPES = ("int8", "int16", "int32", "int64")
    UINT_DTYPES = ("uint8", "uint16", "uint32", "uint64")

    value_dtype = ftype._dtype(value)
    if value_dtype in INT_DTYPES:
        DTYPES = INT_DTYPES
    elif value_dtype in UINT_DTYPES:
        DTYPES = UINT_DTYPES
    else:
        return None

    min_value = ftype._min(value)
    max_value = ftype._max(value)
    for dtype in DTYPES:
        if np.iinfo(dtype).min <= min_value and np.iinfo(dtype).max >= max_value:
            return dtype
    raise ValueError


def _convert_pa_type_to_ftype(pa_type: pa.DataType) -> FieldType:
    if pa.types.is_boolean(pa_type):
        return bool_()
    if pa.types.is_signed_integer(pa_type):
        return int_()
    if pa.types.is_floating(pa_type):
        return float_()
    if pa.types.is_unsigned_integer(pa_type):
        return category()
    if pa.types.is_fixed_size_list(pa_type):
        pa_etype = pa_type.value_type
        length = pa_type.list_size
        return list_(_convert_pa_type_to_ftype(pa_etype), length)
    if pa.types.is_list(pa_type):
        pa_etype = pa_type.value_type
        return list_(_convert_pa_type_to_ftype(pa_etype))
    if pa.types.is_string(pa_type):
        return text()
    if pa.types.is_binary(pa_type):
        return image()
    raise ValueError


_BASE_FIELD_CHECKER: dict[str, Callable[[FieldType], bool]] = {
    UID: lambda ftype: ftype == category(),
    IID: lambda ftype: ftype == category(),
    LABEL: lambda ftype: ftype == int_() or ftype == float_(),
    TIMESTAMP: lambda ftype: ftype == int_(),
    TRAIN_MASK: lambda ftype: ftype == bool_(),
    VAL_MASK: lambda ftype: ftype == bool_(),
    TEST_MASK: lambda ftype: ftype == bool_(),
    VAL_NEG_IIDS: lambda ftype: isinstance(ftype, FixedSizeListFtype)
    and ftype.element_type == category(),
    TEST_NEG_IIDS: lambda ftype: isinstance(ftype, FixedSizeListFtype)
    and ftype.element_type == category(),
}
