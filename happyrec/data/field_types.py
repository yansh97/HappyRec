from dataclasses import dataclass

import numpy as np

from ..constants import SEQ_SEP
from ..utils.asserts import assert_type
from .core import ElementType, FieldType


def _assert_1d_array(array: np.ndarray) -> None:
    if array.ndim != 1:
        raise ValueError


def _get_smallest_int_dtype(min_value: int, max_value: int) -> str:
    for dtype in ("int8", "int16", "int32", "int64"):
        if np.iinfo(dtype).min <= min_value and np.iinfo(dtype).max >= max_value:
            return dtype
    raise ValueError


@dataclass(frozen=True, slots=True)
class BoolEtype(ElementType):
    def __str__(self) -> str:
        return "bool"

    def _can_be_nested(self) -> bool:
        return True

    def _can_hold_null(self) -> bool:
        return False

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array)

    def _mean(self, array: np.ndarray) -> float:
        _assert_1d_array(array)
        return float(array.mean())

    def _min(self, array: np.ndarray) -> bool:
        _assert_1d_array(array)
        return bool(array.min())

    def _max(self, array: np.ndarray) -> bool:
        _assert_1d_array(array)
        return bool(array.max())

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype != "bool":
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        return array.astype(np.int8).astype(str).astype(object)


@dataclass(frozen=True, slots=True)
class IntEtype(ElementType):
    def __str__(self) -> str:
        return "int"

    def _can_be_nested(self) -> bool:
        return True

    def _can_hold_null(self) -> bool:
        return False

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array)

    def _mean(self, array: np.ndarray) -> float:
        _assert_1d_array(array)
        return float(array.mean())

    def _min(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return int(array.min())

    def _max(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return int(array.max())

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype not in ("int8", "int16", "int32", "int64"):
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        return array.astype(str).astype(object)


@dataclass(frozen=True, slots=True)
class FloatEtype(ElementType):
    def __str__(self) -> str:
        return "float"

    def _can_be_nested(self) -> bool:
        return True

    def _can_hold_null(self) -> bool:
        return False

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array)

    def _mean(self, array: np.ndarray) -> float:
        _assert_1d_array(array)
        return float(array.mean())

    def _min(self, array: np.ndarray) -> float:
        _assert_1d_array(array)
        return float(array.min())

    def _max(self, array: np.ndarray) -> float:
        _assert_1d_array(array)
        return float(array.max())

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype not in ("float16", "float32", "float64"):
            raise ValueError
        if np.isnan(array).any():
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        return array.astype(str).astype(object)


@dataclass(frozen=True, slots=True)
class CategoryEtype(ElementType):
    def __str__(self) -> str:
        return "category"

    def _can_be_nested(self) -> bool:
        return True

    def _can_hold_null(self) -> bool:
        return False

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array)

    def _mean(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _min(self, array: np.ndarray) -> bool | int | float | str | None:
        _assert_1d_array(array)
        ret = array.min()
        if isinstance(ret, np.generic):
            ret = ret.item()
        return ret

    def _max(self, array: np.ndarray) -> bool | int | float | str | None:
        _assert_1d_array(array)
        ret = array.max()
        if isinstance(ret, np.generic):
            ret = ret.item()
        return ret

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype not in (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "object",
        ):
            raise ValueError
        if dtype in ("float16", "float32", "float64") and np.isnan(array).any():
            raise ValueError
        if dtype == "object" and not np.vectorize(isinstance)(array, str).all():
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        if array.dtype == "object":
            return np.vectorize(repr)(array).astype(object)
        return array.astype(str).astype(object)


@dataclass(frozen=True, slots=True)
class StringEtype(ElementType):
    def __str__(self) -> str:
        return "string"

    def _can_be_nested(self) -> bool:
        return False

    def _can_hold_null(self) -> bool:
        return True

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array) - int(np.vectorize(isinstance)(array, type(None)).sum())

    def _mean(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _min(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _max(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype != "object":
            raise ValueError
        if not np.vectorize(isinstance)(array, str | type(None)).all():
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        str_array = array.copy()
        mask = np.vectorize(isinstance)(array, type(None))
        str_array[mask] = ""
        return str_array


@dataclass(frozen=True, slots=True)
class ImageEtype(ElementType):
    def __str__(self) -> str:
        return "image"

    def _can_be_nested(self) -> bool:
        return False

    def _can_hold_null(self) -> bool:
        return True

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array) - int(np.vectorize(isinstance)(array, type(None)).sum())

    def _mean(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _min(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _max(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype != "object":
            raise ValueError

        def _valid_image(image: np.ndarray | None) -> bool:
            if image is None:
                return True
            return (
                isinstance(image, np.ndarray)
                and image.ndim == 1
                and image.size > 0
                and image.dtype == "uint8"
            )

        if not np.vectorize(_valid_image)(array).all():
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        str_array = np.full(len(array), "<image>", dtype=object)
        mask = np.vectorize(isinstance)(array, type(None))
        str_array[mask] = ""
        return str_array


@dataclass(frozen=True, slots=True)
class ObjectEtype(ElementType):
    def __str__(self) -> str:
        return "object"

    def _can_be_nested(self) -> bool:
        return False

    def _can_hold_null(self) -> bool:
        return True

    def _non_null_count(self, array: np.ndarray) -> int:
        _assert_1d_array(array)
        return len(array) - int(np.vectorize(isinstance)(array, type(None)).sum())

    def _mean(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _min(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _max(self, array: np.ndarray) -> None:
        _assert_1d_array(array)
        return None

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        _assert_1d_array(array)
        if dtype != "object":
            raise ValueError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        _assert_1d_array(array)
        return np.vectorize(repr)(array).astype(object)


@dataclass(frozen=True, slots=True)
class ScalarFtype(FieldType):
    @property
    def name(self) -> str:
        return f"scalar<{self.element_type}>"

    def _dtype(self, value: np.ndarray) -> str:
        return str(value.dtype)

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        if self.element_type._can_hold_null():
            return self.element_type._non_null_count(value)
        return len(value)

    def _mean(self, value: np.ndarray) -> float | None:
        return self.element_type._mean(value)

    def _min(self, value: np.ndarray) -> bool | int | float | str | None:
        return self.element_type._min(value)

    def _max(self, value: np.ndarray) -> bool | int | float | str | None:
        return self.element_type._max(value)

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 1:
            raise ValueError

    def _check_value_dtype(self, value: np.ndarray) -> None:
        dtype = self._dtype(value)
        self.element_type._check_value_data(value, dtype)

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        dtype = self._dtype(value)
        if dtype not in ("int8", "int16", "int32", "int64"):
            return value.copy()
        min_value: int = self._min(value)  # type: ignore
        max_value: int = self._max(value)  # type: ignore
        smallest_dtype = _get_smallest_int_dtype(min_value, max_value)
        return value.astype(smallest_dtype)

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        return self.element_type._to_str_array(value)


@dataclass(frozen=True, slots=True)
class FixedSizeListFtype(FieldType):
    length: int

    def __post_init__(self) -> None:
        FieldType.__post_init__(self)
        if not self.element_type._can_be_nested():
            raise ValueError
        assert_type(self.length, int)
        if self.length <= 0:
            raise ValueError

    @property
    def name(self) -> str:
        return f"list<{self.element_type}, {self.length}>"

    def _dtype(self, value: np.ndarray) -> str:
        return str(value.dtype)

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        if self.element_type._can_hold_null():
            raise ValueError
        return len(value)

    def _mean(self, value: np.ndarray) -> float | None:
        return None

    def _min(self, value: np.ndarray) -> bool | int | float | str | None:
        return self.element_type._min(value.reshape(-1))

    def _max(self, value: np.ndarray) -> bool | int | float | str | None:
        return self.element_type._max(value.reshape(-1))

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 2:
            raise ValueError
        if value.shape[1] != self.length:
            raise ValueError

    def _check_value_dtype(self, value: np.ndarray) -> None:
        dtype = self._dtype(value)
        self.element_type._check_value_data(value.reshape(-1), dtype)

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        dtype = self._dtype(value)
        if dtype not in ("int8", "int16", "int32", "int64"):
            return value.copy()
        min_value: int = self._min(value)  # type: ignore
        max_value: int = self._max(value)  # type: ignore
        smallest_dtype = _get_smallest_int_dtype(min_value, max_value)
        return value.astype(smallest_dtype)

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        def _to_str(x: np.ndarray) -> str:
            return SEQ_SEP.join(self.element_type._to_str_array(x))

        return np.fromiter(map(_to_str, value), dtype=object, count=len(value))


@dataclass(frozen=True, slots=True)
class ListFtype(FieldType):
    def __post_init__(self) -> None:
        FieldType.__post_init__(self)
        if not self.element_type._can_be_nested():
            raise ValueError

    @property
    def name(self) -> str:
        return f"list<{self.element_type}>"

    def _dtype(self, value: np.ndarray) -> str:
        dtypes = np.vectorize(lambda x: x.dtype)(value)
        if not all(dtypes == dtypes[0]):
            raise ValueError
        return str(dtypes[0])

    def _shape(self, value: np.ndarray) -> tuple:
        seq_lens = np.vectorize(len)(value)
        min_len = int(seq_lens.min())
        max_len = int(seq_lens.max())
        return (len(value), (min_len, max_len))

    def _non_null_count(self, value: np.ndarray) -> int:
        if self.element_type._can_hold_null():
            raise ValueError
        return len(value)

    def _mean(self, value: np.ndarray) -> float | None:
        return None

    def _min(self, value: np.ndarray) -> bool | int | float | str | None:
        dtype = self._dtype(value)
        mask = np.vectorize(lambda x: len(x) > 0)(value)
        value = value[mask]
        min_values = np.vectorize(self.element_type._min)(value).astype(dtype)
        return self.element_type._min(min_values)

    def _max(self, value: np.ndarray) -> bool | int | float | str | None:
        dtype = self._dtype(value)
        mask = np.vectorize(lambda x: len(x) > 0)(value)
        value = value[mask]
        max_values = np.vectorize(self.element_type._max)(value).astype(dtype)
        return self.element_type._max(max_values)

    def _check_value_shape(self, value: np.ndarray) -> None:
        if len(value) == 0:
            raise ValueError
        if value.ndim != 1:
            raise ValueError

    def _check_value_dtype(self, value: np.ndarray) -> None:
        if value.dtype != object:
            raise ValueError

        def _valid_item(item: np.ndarray) -> bool:
            return isinstance(item, np.ndarray) and item.ndim == 1

        if not np.vectorize(_valid_item)(value).all():
            raise ValueError

        dtype = self._dtype(value)
        for item in value:
            self.element_type._check_value_data(item, dtype)

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        dtype = self._dtype(value)
        if dtype not in ("int8", "int16", "int32", "int64"):
            return value.copy()
        min_value: int = self._min(value)  # type: ignore
        max_value: int = self._max(value)  # type: ignore
        smallest_dtype = _get_smallest_int_dtype(min_value, max_value)
        return np.fromiter(
            map(lambda x: x.astype(smallest_dtype), value),
            dtype=object,
            count=len(value),
        )

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        def _to_str(x: np.ndarray) -> str:
            return SEQ_SEP.join(self.element_type._to_str_array(x))

        return np.fromiter(map(_to_str, value), dtype=object, count=len(value))


def bool_() -> FieldType:
    return ScalarFtype(BoolEtype())


def int_() -> FieldType:
    return ScalarFtype(IntEtype())


def float_() -> FieldType:
    return ScalarFtype(FloatEtype())


def category() -> FieldType:
    return ScalarFtype(CategoryEtype())


def string() -> FieldType:
    return ScalarFtype(StringEtype())


def image() -> FieldType:
    return ScalarFtype(ImageEtype())


def object_() -> FieldType:
    return ScalarFtype(ObjectEtype())


def list_(type_: FieldType, length: int | None = None) -> FieldType:
    assert_type(type_, ScalarFtype)
    if length is None:
        return ListFtype(type_.element_type)
    return FixedSizeListFtype(type_.element_type, length)
