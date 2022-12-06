import copy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.asserts import assert_never_type, assert_type, assert_typed_list
from .field_types import FieldType


@dataclass(eq=False, slots=True)
class Field(Sequence):
    _ftype: FieldType
    _value: np.ndarray

    def __post_init__(self) -> None:
        assert_type(self._ftype, FieldType)
        assert_type(self._value, np.ndarray)
        self._ftype._check_value_shape(self._value)

    # implement object methods

    def __str__(self) -> str:
        return f"{self._ftype}{list(self.shape())}"

    def __copy__(self) -> "Field":
        return Field(copy.copy(self._ftype), copy.copy(self._value))

    def __deepcopy__(self, memo=None) -> "Field":
        return Field(copy.deepcopy(self._ftype), copy.deepcopy(self._value))

    # implement Sequence methods

    def __len__(self) -> int:
        return len(self._value)

    def __getitem__(self, index: int) -> Any:
        return self._value[index]

    def __contains__(self, value) -> bool:
        for v in self:
            if v is value or v == value:
                return True
        return False

    def __iter__(self) -> Iterator[Any]:
        return iter(self._value)

    def __reversed__(self) -> Iterator[Any]:
        return reversed(self._value)

    def index(self, value, start: int = 0, stop: int | None = None) -> int:
        if start < 0:
            start = max(len(self) + start, 0)
        if stop is None:
            stop = len(self)
        elif stop < 0:
            stop += len(self)

        for i in range(start, stop):
            v = self[i]
            if v is value or v == value:
                return i
        raise ValueError

    def count(self, value):
        return sum(1 for v in self if v is value or v == value)

    # Field properties

    @property
    def ftype(self) -> FieldType:
        return self._ftype

    @property
    def value(self) -> np.ndarray:
        return self._value

    class Loc:
        def __init__(self, field: "Field") -> None:
            self.field = field

        def __getitem__(
            self, index: slice | list[int] | list[bool] | np.ndarray
        ) -> "Field":
            if isinstance(index, slice):
                return Field(self.field._ftype, self.field._value[index].copy())
            if (
                assert_typed_list(index, int)
                or assert_typed_list(index, bool)
                or isinstance(index, np.ndarray)
            ):
                return Field(self.field._ftype, self.field._value[index])
            assert_never_type(index)

    @property
    def loc(self) -> Loc:
        return self.Loc(self)

    # Field methods

    def dtype(self) -> str:
        return self._ftype._dtype(self._value)

    def shape(self) -> tuple:
        return self._ftype._shape(self._value)

    def non_null_count(self) -> int:
        return self._ftype._non_null_count(self._value)

    def mean(self) -> float | None:
        return self._ftype._mean(self._value)

    def min(self) -> bool | int | float | str | None:
        return self._ftype._min(self._value)

    def max(self) -> bool | int | float | str | None:
        return self._ftype._max(self._value)

    def validate(self) -> None:
        self._ftype._validate(self._value)

    def downcast(self) -> "Field":
        return Field(self._ftype, self._ftype._downcast(self._value))

    def _to_str_array(self) -> np.ndarray:
        return self._ftype._to_str_array(self._value)

    def _info(self) -> dict[str, Any]:
        ret = {
            "field type": str(self._ftype),
            "dtype": self.dtype(),
            "shape": self.shape(),
            "non-null count": self.non_null_count(),
            "mean": self.mean(),
            "min": self.min(),
            "max": self.max(),
        }
        return ret
