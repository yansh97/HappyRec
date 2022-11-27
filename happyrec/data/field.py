import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, ClassVar, SupportsIndex, overload

import numpy as np

from ..constants import SEQ_SEP
from ..utils.type import is_typed_sequence


@unique
class ScalarType(Enum):
    """ScalarType defines the data type of the scalar."""

    BOOL = "bool"
    """Boolean."""

    INT = "int"
    """Integer."""

    FLOAT = "float"
    """Float."""

    def __repr__(self) -> str:
        return str(self)

    @property
    def _valid_dtypes(self) -> tuple[str, ...]:
        match self:
            case ScalarType.BOOL:
                return ("bool",)
            case ScalarType.INT:
                return ("int8", "int16", "int32", "int64")
            case ScalarType.FLOAT:
                return ("float16", "float32", "float64")

    def _contains_nan(self, value: np.ndarray) -> bool:
        match self:
            case ScalarType.BOOL | ScalarType.INT:
                return False
            case ScalarType.FLOAT:
                return bool(np.isnan(value).any())

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        match self:
            case ScalarType.INT:
                min_, max_ = value.min(), value.max()
                for dtype in self._valid_dtypes:
                    dmin_, dmax_ = np.iinfo(dtype).min, np.iinfo(dtype).max
                    if dmin_ <= min_ and max_ <= dmax_:
                        return value.astype(dtype)
                return value
            case ScalarType.BOOL | ScalarType.FLOAT:
                return value

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        match self:
            case ScalarType.BOOL:
                return value.astype("int8").astype(str).astype(object)
            case ScalarType.INT | ScalarType.FLOAT:
                return value.astype(str).astype(object)

    def _to_str(self, value: np.ndarray) -> str:
        return SEQ_SEP.join(self._to_str_array(value))


@unique
class ItemType(Enum):
    """ItemType defines the type of the item in the field."""

    SCALAR = "scalar"
    """Scalar."""

    ARRAY = "array"
    """1D array of fixed length."""

    SEQUENCE = "sequence"
    """1D sequence of variable length."""

    def __repr__(self) -> str:
        return str(self)

    @property
    def _valid_ndim(self) -> int:
        """Return the valid numpy ndim for the item type."""
        match self:
            case ItemType.SCALAR | ItemType.SEQUENCE:
                return 1
            case ItemType.ARRAY:
                return 2

    def _shape(self, value: np.ndarray) -> tuple:
        match self:
            case ItemType.SCALAR | ItemType.ARRAY:
                return value.shape
            case ItemType.SEQUENCE:
                seq_lens: np.ndarray = np.vectorize(len)(value)
                return value.shape + ((seq_lens.min(), seq_lens.max()),)

    def _mean(self, value: np.ndarray) -> Any:
        match self:
            case ItemType.SCALAR:
                return value.mean()
            case ItemType.ARRAY | ItemType.SEQUENCE:
                return None

    def _min(self, value: np.ndarray) -> Any:
        match self:
            case ItemType.SCALAR | ItemType.ARRAY:
                return value.min()
            case ItemType.SEQUENCE:
                return np.concatenate(value).min()

    def _max(self, value: np.ndarray) -> Any:
        match self:
            case ItemType.SCALAR | ItemType.ARRAY:
                return value.max()
            case ItemType.SEQUENCE:
                return np.concatenate(value).max()


@dataclass(frozen=True, slots=True)
class FieldType(ABC):
    """FieldType defines the data type of the field."""

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def _shape(self, value: np.ndarray) -> tuple:
        ...

    @abstractmethod
    def _non_null_count(self, value: np.ndarray) -> int:
        ...

    @abstractmethod
    def _mean(self, value: np.ndarray) -> Any:
        ...

    @abstractmethod
    def _min(self, value: np.ndarray) -> Any:
        ...

    @abstractmethod
    def _max(self, value: np.ndarray) -> Any:
        ...

    @abstractmethod
    def _memory_usage(self, value: np.ndarray) -> int:
        ...

    @abstractmethod
    def _validate(self, value: np.ndarray) -> None:
        ...

    @abstractmethod
    def _downcast(self, value: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class NumericType(FieldType):
    """NumericType defines the type of the numeric field."""

    item_type: ItemType
    """The item type."""

    scalar_type: ScalarType
    """The scalar type."""

    def __str__(self) -> str:
        return f"{self.item_type.value}<{self.scalar_type.value}>"

    def _shape(self, value: np.ndarray) -> tuple:
        return self.item_type._shape(value)

    def _non_null_count(self, value: np.ndarray) -> int:
        return value.shape[0]

    def _mean(self, value: np.ndarray) -> Any:
        return self.item_type._mean(value)

    def _min(self, value: np.ndarray) -> Any:
        return self.item_type._min(value)

    def _max(self, value: np.ndarray) -> Any:
        return self.item_type._max(value)

    def _memory_usage(self, value: np.ndarray) -> int:
        match self.item_type:
            case ItemType.SCALAR | ItemType.ARRAY:
                return value.nbytes
            case ItemType.SEQUENCE:
                return value.nbytes + np.vectorize(lambda x: x.nbytes)(value).sum()

    def _validate(self, value: np.ndarray) -> None:  # noqa: C901
        if value.size == 0:
            raise ValueError("The field is empty.")
        if value.ndim != self.item_type._valid_ndim:
            raise ValueError("The field has invalid ndim.")

        if self.item_type in (ItemType.SCALAR, ItemType.ARRAY):
            if value.dtype not in self.scalar_type._valid_dtypes:
                raise ValueError("The field has invalid dtype.")
            if self.scalar_type._contains_nan(value):
                raise ValueError("Some values of the field are NaN.")

        if self.item_type == ItemType.SEQUENCE:
            if value.dtype != "object":
                raise ValueError("The field has invalid dtype.")
            if not np.vectorize(isinstance)(value, np.ndarray).all():
                raise ValueError("Some items of the field are not numpy.ndarray.")
            if not np.vectorize(lambda x: x.size > 0)(value).all():
                raise ValueError("Some items of the field are empty.")
            if not np.vectorize(lambda x: x.ndim == 1)(value).all():
                raise ValueError("Some items of the field have invalid ndim.")
            valid_dtypes = self.scalar_type._valid_dtypes
            if not np.vectorize(lambda x: x.dtype in valid_dtypes)(value).all():
                raise ValueError("Some items of the field have invalid dtype.")
            if self.scalar_type._contains_nan(np.concatenate(value)):
                raise ValueError("Some items of the field contain NaN.")

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        match self.item_type:
            case ItemType.SCALAR | ItemType.ARRAY:
                return self.scalar_type._downcast(value)
            case ItemType.SEQUENCE:
                return np.fromiter(
                    np.split(
                        self.scalar_type._downcast(np.concatenate(value)),
                        np.vectorize(len)(value).cumsum()[:-1].tolist(),
                    ),
                    dtype="object",
                    count=len(value),
                )

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        match self.item_type:
            case ItemType.SCALAR:
                return self.scalar_type._to_str_array(value)
            case ItemType.ARRAY:
                return np.fromiter(
                    map(self.scalar_type._to_str, value),
                    dtype="object",
                    count=len(value),
                )
            case ItemType.SEQUENCE:
                return np.vectorize(self.scalar_type._to_str)(value).astype(object)


@dataclass(frozen=True, slots=True)
class CategoricalType(FieldType):
    """CategoricalType defines the type of the categorical field."""

    item_type: ItemType
    """The item type."""

    _VALID_TYPES: ClassVar[tuple[str, ...]] = ScalarType.INT._valid_dtypes + ("object",)

    def __str__(self) -> str:
        return f"{self.item_type.value}<categorical>"

    def _shape(self, value: np.ndarray) -> tuple:
        return self.item_type._shape(value)

    def _non_null_count(self, value: np.ndarray) -> int:
        return value.shape[0]

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return self.item_type._min(value)

    def _max(self, value: np.ndarray) -> Any:
        return self.item_type._max(value)

    def _memory_usage(self, value: np.ndarray) -> int:
        match self.item_type:
            case ItemType.SCALAR | ItemType.ARRAY:
                memory = value.nbytes
                if value.dtype == "object":
                    memory += np.vectorize(sys.getsizeof)(value).sum()
                return memory
            case ItemType.SEQUENCE:
                memory = value.nbytes + np.vectorize(lambda x: x.nbytes)(value).sum()
                concated_value = np.concatenate(value)
                if concated_value.dtype == "object":
                    memory += np.vectorize(sys.getsizeof)(concated_value).sum()
                return memory

    def _validate(self, value: np.ndarray) -> None:  # noqa: C901
        if value.size == 0:
            raise ValueError("The field is empty.")
        if value.ndim != self.item_type._valid_ndim:
            raise ValueError("The field has invalid ndim.")

        if self.item_type in (ItemType.SCALAR, ItemType.ARRAY):
            if value.dtype not in self._VALID_TYPES:
                raise ValueError("The field has invalid dtype.")
            if (
                value.dtype == "object"
                and np.vectorize(lambda x: x is None)(value).any()
            ):
                raise ValueError("Some values of the field are None.")

        if self.item_type == ItemType.SEQUENCE:
            if value.dtype != "object":
                raise ValueError("The field has invalid dtype.")
            if not np.vectorize(isinstance)(value, np.ndarray).all():
                raise ValueError("Some items of the field are not numpy.ndarray.")
            if not np.vectorize(lambda x: x.size > 0)(value).all():
                raise ValueError("Some items of the field are empty.")
            if not np.vectorize(lambda x: x.ndim == 1)(value).all():
                raise ValueError("Some items of the field have invalid ndim.")
            if not np.vectorize(lambda x: x.dtype in self._VALID_TYPES)(value).all():
                raise ValueError("Some items of the field have invalid dtype.")
            concated_value = np.concatenate(value)
            if (
                concated_value.dtype == "object"
                and np.vectorize(lambda x: x is None)(concated_value).any()
            ):
                raise ValueError("Some items of the field contain None.")

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        helper = NumericType(self.item_type, ScalarType.INT)
        match self.item_type:
            case ItemType.SCALAR | ItemType.ARRAY:
                if value.dtype == "object":
                    return value
                return helper._downcast(value)
            case ItemType.SEQUENCE:
                concated_value = np.concatenate(value)
                if concated_value.dtype == "object":
                    return value
                return helper._downcast(concated_value)

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        helper = NumericType(self.item_type, ScalarType.INT)

        def _to_str_array(x: np.ndarray) -> np.ndarray:
            return np.vectorize(repr)(x).astype(object)

        def _to_str(x: np.ndarray) -> str:
            return SEQ_SEP.join(_to_str_array(x))

        match self.item_type:
            case ItemType.SCALAR:
                if value.dtype == "object":
                    return _to_str_array(value)
                return helper._to_str_array(value)
            case ItemType.ARRAY:
                if value.dtype == "object":
                    return np.fromiter(
                        map(_to_str, value), dtype="object", count=len(value)
                    )
                return helper._to_str_array(value)
            case ItemType.SEQUENCE:
                if value.dtype == "object":
                    return np.vectorize(_to_str)(value).astype(object)
                return helper._to_str_array(value)


@dataclass(frozen=True, slots=True)
class TextType(FieldType):
    """TextType defines the type of the text field."""

    def __str__(self) -> str:
        return "text"

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return np.vectorize(lambda x: x is not None)(value).sum()

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return None

    def _max(self, value: np.ndarray) -> Any:
        return None

    def _memory_usage(self, value: np.ndarray) -> int:
        return value.nbytes + np.vectorize(sys.getsizeof)(value).sum()

    def _validate(self, value: np.ndarray) -> None:
        if value.size == 0:
            raise ValueError("The value is empty.")
        if value.ndim != 1:
            raise ValueError("The value has invalid dimension.")
        if value.dtype != "object":
            raise ValueError("The value has invalid dtype.")
        if not np.vectorize(isinstance)(value, str | None).all():
            raise ValueError("Some items are not str.")

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        return value

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        return value


@dataclass(frozen=True, slots=True)
class ImageType(FieldType):
    """ImageType defines the type of the image field."""

    def __str__(self) -> str:
        return "image"

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return np.vectorize(lambda x: x is not None)(value).sum()

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return None

    def _max(self, value: np.ndarray) -> Any:
        return None

    def _memory_usage(self, value: np.ndarray) -> int:
        return (
            value.nbytes
            + np.vectorize(lambda x: 0 if x is None else x.nbytes)(value).sum()
        )

    def _validate(self, value: np.ndarray) -> None:
        def _validate_item(x: np.ndarray | None) -> bool:
            if x is None:
                return True
            if not isinstance(x, np.ndarray):
                return False
            if x.ndim != 1 or x.dtype != "uint8" or x.size == 0:
                return False
            return True

        if value.size == 0:
            raise ValueError("The value is empty.")
        if value.ndim != 1:
            raise ValueError("The value has invalid dimension.")
        if value.dtype != "object":
            raise ValueError("The value has invalid dtype.")
        if not np.vectorize(_validate_item)(value).all():
            raise ValueError("Some items are not valid.")

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        return value

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        def _to_str(x: np.ndarray | None) -> str | None:
            if x is None:
                return None
            return "<image>"

        return np.vectorize(_to_str)(value).astype(object)


@dataclass(frozen=True, slots=True)
class ObjectType(FieldType):
    """ObjectType defines the type of the object field."""

    def __str__(self) -> str:
        return "object"

    def _shape(self, value: np.ndarray) -> tuple:
        return value.shape

    def _non_null_count(self, value: np.ndarray) -> int:
        return np.vectorize(lambda x: x is not None)(value).sum()

    def _mean(self, value: np.ndarray) -> Any:
        return None

    def _min(self, value: np.ndarray) -> Any:
        return None

    def _max(self, value: np.ndarray) -> Any:
        return None

    def _memory_usage(self, value: np.ndarray) -> int:
        return value.nbytes + np.vectorize(sys.getsizeof)(value).sum()

    def _validate(self, value: np.ndarray) -> None:
        if value.size == 0:
            raise ValueError("The value is empty.")
        if value.ndim != 1:
            raise ValueError("The value has invalid dimension.")
        if value.dtype != "object":
            raise ValueError("The value has invalid dtype.")

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        return value

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        return np.vectorize(repr)(value).astype(object)


@dataclass(eq=False, slots=True)
class Field:
    """Field contains the type and value of the field."""

    ftype: FieldType
    """The type of the field."""
    value: np.ndarray
    """The value of the field."""

    def __len__(self) -> int:
        return len(self.value)

    def __str__(self) -> str:
        return f"{self.ftype}{list(self.shape())}"

    @overload
    def __getitem__(self, key: slice) -> "Field":
        ...

    @overload
    def __getitem__(self, key: Sequence[SupportsIndex]) -> "Field":
        ...

    @overload
    def __getitem__(self, key: np.ndarray) -> "Field":
        ...

    @overload
    def __getitem__(self, key: SupportsIndex) -> Any:
        # np.ndarray is a subclass of SupportsIndex, so this overload must be
        # placed after the overload for np.ndarray.
        ...

    def __getitem__(self, key):
        """Get the items of the field.

        :param key: The index of the items.
        :raises TypeError: The parameter ``key`` is not a
            :external:py:class:`typing.SupportsIndex`, :external:py:class:`slice`,
            :external:py:class:`collections.abc.Sequence` or
            :external:py:class:`numpy.ndarray`.
        :return: The single item or the items of multiple items.
        """
        if isinstance(key, slice):
            return Field(self.ftype, self.value[key])
        if is_typed_sequence(key, SupportsIndex):
            key = np.array(key)
            return Field(self.ftype, self.value[key])
        if isinstance(key, np.ndarray):
            return Field(self.ftype, self.value[key])
        if isinstance(key, SupportsIndex):
            return self.value[key]
        raise TypeError(
            "The parameter `key` is not a SupportsIndex, slice, "
            "sequence[SupportsIndex] or numpy.ndarray"
        )

    def shape(self) -> tuple:
        """Get the shape of the field.

        :return: The shape of the field.
        """
        return self.ftype._shape(self.value)

    def non_null_count(self) -> int:
        """Get the number of non-null values.

        :return: The number of non-null values.
        """
        return self.ftype._non_null_count(self.value)

    def mean(self) -> Any:
        """Calculate the mean value of the field.

        :return: The mean value of the field.
        """
        return self.ftype._mean(self.value)

    def min(self) -> Any:
        """Calculate the min value of the field.

        :return: The min value of the field.
        """
        return self.ftype._min(self.value)

    def max(self) -> Any:
        """Calculate the max value of the field.

        :return: The max value of the field.
        """
        return self.ftype._max(self.value)

    def memory_usage(self) -> int:
        """Get the memory usage of the field.

        :return: The memory usage of the field.
        """
        return self.ftype._memory_usage(self.value)

    def validate(self) -> "Field":  # noqa: C901
        """Validate the field.

        :raises ValueError: The field is invalid.
        :return: The validated field itself.
        """
        self.ftype._validate(self.value)
        return self

    def downcast(self) -> "Field":
        """Downcast the value of the field to the smallest data type.

        :return: The downcasted field itself.
        """
        self.value = self.ftype._downcast(self.value)
        return self

    def copy(self) -> "Field":
        """Copy the field.

        :return: The copied field.
        """
        return Field(self.ftype, self.value.copy())

    def _to_str_array(self) -> np.ndarray:
        return self.ftype._to_str_array(self.value)

    def _info(self) -> dict[str, Any]:
        ret = {
            "field type": str(self.ftype),
            "numpy dtype": self.value.dtype.type.__name__,
            "shape": self.shape(),
            "non-null count": self.non_null_count(),
            "mean": self.mean(),
            "min": self.min(),
            "max": self.max(),
            "memory usage": self.memory_usage(),
        }
        for key, value in ret.items():
            if isinstance(value, np.generic):
                ret[key] = value.item()
        return ret

    @staticmethod
    def concat(fields: Sequence["Field"]) -> "Field":
        """Concatenate the given fields.

        :param fields: The fields to be concatenated.
        :raise ValueError: The fields have different types or shapes.
        :return: The concatenated field.
        """
        ftype = fields[0].ftype
        if not all(field.ftype == ftype for field in fields):
            raise ValueError("Some fields in `fields` have different types.")
        shape = fields[0].value.shape[1:]
        if not all(field.value.shape[1:] == shape for field in fields):
            raise ValueError("Some fields in `fields` have different shapes")
        value = np.concatenate([field.value for field in fields], axis=0)
        return Field(ftype, value)
