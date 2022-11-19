from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique
from os import PathLike
from typing import Any, SupportsIndex, overload

import numpy as np

from ..constants import SEQ_SEP
from ..utils.type import is_typed_sequence


@unique
class ScalarType(Enum):
    """ScaleType defines the data type of the scalar."""

    BOOL = "bool"
    """Boolean scalar."""
    INT = "int"
    """Integer scalar."""
    CATEGORICAL = "categorical"
    """Categorical scalar."""
    FLOAT = "float"
    """Float scalar."""
    STR = "str"
    """String scalar."""
    IMAGE = "image"
    """Image path."""
    OBJECT = "o"
    """Object scalar."""

    def _valid_dtypes(self) -> tuple:
        match self:
            case ScalarType.BOOL:
                return (np.bool_,)
            case ScalarType.INT | ScalarType.CATEGORICAL:
                return (np.int8, np.int16, np.int32, np.int64)
            case ScalarType.FLOAT:
                return (np.float16, np.float32, np.float64)
            case ScalarType.STR:
                return (np.object_,)
            case ScalarType.IMAGE:
                return (np.object_,)
            case ScalarType.OBJECT:
                return (np.object_,)

    def _to_1d_str_array(self, value: np.ndarray) -> np.ndarray:
        match self:
            case ScalarType.BOOL:
                return value.astype(np.int8).astype(np.str_).astype(np.object_)
            case ScalarType.INT | ScalarType.CATEGORICAL | ScalarType.FLOAT:
                return value.astype(np.str_).astype(np.object_)
            case ScalarType.STR | ScalarType.IMAGE:
                return value.astype(np.object_)
            case ScalarType.OBJECT:
                return np.vectorize(repr)(value).astype(np.object_)

    def _from_1d_str_array(self, value: np.ndarray) -> np.ndarray:
        match self:
            case ScalarType.BOOL:
                return value.astype(np.int8).astype(np.bool_)
            case ScalarType.INT | ScalarType.CATEGORICAL:
                return value.astype(np.int64)
            case ScalarType.FLOAT:
                return value.astype(np.float32)
            case ScalarType.STR | ScalarType.IMAGE:
                return value.astype(np.object_)
            case ScalarType.OBJECT:
                try:
                    return np.vectorize(eval)(value).astype(np.object_)
                except (NameError, SyntaxError):
                    return value.astype(np.object_)


@unique
class ItemType(Enum):
    """ItemType defines the type of the item in the field."""

    SCALAR = "scalar"
    """Scalar."""
    ARRAY = "array"
    """1D array of fixed length."""
    SEQUENCE = "sequence"
    """1D sequence of variable length."""

    def _valid_ndim(self) -> int:
        """Return the valid numpy ndim for the item type."""
        match self:
            case ItemType.SCALAR:
                return 1
            case ItemType.ARRAY:
                return 2
            case ItemType.SEQUENCE:
                return 1


@dataclass(frozen=True, slots=True)
class FieldType:
    """FieldType defines the type of the field."""

    scalar_type: ScalarType
    """The scalar type."""
    item_type: ItemType
    """The item type."""

    def __str__(self) -> str:
        return f"{self.item_type.value}<{self.scalar_type.value}>"

    @classmethod
    def from_string(cls, ftype: str) -> "FieldType":
        """Create a field type from a string.

        :param ftype: The string of the field type.
        :return: The field type.
        """
        item_type, scalar_type = ftype[:-1].split("<")
        return cls(ScalarType(scalar_type), ItemType(item_type))


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
        return f"Field<{self.ftype}>{list(self.shape)}"

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

    @property
    def shape(self) -> tuple:
        """Get the shape of the field.

        :return: The shape of the field.
        """
        match self.ftype.item_type:
            case ItemType.SCALAR | ItemType.ARRAY:
                return self.value.shape
            case ItemType.SEQUENCE:
                seq_lens: np.ndarray = np.vectorize(len)(self.value)
                return self.value.shape + ((seq_lens.min(), seq_lens.max()),)

    def mean(self) -> Any:
        """Calculate the mean value of the field.

        :return: The mean value of the field.
        """
        match (self.ftype.scalar_type, self.ftype.item_type):
            case (ScalarType.INT | ScalarType.FLOAT, ItemType.SCALAR | ItemType.ARRAY):
                return self.value.mean(axis=0)
            case _:
                return None

    def min(self) -> Any:
        """Calculate the min value of the field.

        :return: The min value of the field.
        """
        match (self.ftype.scalar_type, self.ftype.item_type):
            case (
                ScalarType.INT | ScalarType.CATEGORICAL | ScalarType.FLOAT,
                ItemType.SCALAR | ItemType.ARRAY,
            ):
                return self.value.min()
            case (
                ScalarType.INT | ScalarType.CATEGORICAL | ScalarType.FLOAT,
                ItemType.SEQUENCE,
            ):
                return np.concatenate(self.value).min()
            case _:
                return None

    def max(self) -> Any:
        """Calculate the max value of the field.

        :return: The max value of the field.
        """
        match (self.ftype.scalar_type, self.ftype.item_type):
            case (
                ScalarType.INT | ScalarType.CATEGORICAL | ScalarType.FLOAT,
                ItemType.SCALAR | ItemType.ARRAY,
            ):
                return self.value.max()
            case (
                ScalarType.INT | ScalarType.CATEGORICAL | ScalarType.FLOAT,
                ItemType.SEQUENCE,
            ):
                return np.concatenate(self.value).max()
            case _:
                return None

    def validate(self) -> "Field":  # noqa: C901
        """Validate the field.

        :raises ValueError: The field is invalid.
        :return: The validated field itself.
        """

        def is_str_array(arr: np.ndarray) -> bool:
            return np.vectorize(isinstance)(arr, str).all()

        if self.value.size == 0:
            raise ValueError("The field is empty.")

        if self.value.ndim != self.ftype.item_type._valid_ndim():
            raise ValueError("The field has invalid ndim.")

        if self.ftype.item_type in (ItemType.SCALAR, ItemType.ARRAY):
            if self.value.dtype.type not in self.ftype.scalar_type._valid_dtypes():
                raise ValueError("The field has invalid dtype.")

            if (
                self.ftype.scalar_type != ScalarType.OBJECT
                and np.isnan(self.value).any()
            ):
                raise ValueError("Some values of the field are NaN.")

            if self.ftype.scalar_type in (
                ScalarType.STR,
                ScalarType.IMAGE,
            ) and not is_str_array(self.value):
                raise ValueError("Some items of the field are not str.")

        if self.ftype.item_type == ItemType.SEQUENCE:
            if self.value.dtype != object:
                raise ValueError("The field has invalid dtype.")

            if not np.vectorize(isinstance)(self.value, np.ndarray).all():
                raise ValueError("Some items of the field are not numpy.ndarray.")

            if not np.vectorize(lambda item: item.size > 0)(self.value).all():
                raise ValueError("Some items of the field are empty.")

            if not np.vectorize(lambda item: item.ndim == 1)(self.value).all():
                raise ValueError("Some items of the field have invalid ndim.")

            valid_dtypes = self.ftype.scalar_type._valid_dtypes()

            if not np.vectorize(lambda item: item.dtype.type in valid_dtypes)(
                self.value
            ).all():
                raise ValueError("Some items of the field have invalid dtype.")

            all_value = np.concatenate(self.value)

            if (
                self.ftype.scalar_type != ScalarType.OBJECT
                and np.isnan(all_value).any()
            ):
                raise ValueError("Some items of the field contain NaN.")

            if self.ftype.scalar_type in (
                ScalarType.STR,
                ScalarType.IMAGE,
            ) and not is_str_array(all_value):
                raise ValueError(
                    "Some items of the field contain other types instead of str."
                )

        return self

    def downcast(self) -> "Field":
        """Downcast the value of the field to the smallest data type.

        :return: The downcasted field itself.
        """

        def _downcast_int_array(value: np.ndarray) -> np.ndarray:
            min_value = value.min()
            max_value = value.max()
            for dtype in ScalarType.INT._valid_dtypes():
                if np.iinfo(dtype).min <= min_value <= max_value <= np.iinfo(dtype).max:
                    return value.astype(dtype)
            return value

        match (self.ftype.scalar_type, self.ftype.item_type):
            case (
                ScalarType.INT | ScalarType.CATEGORICAL,
                ItemType.SCALAR | ItemType.ARRAY,
            ):
                self.value = _downcast_int_array(self.value)
            case (ScalarType.INT | ScalarType.CATEGORICAL, ItemType.SEQUENCE):
                self.value = np.fromiter(
                    np.split(
                        _downcast_int_array(np.concatenate(self.value)),
                        np.vectorize(len)(self.value).cumsum()[:-1].tolist(),
                    ),
                    dtype=np.object_,
                    count=len(self.value),
                )  # 使用np.array可能会将嵌套数组转化为二维数组
        return self

    def copy(self) -> "Field":
        """Copy the field.

        :return: The copied field.
        """
        return Field(self.ftype, self.value.copy())

    def _to_str_array(self) -> np.ndarray:
        def _join_1d_str_array(arr: np.ndarray) -> str:
            return SEQ_SEP.join(self.ftype.scalar_type._to_1d_str_array(arr))

        match self.ftype.item_type:
            case ItemType.SCALAR:
                return self.ftype.scalar_type._to_1d_str_array(self.value)
            case ItemType.ARRAY:
                return np.fromiter(
                    map(_join_1d_str_array, self.value),
                    dtype=np.object_,
                    count=len(self.value),
                )
            case ItemType.SEQUENCE:
                return np.vectorize(_join_1d_str_array)(self.value).astype(np.object_)

    @classmethod
    def _from_str_array(cls, ftype: FieldType, array: np.ndarray) -> "Field":
        def _split_1d_str_array(s: str) -> np.ndarray:
            return ftype.scalar_type._from_1d_str_array(
                np.array(s.split(SEQ_SEP), dtype=np.object_)
            )

        match ftype.item_type:
            case ItemType.SCALAR:
                value = ftype.scalar_type._from_1d_str_array(array)
            case ItemType.ARRAY:
                value = np.vstack(list(map(_split_1d_str_array, array)))
            case ItemType.SEQUENCE:
                value = np.fromiter(
                    map(_split_1d_str_array, array), dtype=np.object_, count=len(array)
                )
        return Field(ftype, value).downcast()

    def to_npy(self, path: str | PathLike) -> None:
        """Save the field to a .npy file.

        :param file: The file path.
        """
        np.save(path, self.value)

    @classmethod
    def from_npy(cls, ftype: FieldType, path: str | PathLike) -> "Field":
        """Load the field from a .npy file.

        :param ftype: The field type.
        :param file: The file path.
        :return: The loaded field.
        """
        return Field(ftype, np.load(path))

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
        if ftype.item_type == ItemType.ARRAY:
            shape = fields[0].value.shape[1:]
            if not all(field.value.shape[1:] == shape for field in fields):
                raise ValueError("Some fields in `fields` have different shapes")
        value = np.concatenate([field.value for field in fields], axis=0)
        return Field(ftype, value)
