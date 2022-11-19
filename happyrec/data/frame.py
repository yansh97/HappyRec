from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import Any, SupportsIndex, overload

import numpy as np
import pandas as pd

from ..constants import FIELD_SEP
from ..utils.type import is_typed_mapping, is_typed_sequence
from .field import Field, FieldType, ItemType, ScalarType


class Frame(MutableMapping[str, Field]):
    """Frame defines a container to store multiple fields containing the same number
    of items. Frame implements all the interfaces for a mutable mapping of strings
    to fields, similar to ``dict[str, Field]``.

    .. note::
        Frame disables the ``__lt__``, ``__le__``, ``__eq__``, ``__ne__``,
        ``__gt__``, ``__ge__``, ``__hash__`` interfaces, as the Frame class is not
        frozen and comparable.
    """

    __slots__ = ["fields"]

    def __init__(self, fields: Mapping[str, Field]) -> None:
        """Initialize the frame with a mapping of names to fields.

        :param fields: The mapping of names to fields.
        :raises TypeError: The parameter ``fields`` must be a mapping of strings to
            fields.
        :raises ValueError: Some fields in ``fields`` contain different number of
            items.
        """
        if not is_typed_mapping(fields, str, Field):
            raise TypeError(
                "The paramter `fields` is not a mapping of strings to fields."
            )

        field_len = len(next(iter(fields.values())))
        for field in fields.values():
            if len(field) != field_len:
                raise ValueError(
                    "Some fields in `fields` contain different number of items."
                )

        self._fields: dict[str, Field] = dict(fields)
        """The underlying data of the frame."""

    # Object
    def __repr__(self) -> str:
        info = ["<Frame"]
        for name, field in self.items():
            info.append(f' "{name}"->{str(field)};')
        info.append(">")
        return "".join(info)

    def __str__(self) -> str:
        return (
            self.to_string() + f"\n[{self.num_items} items x {self.num_fields} fields]"
        )

    def __lt__(self, other: "Frame") -> bool:
        raise NotImplementedError

    def __le__(self, other: "Frame") -> bool:
        raise NotImplementedError

    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError

    def __ne__(self, __o: object) -> bool:
        raise NotImplementedError

    def __gt__(self, __o: "Frame") -> bool:
        raise NotImplementedError

    def __ge__(self, __o: "Frame") -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    # Sized
    def __len__(self) -> int:
        return len(self._fields)

    # Iterable
    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    # Container
    def __contains__(self, key: str) -> bool:
        return key in self._fields

    # Mapping
    def __getitem__(self, key: str) -> Field:
        """Get the field with the given name.

        :param key: The name of the field.
        :return: The field.
        """
        return self._fields[key]

    # MutableMapping
    def __setitem__(self, key: str, value: Field) -> None:
        """Set the field with the given name.

        :param key: The name of the field.
        :param value: The field.
        :raises TypeError: The parameter ``key`` must be a string.
        :raises TypeError: The parameter ``value`` must be a field.
        :raises ValueError: The parameter ``value`` contains different number of items.
        """
        if not isinstance(key, str):
            raise TypeError("The parameter `key` must be a string.")
        if not isinstance(value, Field):
            raise TypeError("The parameter `value` must be a field.")
        if self.num_fields > 0 and len(value) != self.num_items:
            raise ValueError(
                "The parameter `value` contains different number of items."
            )
        self._fields[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete the field with the given name.

        :param key: The name of the field.
        """
        del self._fields[key]

    # Frame
    @property
    def num_items(self) -> int:
        """Get the number of items in the frame."""
        if self.num_fields == 0:
            return 0
        return len(next(iter(self.values())))

    @property
    def loc_items(self) -> "ItemLoc":
        """Select items from the frame.

        ``.loc_items[]`` is used to select items of the frame. The allowed key
        types are listed at :py:meth:`ItemLoc.__getitem__`.
        """
        return self.ItemLoc(self)

    @property
    def num_fields(self) -> int:
        """The number of fields of the frame."""
        return len(self)

    @property
    def loc_fields(self) -> "FieldLoc":
        """Select fields of the frame.

        ``.loc_fields[]`` is used to select columns of the frame. The allowed key
        types are listed at :py:meth:`FieldLoc.__getitem__`.
        """
        return self.FieldLoc(self)

    def describe(self) -> str:
        """Get the description of the frame.

        :return: The description of the frame.
        """
        desc_df = pd.DataFrame(
            {
                "name": list(self.keys()),
                "scalar type": [
                    field.ftype.scalar_type.value for field in self.values()
                ],
                "numpy dtype": [
                    field.value.dtype.type.__name__ for field in self.values()
                ],
                "item type": [field.ftype.item_type.value for field in self.values()],
                "shape": [str(field.shape) for field in self.values()],
                "mean": [str(field.mean()) for field in self.values()],
                "min": [str(field.min()) for field in self.values()],
                "max": [str(field.max()) for field in self.values()],
            }
        )
        return desc_df.to_string()

    def validate(self) -> "Frame":
        """Validate the frame.

        :raises ValueError: Some fields contain different number of items.
        :raises ValueError: Some fields have invalid values.
        :return: The validated frame itself.
        """
        for field in self.values():
            if len(field) != self.num_items:
                raise ValueError("Some fields contain different number of items.")
            field.validate()
        return self

    def downcast(self) -> "Frame":
        """Downcast each field to the smallest possible type.

        :return: The downcasted frame itself.
        """
        for field in self.values():
            field.downcast()
        return self

    def sort(self, name: str) -> "Frame":
        """Sort the frame by one field with scalar item type.

        :param name: The name of the field to sort by.
        :raises ValueError: The field ``name`` does not exist.
        :raises ValueError: The item type of the field is not scalar.
        :return: The sorted frame itself.
        """
        if name not in self:
            raise ValueError(f'Field "{name}" does not exist.')
        if self[name].ftype.item_type != ItemType.SCALAR:
            raise ValueError("The item type of the field is not scalar.")
        indices = np.argsort(self[name].value, axis=0, kind="stable")
        for field in self.values():
            field.value = field.value[indices]
        return self

    def copy(self) -> "Frame":
        """Copy the frame.

        :return: The copied frame.
        """
        return Frame({name: field.copy() for name, field in self.items()})

    def to_csv(self, path: str | PathLike) -> None:
        """Save the frame to a CSV file.

        :param path: The path to the CSV file.
        """
        str_df = pd.DataFrame()
        for name, field in self.items():
            name_and_type = _join_name_and_type(name, field)
            str_df[name_and_type] = field._to_str_array()
        str_df.to_csv(path, sep=FIELD_SEP, index=False, header=True)

    @classmethod
    def from_csv(cls, path: str | PathLike) -> "Frame":
        """Load the frame from a CSV file.

        :param path: The path to the CSV file.
        :return: The loaded frame.
        """
        str_df = pd.read_csv(
            path, sep=FIELD_SEP, header=0, index_col=False, dtype=np.str_
        )
        fields: dict[str, Field] = {}
        for name_and_type in str_df.columns:
            name, ftype = _split_name_and_type(name_and_type)
            array = str_df[name_and_type].to_numpy()
            fields[name] = Field._from_str_array(ftype, array)
        return cls(fields)

    def to_npz(self, path: str | PathLike) -> None:
        """Save the frame to a NPZ file.

        :param path: The path to the NPZ file.
        """
        array_dict: dict[str, np.ndarray] = {}
        for name, field in self.items():
            name_and_type = _join_name_and_type(name, field)
            array_dict[name_and_type] = field.value
        np.savez(path, **array_dict)

    @classmethod
    def from_npz(cls, path: str | PathLike) -> "Frame":
        """Load the frame from a NPZ file.

        :param path: The path to the NPZ file.
        :return: The loaded frame.
        """
        fields: dict[str, Field] = {}
        with np.load(path, allow_pickle=True) as npz:
            for name_and_type in npz:
                name, ftype = _split_name_and_type(name_and_type)
                fields[name] = Field(ftype, npz[name_and_type])
        return cls(fields)

    def to_string(self, max_rows: int | None = 10) -> str:
        """Convert the frame to a informal string representation.

        :param max_rows: The maximum number of rows to show. If ``None``, all rows
            are shown.
        :return: The string representation.
        """
        if max_rows is None or max_rows >= self.num_items:
            str_df = pd.DataFrame(
                {name: field._to_str_array() for name, field in self.items()}
            )
        else:
            head = (max_rows + 1) // 2
            tail = max_rows - head
            head_str_df = pd.DataFrame(
                {name: field[:head]._to_str_array() for name, field in self.items()}
            )
            mid_str_df = pd.DataFrame({name: ["..."] for name in self.keys()})
            tail_str_df = pd.DataFrame(
                {name: field[-tail:]._to_str_array() for name, field in self.items()}
            )
            str_df = pd.concat(
                [head_str_df, mid_str_df, tail_str_df], ignore_index=True
            )
        return str_df.to_string()

    @staticmethod
    def concat(frames: Sequence["Frame"]) -> "Frame":
        """Concatenate the given frames.

        :param frames: The frames to be concatenated.
        :raises ValueError: The frames have different fields.
        :return: The concatenated frame.
        """
        names: set[str] = set(frames[0].keys())
        if not all((set(frame.keys()) == names for frame in frames)):
            raise ValueError("The frames to concatenate have different fields")
        fields: dict[str, Field] = {}
        for name in frames[0]:
            fields[name] = Field.concat([frame[name] for frame in frames])
        return Frame(fields)

    class ItemLoc:
        """ItemLoc defines a helper class to select items from the frame."""

        def __init__(self, frame: "Frame") -> None:
            """Initialize the helper object.

            :param frame: The frame to select items from.
            """
            self.frame = frame

        @overload
        def __getitem__(self, key: slice) -> "Frame":
            ...

        @overload
        def __getitem__(self, key: Sequence[SupportsIndex]) -> "Frame":
            ...

        @overload
        def __getitem__(self, key: np.ndarray) -> "Frame":
            ...

        @overload
        def __getitem__(self, key: SupportsIndex) -> dict[str, Any]:
            # np.ndarray is a subclass of SupportsIndex, so this overload must be
            # placed after the overload for np.ndarray.
            ...

        def __getitem__(self, key):
            """Select items from the frame.

            :param key: The index of the items.
            :raises TypeError: The parameter ``key`` is not a
                :external:py:class:`typing.SupportsIndex`, :external:py:class:`slice`,
                sequence of SupportsIndex or :external:py:class:`numpy.ndarray`.
            :return: The dict of single item or the frame of multiple items.
            """
            if isinstance(key, slice):
                return Frame({name: field[key] for name, field in self.frame.items()})
            if is_typed_sequence(key, SupportsIndex):
                return Frame({name: field[key] for name, field in self.frame.items()})
            if isinstance(key, np.ndarray):
                return Frame({name: field[key] for name, field in self.frame.items()})
            if isinstance(key, SupportsIndex):
                return {k: v[key] for k, v in self.frame.items()}
            raise TypeError(
                "The parameter `key` is not a SupportsIndex, slice, "
                "sequence[SupportsIndex] or numpy.ndarray"
            )

    class FieldLoc:
        """FieldLoc defines a helper class to select fields from the frame."""

        def __init__(self, frame: "Frame") -> None:
            """Initialize the helper object.

            :param frame: The frame to select fields from.
            """
            self.frame = frame

        @overload
        def __getitem__(self, key: str) -> Field:
            ...

        @overload
        def __getitem__(self, key: Sequence[str]) -> "Frame":
            ...

        def __getitem__(self, key):
            """Select fields from the frame.

            :param key: The name of the fields.
            :raises TypeError: The parameter ``key`` is not a string or sequence of
                strings.
            :return: The single field or the frame of multiple fields.
            """
            if isinstance(key, str):
                return self.frame[key]
            if is_typed_sequence(key, str):
                return Frame({k: self.frame[k] for k in key})
            raise TypeError(
                "The parameter `key` is not a string or sequence of strings."
            )


def _join_name_and_type(name: str, field: Field) -> str:
    return f"{name}:{field.ftype.scalar_type.value}:{field.ftype.item_type.value}"


def _split_name_and_type(name_and_type: str) -> tuple[str, FieldType]:
    name, scalar_type, item_type = name_and_type.rsplit(":", 2)
    ftype = FieldType(ScalarType(scalar_type), ItemType(item_type))
    return name, ftype
