from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, SupportsIndex, overload

import numpy as np
import pandas as pd

from ..constants import FIELD_SEP
from ..utils.logger import logger
from ..utils.type import is_typed_mapping, is_typed_sequence
from .field import (  # noqa: F401
    CategoricalType,
    Field,
    FieldType,
    ImageType,
    ItemType,
    NumericType,
    ObjectType,
    ScalarType,
    TextType,
)


class Frame(MutableMapping[str, Field]):
    """Frame defines a container to store multiple fields containing the same number
    of elements. Frame implements all the interfaces for a mutable mapping of strings
    to fields, similar to ``dict[str, Field]``.

    .. note::
        Frame disables the ``__lt__``, ``__le__``, ``__eq__``, ``__ne__``,
        ``__gt__``, ``__ge__``, ``__hash__`` interfaces, as the Frame class is not
        frozen and comparable.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, Field]) -> None:
        """Initialize the frame with a mapping of names to fields.

        :param fields: The mapping of names to fields.
        :raises TypeError: The parameter ``fields`` must be a mapping of strings to
            fields.
        :raises ValueError: Some fields in ``fields`` contain different number of
            elements.
        """
        if not is_typed_mapping(fields, str, Field):
            raise TypeError(
                "The paramter `fields` is not a mapping of strings to fields."
            )

        field_len = len(next(iter(fields.values())))
        for field in fields.values():
            if len(field) != field_len:
                raise ValueError(
                    "Some fields in `fields` contain different number of elements."
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
            self.to_string()
            + f"\n[{self.num_elements} elements x {self.num_fields} fields]"
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
        :raises ValueError: The parameter ``value`` contains different number of
            elements.
        """
        if not isinstance(key, str):
            raise TypeError("The parameter `key` must be a string.")
        if not isinstance(value, Field):
            raise TypeError("The parameter `value` must be a field.")
        if self.num_fields > 0 and len(value) != self.num_elements:
            raise ValueError(
                "The parameter `value` contains different number of elements."
            )
        self._fields[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete the field with the given name.

        :param key: The name of the field.
        """
        del self._fields[key]

    # Frame
    @property
    def num_elements(self) -> int:
        """Get the number of elements in the frame."""
        if self.num_fields == 0:
            return 0
        return len(next(iter(self.values())))

    @property
    def loc_elements(self) -> "ElementLoc":
        """Select elements from the frame.

        ``.loc_elements[]`` is used to select elements of the frame. The allowed key
        types are listed at :py:meth:`ElementLoc.__getitem__`.
        """
        return self.ElementLoc(self)

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

    def describe(self, max_colwidth: int | None = 25) -> str:
        """Get the description of the frame.

        :return: The description of the frame.
        """
        fields_info = [{"name": name, **field._info()} for name, field in self.items()]
        desc_df = pd.DataFrame.from_records(fields_info)
        return desc_df.to_string(max_colwidth=max_colwidth)

    def validate(self) -> "Frame":
        """Validate the frame.

        :raises ValueError: Some fields contain different number of elements.
        :raises ValueError: Some fields have invalid values.
        :return: The validated frame itself.
        """
        for field in self.values():
            if len(field) != self.num_elements:
                raise ValueError("Some fields contain different number of elements.")
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
        :return: The sorted frame itself.
        """
        if name not in self:
            raise ValueError(f'Field "{name}" does not exist.')
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
        path = Path(path)
        logger.info(f"Saving frame to {path.name} ...")
        str_df = pd.DataFrame()
        for name, field in self.items():
            str_df[f"{name}:{repr(field.ftype)}"] = field._to_str_array()
        str_df.to_csv(path, sep=FIELD_SEP, index=False, header=True)

    def to_npz(self, path: str | PathLike) -> None:
        """Save the frame to a NPZ file.

        :param path: The path to the NPZ file.
        """
        path = Path(path)
        logger.info(f"Saving frame to {path.name} ...")
        array_dict: dict[str, np.ndarray] = {}
        for name, field in self.items():
            array_dict[f"{name}:{repr(field.ftype)}"] = field.value
        np.savez(path, **array_dict)

    @classmethod
    def from_npz(cls, path: str | PathLike) -> "Frame":
        """Load the frame from a NPZ file.

        :param path: The path to the NPZ file.
        :return: The loaded frame.
        """
        path = Path(path)
        logger.info(f"Loading frame from {path.name} ...")
        fields: dict[str, Field] = {}
        with np.load(path, allow_pickle=True) as npz:
            for name_and_ftype in npz:
                name, ftype = name_and_ftype.rsplit(":", 1)
                fields[name] = Field(eval(ftype), npz[name_and_ftype])
        return cls(fields)

    def to_string(
        self, max_rows: int | None = 10, max_colwidth: int | None = 25
    ) -> str:
        """Convert the frame to a informal string representation.

        :param max_rows: The maximum number of rows to show. If ``None``, all rows
            are shown.
        :param max_row_width: The maximum width of each row. If ``None``, all
            characters are shown.
        :return: The string representation.
        """
        if max_rows is None or max_rows >= self.num_elements:
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
        return str_df.to_string(max_colwidth=max_colwidth)

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

    class ElementLoc:
        """ElementLoc defines a helper class to select elements from the frame."""

        def __init__(self, frame: "Frame") -> None:
            """Initialize the helper object.

            :param frame: The frame to select elements from.
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
            """Select elements from the frame.

            :param key: The index of the elements.
            :raises TypeError: The parameter ``key`` is not a
                :external:py:class:`typing.SupportsIndex`, :external:py:class:`slice`,
                sequence of SupportsIndex or :external:py:class:`numpy.ndarray`.
            :return: The dict of single element or the frame of multiple elements.
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
