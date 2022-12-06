import copy
import pickle
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd

from ..constants import FIELD_SEP
from ..utils.asserts import (
    assert_never_type,
    assert_type,
    assert_typed_dict,
    assert_typed_list,
)
from ..utils.logger import logger
from .field import Field


@dataclass(eq=False, slots=True)
class Frame(MutableMapping[str, Field]):
    _fields: dict[str, Field]

    def __post_init__(self) -> None:
        if not assert_typed_dict(self._fields, str, Field):
            raise TypeError

        if len(self._fields) <= 1:
            return

        field_len = len(next(iter(self._fields.values())))
        if not all(len(field) == field_len for field in self._fields.values()):
            raise ValueError

    # implement object methods

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

    def __copy__(self) -> "Frame":
        return Frame(copy.copy(self._fields))

    def __deepcopy__(self, memo=None) -> "Frame":
        return Frame(copy.deepcopy(self._fields))

    # implement MutableMapping methods

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __getitem__(self, key: str) -> Field:
        return self._fields[key]

    def __setitem__(self, key: str, value: Field) -> None:
        assert_type(key, str)
        assert_type(value, Field)
        if len(self) > 0 and len(value) != len(next(iter(self.values()))):
            raise ValueError
        self._fields[key] = value

    def __delitem__(self, key: str) -> None:
        del self._fields[key]

    # Frame methods

    @property
    def num_elements(self) -> int:
        if self.num_fields == 0:
            return 0
        return len(next(iter(self.values())))

    @property
    def loc_elements(self) -> "ElementLoc":
        return self.ElementLoc(self)

    @property
    def num_fields(self) -> int:
        return len(self)

    @property
    def loc_fields(self) -> "FieldLoc":
        return self.FieldLoc(self)

    def describe(self, max_colwidth: int | None = 25) -> str:
        fields_info = [{"name": name, **field._info()} for name, field in self.items()]
        fields_dict = {
            key: [field_info[key] for field_info in fields_info]
            for key in fields_info[0].keys()
        }
        desc_df = pd.DataFrame(fields_dict, dtype=object)
        return desc_df.to_string(index=False, max_colwidth=max_colwidth)

    def validate(self) -> None:
        for _, field in self.items():
            if len(field) != self.num_elements:
                raise ValueError
            field.validate()

    def downcast(self) -> "Frame":
        return Frame({name: field.downcast() for name, field in self.items()})

    def to_csv(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving frame to {path.name} ...")
        str_df = pd.DataFrame(
            {name: field._to_str_array() for name, field in self.items()}
        )
        str_df.to_csv(path, sep=FIELD_SEP, index=False, header=True)

    def to_pickle(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving frame to {path.name} ...")
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path: str | PathLike) -> "Frame":
        path = Path(path)
        logger.info(f"Loading frame from {path.name} ...")
        with open(path, "rb") as file:
            return pickle.load(file)

    def to_string(
        self, max_rows: int | None = 10, max_colwidth: int | None = 25
    ) -> str:
        if max_rows is None or max_rows >= self.num_elements:
            str_df = pd.DataFrame()
            str_df["INDEX"] = np.arange(self.num_elements)
            for name, field in self.items():
                str_df[name] = field._to_str_array()
        else:
            head = (max_rows + 1) // 2
            tail = self.num_elements - (max_rows - head)
            ellipsis = np.array(["..."], dtype=object)
            str_df = pd.DataFrame()
            str_df["INDEX"] = np.concatenate(
                [
                    np.arange(head).astype(object),
                    ellipsis,
                    np.arange(tail, self.num_elements).astype(object),
                ]
            )
            for name, field in self.items():
                str_df[name] = np.concatenate(
                    [
                        field.loc[:head]._to_str_array(),
                        ellipsis,
                        field.loc[tail:]._to_str_array(),
                    ]
                )
        return str_df.to_string(index=False, max_colwidth=max_colwidth)

    class ElementLoc:
        def __init__(self, frame: "Frame") -> None:
            self.frame = frame

        @overload
        def __getitem__(self, index: slice) -> "Frame":
            ...

        @overload
        def __getitem__(self, index: list[int]) -> "Frame":
            ...

        @overload
        def __getitem__(self, index: list[bool]) -> "Frame":
            ...

        @overload
        def __getitem__(self, index: np.ndarray) -> "Frame":
            ...

        @overload
        def __getitem__(self, index: int) -> dict[str, Any]:
            ...

        def __getitem__(self, index):
            if isinstance(index, int):
                return {name: field[index] for name, field in self.frame.items()}
            if (
                isinstance(index, slice)
                or assert_typed_list(index, int)
                or assert_typed_list(index, bool)
                or isinstance(index, np.ndarray)
            ):
                return Frame(
                    {name: field.loc[index] for name, field in self.frame.items()}
                )
            assert_never_type(index)

    class FieldLoc:
        def __init__(self, frame: "Frame") -> None:
            self.frame = frame

        @overload
        def __getitem__(self, index: str) -> Field:
            ...

        @overload
        def __getitem__(self, index: list[str]) -> "Frame":
            ...

        def __getitem__(self, index):
            if isinstance(index, str):
                return self.frame[index]
            if assert_typed_list(index, str):
                return Frame({k: self.frame[k] for k in index})
            assert_never_type(index)
