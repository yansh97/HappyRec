import copy
import pickle
from collections.abc import ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
from dataclasses import dataclass
from enum import Enum, unique
from os import PathLike
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from ..constants import (
    FIELD_SEP,
    IID,
    LABEL,
    TEST_MASK,
    TEST_NEG_IIDS,
    TIMESTAMP,
    TRAIN_MASK,
    UID,
    VAL_MASK,
    VAL_NEG_IIDS,
)
from ..utils.asserts import assert_never_type, assert_type, is_typed_dict, is_typed_list
from ..utils.logger import logger
from .ftypes import _BASE_FIELD_CHECKER, FieldType, _convert_pa_type_to_ftype, category


@dataclass(eq=False, frozen=True, slots=True)
class Field(Sequence):
    ftype: FieldType
    value: np.ndarray

    def __post_init__(self) -> None:
        assert_type(self.ftype, FieldType)
        assert_type(self.value, np.ndarray)
        self.ftype._check_value_shape(self.value)

    # implement object methods

    def __str__(self) -> str:
        return f"{self.ftype}{list(self.shape())}"

    def __copy__(self) -> "Field":
        return Field(copy.copy(self.ftype), copy.copy(self.value))

    def __deepcopy__(self, memo=None) -> "Field":
        return Field(
            copy.deepcopy(self.ftype, memo=memo), copy.deepcopy(self.value, memo=memo)
        )

    # implement Sequence methods

    def __getitem__(self, index: int) -> Any:
        return self.value[index]

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, value) -> bool:
        for v in self:
            if v is value or v == value:
                return True
        return False

    def __iter__(self) -> Iterator[Any]:
        return iter(self.value)

    def __reversed__(self) -> Iterator[Any]:
        return reversed(self.value)

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

    # Field methods

    def can_be_sorted(self) -> bool:
        return self.ftype._can_be_sorted()

    class Loc:
        def __init__(self, field: "Field") -> None:
            self.field = field

        def __getitem__(
            self, index: slice | list[int] | list[bool] | np.ndarray
        ) -> "Field":
            if isinstance(index, slice):
                return Field(self.field.ftype, self.field.value[index].copy())
            if (
                is_typed_list(index, int)
                or is_typed_list(index, bool)
                or isinstance(index, np.ndarray)
            ):
                return Field(self.field.ftype, self.field.value[index])
            assert_never_type(index)

    @property
    def loc(self) -> Loc:
        return self.Loc(self)

    def dtype(self) -> str:
        return self.ftype._dtype(self.value)

    def shape(self) -> tuple:
        return self.ftype._shape(self.value)

    def non_null_count(self) -> int:
        return self.ftype._non_null_count(self.value)

    def mean(self) -> Any:
        return self.ftype._mean(self.value)

    def min(self) -> Any:
        return self.ftype._min(self.value)

    def max(self) -> Any:
        return self.ftype._max(self.value)

    def validate(self) -> None:
        self.ftype._validate(self.value)

    def downcast(self) -> "Field":
        return Field(self.ftype, self.ftype._downcast(self.value))

    def to_str_array(self) -> np.ndarray:
        return self.ftype._to_str_array(self.value)

    def to_pa_array(self) -> pa.Array:
        return self.ftype._to_pa_array(self.value)

    @classmethod
    def from_pa_array(cls, pa_array: pa.Array) -> "Field":
        ftype = _convert_pa_type_to_ftype(pa_array.type)
        value = ftype._from_pa_array(pa_array)
        return cls(ftype, value)

    def get_info_dict(self) -> dict[str, Any]:
        ret = {
            "field type": str(self.ftype),
            "dtype": self.dtype(),
            "shape": self.shape(),
            "non-null count": self.non_null_count(),
            "mean": self.mean(),
            "min": self.min(),
            "max": self.max(),
        }
        return ret


@dataclass(eq=False, frozen=True, slots=True)
class Frame(Mapping[str, Field]):
    fields: dict[str, Field]

    def __post_init__(self) -> None:
        if not is_typed_dict(self.fields, str, Field):
            raise TypeError

        if len(self.fields) <= 1:
            return

        field_len = len(next(iter(self.fields.values())))
        if not all(len(field) == field_len for field in self.fields.values()):
            raise ValueError

    # implement object methods

    def __str__(self) -> str:
        return (
            self.to_string()
            + f"\n[{self.num_elements} elements x {self.num_fields} fields]"
        )

    def __copy__(self) -> "Frame":
        return Frame(copy.copy(self.fields))

    def __deepcopy__(self, memo=None) -> "Frame":
        return Frame(copy.deepcopy(self.fields, memo=memo))

    # implement Mapping methods

    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def keys(self) -> KeysView[str]:
        return self.fields.keys()

    def items(self) -> ItemsView[str, Field]:
        return self.fields.items()

    def values(self) -> ValuesView[Field]:
        return self.fields.values()

    def get(self, key: str, default: Field | None = None) -> Field | None:
        if default is not None:
            assert_type(default, Field)
        return self.fields.get(key, default)

    # Frame methods

    @property
    def num_elements(self) -> int:
        if self.num_fields == 0:
            return 0
        return len(next(iter(self.values())))

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
                or is_typed_list(index, int)
                or is_typed_list(index, bool)
                or isinstance(index, np.ndarray)
            ):
                return Frame(
                    {name: field.loc[index] for name, field in self.frame.items()}
                )
            assert_never_type(index)

    @property
    def loc_elements(self) -> "ElementLoc":
        return self.ElementLoc(self)

    @property
    def num_fields(self) -> int:
        return len(self)

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
            if is_typed_list(index, str):
                return Frame({k: self.frame[k] for k in index})
            assert_never_type(index)

    @property
    def loc_fields(self) -> "FieldLoc":
        return self.FieldLoc(self)

    def describe(self, max_colwidth: int | None = 25) -> str:
        fields_info = [
            {"name": name, **field.get_info_dict()} for name, field in self.items()
        ]
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

    def sort_values(
        self,
        by: str,
        *,
        ascending: bool = True,
        kind: Literal["quicksort", "stable"] = "stable",
    ) -> "Frame":
        if not self[by].can_be_sorted():
            raise ValueError
        indices = np.argsort(self[by].value, kind=kind)
        if not ascending:
            indices = indices[::-1]
        return self.loc_elements[indices]

    def groupby(self, by: str) -> Iterator[tuple[Any, "Frame"]]:
        if not self[by].can_be_sorted():
            raise ValueError
        unique_values, codes = np.unique(self[by].value, return_inverse=True)
        for i in range(len(unique_values)):
            yield unique_values[i], self.loc_elements[codes == i]

    def to_csv(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving frame to {path.name} ...")
        str_df = pd.DataFrame(
            {name: field.to_str_array() for name, field in self.items()}
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

    def to_feather(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving frame to {path.name} ...")
        pa_array_dict: dict[str, pa.Array] = {
            name: field.to_pa_array() for name, field in self.items()
        }
        pa_table: pa.Table = pa.table(pa_array_dict)
        feather.write_feather(pa_table, path)

    @classmethod
    def from_feather(cls, path: str | PathLike) -> "Frame":
        path = Path(path)
        logger.info(f"Loading frame from {path.name} ...")
        pa_table = feather.read_table(path)
        fields: dict[str, Field] = {
            name: Field.from_pa_array(pa_table.column(name))
            for name in pa_table.column_names
        }
        return Frame(fields)

    def to_string(
        self, max_rows: int | None = 10, max_colwidth: int | None = 25
    ) -> str:
        if max_rows is None or max_rows >= self.num_elements:
            str_df = pd.DataFrame()
            str_df["INDEX"] = np.arange(self.num_elements)
            for name, field in self.items():
                str_df[name] = field.to_str_array()
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
                        field.loc[:head].to_str_array(),
                        ellipsis,
                        field.loc[tail:].to_str_array(),
                    ]
                )
        return str_df.to_string(index=False, max_colwidth=max_colwidth)


@unique
class Source(Enum):
    """Field source."""

    INTERACTION = "interaction"
    """Fields associated with the interactions."""

    USER = "user"
    """Fields associated with the users."""

    ITEM = "item"
    """Fields associated with the items."""

    def __repr__(self) -> str:
        return str(self)


@unique
class Phase(Enum):
    """Phase."""

    TRAINING = "training"
    """Training phase."""

    VALIDATION = "validation"
    """Validation phase."""

    TEST = "test"
    """Test phase."""

    def __repr__(self) -> str:
        return str(self)

    @property
    def mask_field(self) -> str:
        """Mask field name."""
        match self:
            case Phase.TRAINING:
                return TRAIN_MASK
            case Phase.VALIDATION:
                return VAL_MASK
            case Phase.TEST:
                return TEST_MASK

    @property
    def neg_iids_field(self) -> str:
        """Negative IIDs field name."""
        match self:
            case Phase.TRAINING:
                raise ValueError
            case Phase.VALIDATION:
                return VAL_NEG_IIDS
            case Phase.TEST:
                return TEST_NEG_IIDS


@dataclass(eq=False, frozen=True, slots=True)
class Data(Mapping[str, Frame]):
    frames: dict[Source, Frame]

    def __post_init__(self) -> None:
        if not is_typed_dict(self.frames, Source, Frame):
            raise ValueError

        if len(self.frames) != 3:
            raise ValueError

        if list(self.frames.keys()) != list(Source):
            raise ValueError

    @classmethod
    def from_frames(
        cls, interaction_frame: Frame, user_frame: Frame, item_frame: Frame
    ) -> "Data":
        return cls(
            {
                Source.INTERACTION: interaction_frame,
                Source.USER: user_frame,
                Source.ITEM: item_frame,
            }
        )

    # implement object methods

    def __str__(self) -> str:
        return "\n".join(
            f"{source.value} frame:\n" + str(frame)
            for source, frame in self.frames.items()
        )

    def __copy__(self) -> "Data":
        return Data(copy.copy(self.frames))

    def __deepcopy__(self, memo=None) -> "Data":
        return Data(copy.deepcopy(self.frames, memo=memo))

    # implement Mapping methods

    def __getitem__(self, key: Source) -> Frame:
        return self.frames[key]

    def __iter__(self) -> Iterator[Source]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __contains__(self, key: Source) -> bool:
        return key in self.frames

    def keys(self) -> KeysView[Source]:
        return self.frames.keys()

    def items(self) -> ItemsView[Source, Frame]:
        return self.frames.items()

    def values(self) -> ValuesView[Frame]:
        return self.frames.values()

    def get(self, key: Source, default: Frame | None = None) -> Frame | None:
        if default is not None:
            assert_type(default, Frame)
        return self.frames.get(key, default)

    # Data methods

    @property
    def num_elements(self) -> dict[Source, int]:
        """The number of elements for each source."""
        return {source: frame.num_elements for source, frame in self.items()}

    @property
    def has_timestamp(self) -> bool:
        """Whether the data has timestamp."""
        return TIMESTAMP in self[Source.INTERACTION]

    @property
    def has_phase_mask(self) -> bool:
        """Whether the data has phase mask."""
        return {TRAIN_MASK, VAL_MASK, TEST_MASK}.issubset(self[Source.INTERACTION])

    @property
    def num_phase_interactions(self) -> dict[Phase, int]:
        """The number of interactions in each phase."""
        interaction_frame = self[Source.INTERACTION]
        return {
            phase: interaction_frame[phase.mask_field].value.sum() for phase in Phase
        }

    @property
    def has_eval_negative_samples(self) -> bool:
        """Whether the data has evaluation negative samples."""
        return {VAL_NEG_IIDS, TEST_NEG_IIDS}.issubset(self[Source.USER])

    @property
    def num_eval_negative_samples(self) -> dict[Phase, int]:
        """The number of evaluation negative samples in each phase."""
        user_frame = self[Source.USER]
        return {
            phase: user_frame[phase.neg_iids_field].value.shape[1]
            for phase in Phase
            if phase != Phase.TRAINING
        }

    @property
    def fields(self) -> dict[Source, list[str]]:
        """The fields for each source."""
        return {source: list(frame.keys()) for source, frame in self.items()}

    @property
    def base_fields(self) -> dict[Source, list[str]]:
        """The base fields for each source."""
        fields = {
            Source.INTERACTION: [UID, IID, LABEL],
            Source.USER: [UID],
            Source.ITEM: [IID],
        }
        if self.has_timestamp:
            fields[Source.INTERACTION].append(TIMESTAMP)
        if self.has_phase_mask:
            fields[Source.INTERACTION].extend([TRAIN_MASK, VAL_MASK, TEST_MASK])
        if self.has_eval_negative_samples:
            fields[Source.USER].extend([VAL_NEG_IIDS, TEST_NEG_IIDS])
        return fields

    @property
    def context_fields(self) -> dict[Source, list[str]]:
        """The context fields for each source."""
        base_fields = self.base_fields
        fields = {
            Source.INTERACTION: [],
            Source.USER: [],
            Source.ITEM: [],
        }
        for source, frame in self.items():
            for field in frame:
                if field not in base_fields[source]:
                    fields[source].append(field)
        return fields

    def validate(self) -> None:  # noqa: C901
        field_names: set[str] = set()
        # validate the base fields
        for source, fields in self.base_fields.items():
            for field in fields:
                if field not in self[source]:
                    raise ValueError
                if not _BASE_FIELD_CHECKER[field](self[source][field].ftype):
                    raise ValueError
                field_names.add(field)

        # validate the context fields
        for source, fields in self.context_fields.items():
            for field in fields:
                if field in field_names:
                    raise ValueError
                field_names.add(field)

        # validate the uids and iids
        if not _has_continuous_ids(self[Source.USER][UID]):
            raise ValueError
        if not _has_continuous_ids(self[Source.ITEM][IID]):
            raise ValueError
        num_users = self[Source.USER].num_elements
        num_items = self[Source.ITEM].num_elements
        if self[Source.INTERACTION][UID].max() >= num_users:
            raise ValueError
        if self[Source.INTERACTION][IID].max() >= num_items:
            raise ValueError
        if self.has_eval_negative_samples:
            if self[Source.USER][VAL_NEG_IIDS].max() >= num_items:
                raise ValueError
            if self[Source.USER][TEST_NEG_IIDS].max() >= num_items:
                raise ValueError

        # validate the frames
        for _, frame in self.items():
            frame.validate()

    def downcast(self) -> "Data":
        return Data({source: frame.downcast() for source, frame in self.items()})

    def info(self) -> None:
        print(self.__class__)

        print(f"# interactions: {self.num_elements[Source.INTERACTION]}")
        print(f"with timestamp: {self.has_timestamp}")
        print(f"with phase mask: {self.has_phase_mask}")
        if self.has_phase_mask:
            for phase in Phase:
                num = self.num_phase_interactions[phase]
                print(f"  # {phase.value} interactions: {num}")
        print(f"Base fields of interactions: {self.base_fields[Source.INTERACTION]}")
        print(
            f"Context fields of interactions: {self.context_fields[Source.INTERACTION]}"
        )
        print(self[Source.INTERACTION].describe())

        print(f"# users: {self.num_elements[Source.USER]}")
        print(f"with evaluation negative samples: {self.has_eval_negative_samples}")
        if self.has_eval_negative_samples:
            num_val = self.num_eval_negative_samples[Phase.VALIDATION]
            print(f"  # validation negative samples: {num_val}")
            num_test = self.num_eval_negative_samples[Phase.TEST]
            print(f"  # test negative samples: {num_test}")
        print(f"Base fields of users: {self.base_fields[Source.USER]}")
        print(f"Context fields of users: {self.context_fields[Source.USER]}")
        print(self[Source.USER].describe())

        print(f"# Items: {self.num_elements[Source.ITEM]}")
        print(f"Base fields of items: {self.base_fields[Source.ITEM]}")
        print(f"Context fields of items: {self.context_fields[Source.ITEM]}")
        print(self[Source.ITEM].describe())

    def to_csv(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving data to {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self.items():
            frame.to_csv(path / _FRAMES_CSV[source])

    def to_pickle(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving data to {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self.items():
            frame.to_pickle(path / _FRAMES_PICKLE[source])

    @classmethod
    def from_pickle(cls, path: str | PathLike) -> "Data":
        path = Path(path)
        logger.info(f"Loading data from {path} ...")
        frames: dict[Source, Frame] = {
            source: Frame.from_pickle(path / _FRAMES_PICKLE[source])
            for source in Source
        }
        return cls(frames)

    def to_feather(self, path: str | PathLike) -> None:
        path = Path(path)
        logger.info(f"Saving data to {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self.items():
            frame.to_feather(path / _FRAMES_FEATHER[source])

    @classmethod
    def from_feather(cls, path: str | PathLike) -> "Data":
        path = Path(path)
        logger.info(f"Loading data from {path} ...")
        frames: dict[Source, Frame] = {
            source: Frame.from_feather(path / _FRAMES_FEATHER[source])
            for source in Source
        }
        return cls(frames)


_FRAMES_CSV = {
    Source.INTERACTION: "interaction_frame.csv",
    Source.USER: "user_frame.csv",
    Source.ITEM: "item_frame.csv",
}
_FRAMES_PICKLE = {
    Source.INTERACTION: "interaction_frame.pickle",
    Source.USER: "user_frame.pickle",
    Source.ITEM: "item_frame.pickle",
}
_FRAMES_FEATHER = {
    Source.INTERACTION: "interaction_frame.feather",
    Source.USER: "user_frame.feather",
    Source.ITEM: "item_frame.feather",
}


def _has_continuous_ids(field: Field) -> bool:
    if field.ftype != category():
        raise ValueError
    return (field.value == np.arange(len(field))).all()
