import copy
import json
import pickle
from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum, unique
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
from ..utils.file import checksum, compress
from ..utils.logger import logger
from .predefined_fields import (
    IID,
    LABEL,
    POP_PROB,
    TEST_IIDS_SET,
    TEST_MASK,
    TEST_NEG_IIDS,
    TIMESTAMP,
    TRAIN_IIDS_SET,
    TRAIN_MASK,
    UID,
    VAL_IIDS_SET,
    VAL_MASK,
    VAL_NEG_IIDS,
)


@dataclass(frozen=True, slots=True)
class ElementType:
    def __str__(self) -> str:
        raise NotImplementedError

    def _can_be_nested(self) -> bool:
        raise NotImplementedError

    def _can_hold_null(self) -> bool:
        raise NotImplementedError

    def _non_null_count(self, array: np.ndarray) -> int:
        raise NotImplementedError

    def _mean(self, array: np.ndarray) -> float | None:
        raise NotImplementedError

    def _min(self, array: np.ndarray) -> bool | int | float | str | None:
        raise NotImplementedError

    def _max(self, array: np.ndarray) -> bool | int | float | str | None:
        raise NotImplementedError

    def _check_value_data(self, array: np.ndarray, dtype: str) -> None:
        raise NotImplementedError

    def _to_str_array(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FieldType:
    element_type: ElementType

    def __post_init__(self) -> None:
        assert_type(self.element_type, ElementType)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def _dtype(self, value: np.ndarray) -> str:
        raise NotImplementedError

    def _shape(self, value: np.ndarray) -> tuple:
        raise NotImplementedError

    def _non_null_count(self, value: np.ndarray) -> int:
        raise NotImplementedError

    def _mean(self, value: np.ndarray) -> float | None:
        raise NotImplementedError

    def _min(self, value: np.ndarray) -> bool | int | float | str | None:
        raise NotImplementedError

    def _max(self, value: np.ndarray) -> bool | int | float | str | None:
        raise NotImplementedError

    def _check_value_shape(self, value: np.ndarray) -> None:
        raise NotImplementedError

    def _check_value_dtype(self, value: np.ndarray) -> None:
        raise NotImplementedError

    def _validate(self, value: np.ndarray) -> None:
        self._check_value_shape(value)
        self._check_value_dtype(value)

    def _downcast(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _to_str_array(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError


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
class Partition(Enum):
    """Partition of the data after splitting."""

    TRAINING = "training"
    """Training partition."""

    VALIDATION = "validation"
    """Validation partition."""

    TEST = "test"
    """Test partition."""

    def __repr__(self) -> str:
        return str(self)

    @property
    def mask_field(self) -> str:
        """Mask field name."""
        match self:
            case Partition.TRAINING:
                return TRAIN_MASK
            case Partition.VALIDATION:
                return VAL_MASK
            case Partition.TEST:
                return TEST_MASK

    @property
    def iids_set_field(self) -> str:
        """IIDs set field name."""
        match self:
            case Partition.TRAINING:
                return TRAIN_IIDS_SET
            case Partition.VALIDATION:
                return VAL_IIDS_SET
            case Partition.TEST:
                return TEST_IIDS_SET

    @property
    def neg_iids_field(self) -> str:
        """Negative IIDs field name."""
        match self:
            case Partition.TRAINING:
                raise ValueError
            case Partition.VALIDATION:
                return VAL_NEG_IIDS
            case Partition.TEST:
                return TEST_NEG_IIDS


_FRAMES_PICKLE = {
    Source.INTERACTION: "interaction_frame.pickle",
    Source.USER: "user_frame.pickle",
    Source.ITEM: "item_frame.pickle",
}
_FRAMES_CSV = {
    Source.INTERACTION: "interaction_frame.csv",
    Source.USER: "user_frame.csv",
    Source.ITEM: "item_frame.csv",
}


class Data:
    __slots__ = ("frames",)

    def __init__(
        self, interaction_frame: Frame, user_frame: Frame, item_frame: Frame
    ) -> None:
        """Initialize the data.

        :param interaction_frame: The interaction frame.
        :param user_frame: The user frame.
        :param item_frame: The item frame.
        """
        assert_type(interaction_frame, Frame)
        assert_type(user_frame, Frame)
        assert_type(item_frame, Frame)

        self.frames: dict[Source, Frame] = {
            Source.INTERACTION: interaction_frame,
            Source.USER: user_frame,
            Source.ITEM: item_frame,
        }
        """The underlying frames in the data."""

    @property
    def num_elements(self) -> dict[Source, int]:
        """The number of elements for each source."""
        return {source: frame.num_elements for source, frame in self.frames.items()}

    @property
    def has_timestamp(self) -> bool:
        """Whether the data has timestamp."""
        return TIMESTAMP in self.frames[Source.INTERACTION]

    @property
    def is_splitted(self) -> bool:
        """Whether the data is splitted."""
        return (
            {TRAIN_MASK, VAL_MASK, TEST_MASK}.issubset(self.frames[Source.INTERACTION])
            and {TRAIN_IIDS_SET, VAL_IIDS_SET, TEST_IIDS_SET}.issubset(
                self.frames[Source.USER]
            )
            and POP_PROB in self.frames[Source.ITEM]
        )

    @property
    def num_partition_interactions(self) -> dict[Partition, int]:
        """The number of interactions in each partition."""
        interaction_frame = self.frames[Source.INTERACTION]
        return {
            partition: interaction_frame[partition.mask_field].value.sum()
            for partition in Partition
        }

    @property
    def has_eval_negative_samples(self) -> bool:
        """Whether the data has evaluation negative samples."""
        return {VAL_NEG_IIDS, TEST_NEG_IIDS}.issubset(self.frames[Source.USER])

    @property
    def num_eval_negative_samples(self) -> dict[Partition, int]:
        """The number of evaluation negative samples in each partition."""
        user_frame = self.frames[Source.USER]
        return {
            partition: user_frame[partition.neg_iids_field].value.shape[1]
            for partition in Partition
            if partition != Partition.TRAINING
        }

    @property
    def fields(self) -> dict[Source, list[str]]:
        """The fields for each source."""
        return {source: list(frame.keys()) for source, frame in self.frames.items()}

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
        if self.is_splitted:
            fields[Source.INTERACTION].extend([TRAIN_MASK, VAL_MASK, TEST_MASK])
            fields[Source.USER].extend([TRAIN_IIDS_SET, VAL_IIDS_SET, TEST_IIDS_SET])
            fields[Source.ITEM].append(POP_PROB)
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
        for source, frame in self.frames.items():
            for field in frame:
                if field not in base_fields[source]:
                    fields[source].append(field)
        return fields

    def validate(self) -> None:  # noqa: C901
        """Validate the data.

        :raises ValueError: If the data is invalid.
        :return: The validated data itself.
        """
        # validate the base fields
        for source, fields in self.base_fields.items():
            for field in fields:
                if field not in self.frames[source]:
                    raise ValueError(f"Mising field {field} in {source} frame.")

        # validate the context fields
        context_fields: set[str] = set()
        for source, fields in self.context_fields.items():
            for field in fields:
                if field in context_fields:
                    raise ValueError(f"Duplicate context field {field} in data.")
                context_fields.add(field)

        # validate the uids and iids
        interaction_uids_set = set(self.frames[Source.INTERACTION][UID].value)
        interaction_iids_set = set(self.frames[Source.INTERACTION][IID].value)
        uids_set = set(self.frames[Source.USER][UID].value)
        iids_set = set(self.frames[Source.ITEM][IID].value)
        if interaction_uids_set > uids_set:
            raise ValueError(
                "The user IDs in the interaction frame are not a subset of the user IDs"
                "in the user frame."
            )
        if interaction_iids_set > iids_set:
            raise ValueError(
                "The item IDs in the interaction frame are not a subset of the item IDs"
                "in the item frame."
            )

        # validate the frames
        for _, frame in self.frames.items():
            frame.validate()

    def downcast(self) -> "Data":
        return Data(
            interaction_frame=self.frames[Source.INTERACTION].downcast(),
            user_frame=self.frames[Source.USER].downcast(),
            item_frame=self.frames[Source.ITEM].downcast(),
        )

    def __str__(self) -> str:
        return "\n".join(
            frame.to_string()
            + f"\n[{frame.num_elements} {source.value}s x {frame.num_fields} fields]"
            for source, frame in self.frames.items()
        )

    def info(self) -> None:
        """Print the information of the data."""
        print(self.__class__)

        print(f"# interactions: {self.num_elements[Source.INTERACTION]}")
        print(f"with timestamp: {self.has_timestamp}")
        print(f"splitted: {self.is_splitted}")
        if self.is_splitted:
            for partition in Partition:
                num = self.num_partition_interactions[partition]
                print(f"  # {partition.value} interactions: {num}")
        print(f"Base fields of interactions: {self.base_fields[Source.INTERACTION]}")
        print(
            f"Context fields of interactions: {self.context_fields[Source.INTERACTION]}"
        )
        print(self.frames[Source.INTERACTION].describe())

        print(f"# users: {self.num_elements[Source.USER]}")
        print(f"with evaluation negative samples: {self.has_eval_negative_samples}")
        if self.has_eval_negative_samples:
            num_val = self.num_eval_negative_samples[Partition.VALIDATION]
            print(f"  # validation negative samples: {num_val}")
            num_test = self.num_eval_negative_samples[Partition.TEST]
            print(f"  # test negative samples: {num_test}")
        print(f"Base fields of users: {self.base_fields[Source.USER]}")
        print(f"Context fields of users: {self.context_fields[Source.USER]}")
        print(self.frames[Source.USER].describe())

        print(f"# Items: {self.num_elements[Source.ITEM]}")
        print(f"Base fields of items: {self.base_fields[Source.ITEM]}")
        print(f"Context fields of items: {self.context_fields[Source.ITEM]}")
        print(self.frames[Source.ITEM].describe())

    def to_pickle(self, path: str | PathLike) -> None:
        """Save the data to pickle files.

        :param path: The directory to save the data.
        """
        path = Path(path)
        logger.info(f"Saving data to {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self.frames.items():
            frame.to_pickle(path / _FRAMES_PICKLE[source])

    @classmethod
    def from_pickle(cls, path: str | PathLike) -> "Data":
        """Load the data from pickle files.

        :param path: The directory to load the data.
        :return: The loaded data.
        """
        path = Path(path)
        logger.info(f"Loading data from {path} ...")
        interaction_frame = Frame.from_pickle(path / _FRAMES_PICKLE[Source.INTERACTION])
        user_frame = Frame.from_pickle(path / _FRAMES_PICKLE[Source.USER])
        item_frame = Frame.from_pickle(path / _FRAMES_PICKLE[Source.ITEM])
        return cls(interaction_frame, user_frame, item_frame)

    def to_csv(self, path: str | PathLike) -> None:
        """Save the data to csv files.

        :param path: The directory to save the data.
        """
        path = Path(path)
        logger.info(f"Saving data to {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self.frames.items():
            frame.to_csv(path / _FRAMES_CSV[source])


_FRAMES_PICKLE_XZ = {
    Source.INTERACTION: "interaction_frame.pickle.xz",
    Source.USER: "user_frame.pickle.xz",
    Source.ITEM: "item_frame.pickle.xz",
}
_DATA_INFO_JSON = "data_info.json"


@dataclass(frozen=True, kw_only=True, slots=True)
class DataInfo:
    """Data information."""

    description: str
    """Description of the data."""
    citation: str
    """Citation of the data."""
    homepage: str
    """Homepage of the data."""
    num_elements: dict[str, int]
    """Number of elements of each source."""
    base_fields: dict[str, dict[str, dict[str, Any]]]
    """Base field information of each source."""
    context_fields: dict[str, dict[str, dict[str, Any]]]
    """Context field information of each source."""
    files_size: dict[str, int]
    """Size of the data file for each source."""
    files_checksum: dict[str, str]
    """Checksum of the data file for each source."""
    compressed_files_size: dict[str, int]
    """Size of the compressed data file for each source."""
    compressed_files_checksum: dict[str, str]
    """SHA256 Checksum of the compressed data file for each source."""

    @property
    def fields(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Field information for each source."""
        fields = {}
        for source in Source:
            fields[source.value] = {
                **self.base_fields[source.value],
                **self.context_fields[source.value],
            }
        return fields

    def print_field_info(self) -> None:
        """Print the field information."""
        fields = self.fields
        for source, field_info in fields.items():
            print(f"{source} fields:")
            infos = [{"name": name, **info} for name, info in field_info.items()]
            info_frame = pd.DataFrame.from_records(infos)
            print(info_frame)

    @classmethod
    def from_data_files(
        cls,
        path: str | PathLike,
        description: str = "",
        citation: str = "",
        homepage: str = "",
    ) -> "DataInfo":
        """Create the data information from the data files.

        :param path: The directory to load the data.
        :param description: The description of the data.
        :param citation: The citation of the data.
        :param homepage: The homepage of the data.
        :return: The data information.
        """
        path = Path(path)
        logger.info(f"Creating data information from {path} ...")
        data = Data.from_pickle(path)

        num_elements: dict[str, int] = {
            source.value: num for source, num in data.num_elements.items()
        }

        base_fields: dict[str, dict[str, dict[str, Any]]] = {}
        for source, fields in data.base_fields.items():
            frame = data.frames[source]
            base_fields[source.value] = {name: frame[name]._info() for name in fields}

        context_fields: dict[str, dict[str, dict[str, Any]]] = {}
        for source, fields in data.context_fields.items():
            frame = data.frames[source]
            context_fields[source.value] = {
                name: frame[name]._info() for name in fields
            }

        files_size: dict[str, int] = {}
        files_checksum: dict[str, str] = {}
        for source, filename in _FRAMES_PICKLE.items():
            file_path = path / filename
            files_size[source.value] = file_path.stat().st_size
            files_checksum[source.value] = checksum(file_path)

        compressed_files_size: dict[str, int] = {}
        compressed_files_checksum: dict[str, str] = {}
        for source, filename in _FRAMES_PICKLE_XZ.items():
            compressed_file_path = path / filename
            file_path = path / _FRAMES_PICKLE[source]
            compress(file_path, compressed_file_path)
            compressed_files_size[source.value] = compressed_file_path.stat().st_size
            compressed_files_checksum[source.value] = checksum(compressed_file_path)

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            num_elements=num_elements,
            base_fields=base_fields,
            context_fields=context_fields,
            files_size=files_size,
            files_checksum=files_checksum,
            compressed_files_size=compressed_files_size,
            compressed_files_checksum=compressed_files_checksum,
        )

    def to_json(self, path: str | PathLike, pretty_print=False):
        """Write the data info to a json file.

        :param path: The directory to save the data info.
        :param pretty_print: Whether to pretty print the json file.
        """
        path = Path(path)
        logger.info(f"Saving data info to {path} ...")
        with open(path / _DATA_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4 if pretty_print else None)

    @classmethod
    def from_json(cls, path: str | PathLike) -> "DataInfo":
        """Create a data info from a json file.

        :param path: The directory to load the data info.
        :return: The data info.
        """
        path = Path(path)
        logger.info(f"Loading data info from {path} ...")
        with open(path / _DATA_INFO_JSON, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
