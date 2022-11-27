import json
from dataclasses import asdict, dataclass
from enum import Enum, unique
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.file import checksum, compress
from ..utils.logger import logger
from .field import (  # noqa: F401
    CategoricalType,
    FieldType,
    ImageType,
    ItemType,
    NumericType,
    ObjectType,
    ScalarType,
    TextType,
)
from .frame import Frame
from .predefined_fields import (
    FTYPES,
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
                raise ValueError("Training partition does not have negative IIDs")
            case Partition.VALIDATION:
                return VAL_NEG_IIDS
            case Partition.TEST:
                return TEST_NEG_IIDS


_FRAMES_NPZ = {
    Source.INTERACTION: "interaction_frame.npz",
    Source.USER: "user_frame.npz",
    Source.ITEM: "item_frame.npz",
}
_FRAMES_CSV = {
    Source.INTERACTION: "interaction_frame.csv",
    Source.USER: "user_frame.csv",
    Source.ITEM: "item_frame.csv",
}


class Data:
    """Data is used to store the interaction, user, and item frame."""

    __slots__ = ("frames",)

    def __init__(
        self, interaction_frame: Frame, user_frame: Frame, item_frame: Frame
    ) -> None:
        """Initialize the data.

        :param interaction_frame: The interaction frame.
        :param user_frame: The user frame.
        :param item_frame: The item frame.
        """
        super().__init__()

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

    def validate(self) -> "Data":
        """Validate the data.

        :raises ValueError: If the data is invalid.
        :return: The validated data itself.
        """
        # validate the types of the base fields
        for source, fields in self.base_fields.items():
            for field in fields:
                if field not in self.frames[source]:
                    raise ValueError(f"Mising field {field} in {source} frame.")
                if self.frames[source][field].ftype != FTYPES[field]:
                    raise ValueError(
                        f"Invalid field type of field {field} in {source} frame."
                    )

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
        for frame in self.frames.values():
            frame.validate()

        return self

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

    def to_npz(self, path: str | PathLike) -> None:
        """Save the data to npz files.

        :param path: The directory to save the data.
        """
        path = Path(path)
        logger.info(f"Saving data to {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self.frames.items():
            frame.to_npz(path / _FRAMES_NPZ[source])

    @classmethod
    def from_npz(cls, path: str | PathLike) -> "Data":
        """Load the data from npz files.

        :param path: The directory to load the data.
        :return: The loaded data.
        """
        path = Path(path)
        logger.info(f"Loading data from {path} ...")
        interaction_frame = Frame.from_npz(path / _FRAMES_NPZ[Source.INTERACTION])
        user_frame = Frame.from_npz(path / _FRAMES_NPZ[Source.USER])
        item_frame = Frame.from_npz(path / _FRAMES_NPZ[Source.ITEM])
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


_FRAMES_NPZ_XZ = {
    Source.INTERACTION: "interaction_frame.npz.xz",
    Source.USER: "user_frame.npz.xz",
    Source.ITEM: "item_frame.npz.xz",
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
        data = Data.from_npz(path)

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
        for source, filename in _FRAMES_NPZ.items():
            file_path = path / filename
            files_size[source.value] = file_path.stat().st_size
            files_checksum[source.value] = checksum(file_path)

        compressed_files_size: dict[str, int] = {}
        compressed_files_checksum: dict[str, str] = {}
        for source, filename in _FRAMES_NPZ_XZ.items():
            compressed_file_path = path / filename
            if not compressed_file_path.exists():
                file_path = path / _FRAMES_NPZ[source]
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
