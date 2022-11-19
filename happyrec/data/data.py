import json
from dataclasses import dataclass
from enum import Enum, unique
from os import PathLike
from pathlib import Path

import numpy as np

from ..utils.file import checksum, compress
from .field import FieldType
from .frame import Frame
from .predefined_fields import (
    FTYPES,
    IID,
    LABEL,
    ORIGINAL_IID,
    ORIGINAL_UID,
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


_FRAMES_NPZ = {
    Source.INTERACTION: "inter_frame.npz",
    Source.USER: "user_frame.npz",
    Source.ITEM: "item_frame.npz",
}
_FRAMES_CSV = {
    Source.INTERACTION: "inter_frame.csv",
    Source.USER: "user_frame.csv",
    Source.ITEM: "item_frame.csv",
}


class Data:
    """Data is used to store the interaction, user, and item frame."""

    def __init__(
        self, interaction_frame: Frame, user_frame: Frame, item_frame: Frame
    ) -> None:
        """Initialize the data.

        :param interaction_frame: The interaction frame.
        :param user_frame: The user frame.
        :param item_frame: The item frame.
        """
        super().__init__()

        self._frames: dict[Source, Frame] = {
            Source.INTERACTION: interaction_frame,
            Source.USER: user_frame,
            Source.ITEM: item_frame,
        }
        """The underlying frames in the data."""

    @property
    def interaction_frame(self) -> Frame:
        """The interaction frame."""
        return self._frames[Source.INTERACTION]

    @property
    def num_interactions(self) -> int:
        """The number of interactions."""
        return self.interaction_frame.num_items

    @property
    def user_frame(self) -> Frame:
        """The user frame."""
        return self._frames[Source.USER]

    @property
    def num_users(self) -> int:
        """The number of users."""
        return self.user_frame.num_items

    @property
    def item_frame(self) -> Frame:
        """The item frame."""
        return self._frames[Source.ITEM]

    @property
    def num_items(self) -> int:
        """The number of items."""
        return self.item_frame.num_items

    @property
    def has_timestamp(self) -> bool:
        """Whether the data has timestamp."""
        return TIMESTAMP in self.interaction_frame

    @property
    def is_splitted(self) -> bool:
        """Whether the data is splitted."""
        return (
            {TRAIN_MASK, VAL_MASK, TEST_MASK}.issubset(self.interaction_frame)
            and {TRAIN_IIDS_SET, VAL_IIDS_SET, TEST_IIDS_SET}.issubset(self.user_frame)
            and POP_PROB in self.item_frame
        )

    @property
    def num_train_interactions(self) -> int:
        """The number of training interactions."""
        return self.interaction_frame[TRAIN_MASK].value.astype(np.int64).sum()

    @property
    def num_val_interactions(self) -> int:
        """The number of validation interactions."""
        return self.interaction_frame[VAL_MASK].value.astype(np.int64).sum()

    @property
    def num_test_interactions(self) -> int:
        """The number of test interactions."""
        return self.interaction_frame[TEST_MASK].value.astype(np.int64).sum()

    @property
    def has_eval_negative_samples(self) -> bool:
        """Whether the data has evaluation negative samples."""
        return {VAL_NEG_IIDS, TEST_NEG_IIDS}.issubset(self.user_frame)

    @property
    def num_val_negative_samples(self) -> int:
        """The number of evaluation negative samples."""
        return self.user_frame[VAL_NEG_IIDS].value.shape[1]

    @property
    def num_test_negative_samples(self) -> int:
        """The number of test negative samples."""
        return self.user_frame[TEST_NEG_IIDS].value.shape[1]

    @property
    def base_fields(self) -> dict[Source, list[str]]:
        """The base fields used in HappyRec."""
        fields = {
            Source.INTERACTION: [UID, IID, LABEL],
            Source.USER: [UID, ORIGINAL_UID],
            Source.ITEM: [IID, ORIGINAL_IID],
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
        """The context fields."""
        base_fields = self.base_fields
        fields = {
            Source.INTERACTION: [],
            Source.USER: [],
            Source.ITEM: [],
        }
        for source in Source:
            for field in self._frames[source]:
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
                if field not in self._frames[source]:
                    raise ValueError(f"Mising field {field} in {source} frame.")
                if self._frames[source][field].ftype != FTYPES[field]:
                    raise ValueError(
                        f"Invalid field type of field {field} in {source} frame."
                    )

        # validate the user frame
        if not (
            self.user_frame[UID].value == np.arange(self.user_frame.num_items)
        ).all():
            raise ValueError(
                "UID field in user frame should be a continuous range starting from 0."
            )
        self.user_frame.validate()

        # validate the item frame
        if not (
            self.item_frame[IID].value == np.arange(self.item_frame.num_items)
        ).all():
            raise ValueError(
                "IID field in item frame should be a continuous range starting from 0."
            )
        self.item_frame.validate()

        # validate the inter frame
        if (
            self.interaction_frame[UID].value.min() < 0
            or self.interaction_frame[UID].value.max() >= self.user_frame.num_items
        ):
            raise ValueError(
                "UID field in inter frame should be in the range of [0, num_users)."
            )
        if (
            self.interaction_frame[IID].value.min() < 0
            or self.interaction_frame[IID].value.max() >= self.item_frame.num_items
        ):
            raise ValueError(
                "IID field in inter frame should be in the range of [0, num_items)."
            )
        self.interaction_frame.validate()

        return self

    def __str__(self) -> str:
        return (
            self.interaction_frame.to_string()
            + f"\n[{self.num_interactions} interactions"
            + f" x {self.interaction_frame.num_fields} fields]"
            + self.user_frame.to_string()
            + f"\n[{self.num_users} users"
            + f" x {self.user_frame.num_fields} fields]"
            + self.item_frame.to_string()
            + f"\n[{self.num_items} items"
            + f" x {self.item_frame.num_fields} fields]"
        )

    def info(self) -> None:
        """Print the information of the data."""
        print(self.__class__)
        print(f"# Interactions: {self.num_interactions}")
        print(f"With timestamp: {self.has_timestamp}")
        print(f"Splitted: {self.is_splitted}")
        if self.is_splitted:
            print(f"# Training interactions: {self.num_train_interactions}")
            print(f"# Validation interactions: {self.num_val_interactions}")
            print(f"# Test interactions: {self.num_test_interactions}")
        print(f"Base fields of interactions: {self.base_fields[Source.INTERACTION]}")
        print(
            f"Context fields of interactions: {self.context_fields[Source.INTERACTION]}"
        )
        print(self.interaction_frame.describe())
        print(f"# Users: {self.num_users}")
        print(f"With evaluation negative samples: {self.has_eval_negative_samples}")
        if self.has_eval_negative_samples:
            print(f"# Validation negative samples: {self.num_val_negative_samples}")
            print(f"# Test negative samples: {self.num_test_negative_samples}")
        print(f"Base fields of users: {self.base_fields[Source.USER]}")
        print(f"Context fields of users: {self.context_fields[Source.USER]}")
        print(self.user_frame.describe())
        print(f"# Items: {self.num_items}")
        print(f"Base fields of items: {self.base_fields[Source.ITEM]}")
        print(f"Context fields of items: {self.context_fields[Source.ITEM]}")
        print(self.item_frame.describe())

    def to_npz(self, path: str | PathLike) -> None:
        """Save the data to npz files.

        :param path: The directory to save the data.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self._frames.items():
            frame.to_npz(path / _FRAMES_NPZ[source])

    @classmethod
    def from_npz(cls, path: str | PathLike) -> "Data":
        """Load the data from npz files.

        :param path: The directory to load the data.
        :return: The loaded data.
        """
        path = Path(path)
        interaction_frame = Frame.from_npz(path / _FRAMES_NPZ[Source.INTERACTION])
        user_frame = Frame.from_npz(path / _FRAMES_NPZ[Source.USER])
        item_frame = Frame.from_npz(path / _FRAMES_NPZ[Source.ITEM])
        return cls(interaction_frame, user_frame, item_frame)

    def to_csv(self, path: str | PathLike) -> None:
        """Save the data to csv files.

        :param path: The directory to save the data.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for source, frame in self._frames.items():
            frame.to_csv(path / _FRAMES_CSV[source])

    @classmethod
    def from_csv(cls, path: str | PathLike) -> "Data":
        """Load the data from csv files.

        :param path: The directory to load the data.
        :return: The loaded data.
        """
        path = Path(path)
        interaction_frame = Frame.from_csv(path / _FRAMES_CSV[Source.INTERACTION])
        user_frame = Frame.from_csv(path / _FRAMES_CSV[Source.USER])
        item_frame = Frame.from_csv(path / _FRAMES_CSV[Source.ITEM])
        return cls(interaction_frame, user_frame, item_frame)


_FRAMES_NPZ_XZ = {
    Source.INTERACTION: "inter_frame.npz.xz",
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
    num_interactions: int
    """Number of interactions."""
    num_users: int
    """Number of users."""
    num_items: int
    """Number of items."""
    base_fields: dict[Source, dict[str, FieldType]]
    """Base fields for each source."""
    context_fields: dict[Source, dict[str, FieldType]]
    """Context fields for each source."""
    files_size: dict[Source, int]
    """Size of the data file for each source."""
    compressed_files_size: dict[Source, int]
    """Size of the compressed data file for each source."""
    files_checksum: dict[Source, str]
    """SHA256 Checksum of the compressed data file for each source."""

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
        data = Data.from_npz(path)

        base_fields: dict[Source, dict[str, FieldType]] = {}
        for source, fields in data.base_fields.items():
            frame = data._frames[source]
            base_fields[source] = {name: frame[name].ftype for name in fields}

        context_fields: dict[Source, dict[str, FieldType]] = {}
        for source, fields in data.context_fields.items():
            frame = data._frames[source]
            context_fields[source] = {name: frame[name].ftype for name in fields}

        files_size: dict[Source, int] = {
            source: (path / filename).stat().st_size
            for source, filename in _FRAMES_NPZ.items()
        }

        compressed_files_size: dict[Source, int] = {}
        files_checksum: dict[Source, str] = {}
        for source, filename in _FRAMES_NPZ_XZ.items():
            compressed_file_path = path / filename
            if not compressed_file_path.exists():
                file_path = path / _FRAMES_NPZ[source]
                compress(file_path, compressed_file_path)
            compressed_files_size[source] = compressed_file_path.stat().st_size
            files_checksum[source] = checksum(compressed_file_path)

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            num_interactions=data.num_interactions,
            num_users=data.num_users,
            num_items=data.num_items,
            base_fields=base_fields,
            context_fields=context_fields,
            files_size=files_size,
            compressed_files_size=compressed_files_size,
            files_checksum=files_checksum,
        )

    def to_json(self, path: str | PathLike, pretty_print=False):
        """Write the data info to a json file.

        :param path: The directory to save the data info.
        :param pretty_print: Whether to pretty print the json file.
        """
        path = Path(path)
        data_info_dict = {
            "description": self.description,
            "citation": self.citation,
            "homepage": self.homepage,
            "num_interactions": self.num_interactions,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "base_fields": {
                source.value: {name: str(ftype) for name, ftype in fields.items()}
                for source, fields in self.base_fields.items()
            },
            "context_fields": {
                source.value: {name: str(ftype) for name, ftype in fields.items()}
                for source, fields in self.context_fields.items()
            },
            "files_size": {
                source.value: size for source, size in self.files_size.items()
            },
            "compressed_files_size": {
                source.value: size
                for source, size in self.compressed_files_size.items()
            },
            "files_checksum": {
                source.value: checksum
                for source, checksum in self.files_checksum.items()
            },
        }
        with open(path / _DATA_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(data_info_dict, f, indent=4 if pretty_print else None)

    @classmethod
    def from_json(cls, path: str | PathLike) -> "DataInfo":
        """Create a data info from a json file.

        :param path: The directory to load the data info.
        :return: The data info.
        """
        path = Path(path)
        with open(path / _DATA_INFO_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        data_info_dict = {
            "description": obj["description"],
            "citation": obj["citation"],
            "homepage": obj["homepage"],
            "num_interactions": obj["num_interactions"],
            "num_users": obj["num_users"],
            "num_items": obj["num_items"],
            "base_fields": {
                Source(source): {
                    name: FieldType.from_string(ftype) for name, ftype in fields.items()
                }
                for source, fields in obj["base_fields"].items()
            },
            "context_fields": {
                Source(source): {
                    name: FieldType.from_string(ftype) for name, ftype in fields.items()
                }
                for source, fields in obj["context_fields"].items()
            },
            "files_size": {
                Source(source): size for source, size in obj["files_size"].items()
            },
            "compressed_files_size": {
                Source(source): size
                for source, size in obj["compressed_files_size"].items()
            },
            "files_checksum": {
                Source(source): checksum
                for source, checksum in obj["files_checksum"].items()
            },
        }
        return cls(**data_info_dict)
