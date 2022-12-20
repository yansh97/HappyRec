import json
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import logger
from .core import Data, Source


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

    def info(self, max_colwidth: int | None = 25) -> None:
        """Print the field information."""
        fields = self.fields
        for source, field_info in fields.items():
            print(f"{source} fields:")
            info = [{"name": name, **info} for name, info in field_info.items()]
            info_dict = {
                key: [field_info[key] for field_info in info] for key in info[0].keys()
            }
            info_frame = pd.DataFrame(info_dict, dtype=object)
            print(info_frame.to_string(index=False, max_colwidth=max_colwidth))

    @classmethod
    def from_data_files(
        cls,
        path: str | PathLike,
        description: str = "",
        citation: str = "",
        homepage: str = "",
    ) -> "DataInfo":
        path = Path(path)
        logger.info(f"Creating data information from {path} ...")
        data = Data.from_feather(path)

        num_elements: dict[str, int] = {
            source.value: num for source, num in data.num_elements.items()
        }

        base_fields: dict[str, dict[str, dict[str, Any]]] = {}
        for source, fields in data.base_fields.items():
            frame = data[source]
            base_fields[source.value] = {
                name: frame[name].get_info_dict() for name in fields
            }

        context_fields: dict[str, dict[str, dict[str, Any]]] = {}
        for source, fields in data.context_fields.items():
            frame = data[source]
            context_fields[source.value] = {
                name: frame[name].get_info_dict() for name in fields
            }

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            num_elements=num_elements,
            base_fields=base_fields,
            context_fields=context_fields,
        )

    def to_json(self, path: str | PathLike, pretty_print=False):
        path = Path(path)
        logger.info(f"Saving data info to {path} ...")
        with open(path / _DATA_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4 if pretty_print else None)

    @classmethod
    def from_json(cls, path: str | PathLike) -> "DataInfo":
        path = Path(path)
        logger.info(f"Loading data info from {path} ...")
        with open(path / _DATA_INFO_JSON, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass(frozen=True, kw_only=True, slots=True)
class CategoryInfo:
    """Category information."""

    field_names: list[str] = field(default_factory=list)
    unique_values: dict[str, list[Any]] = field(default_factory=dict)

    def to_json(self, path: str | PathLike, pretty_print=False):
        path = Path(path)
        logger.info(f"Saving data info to {path} ...")
        with open(path / _CATEGORY_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4 if pretty_print else None)

    @classmethod
    def from_json(cls, path: str | PathLike) -> "CategoryInfo":
        path = Path(path)
        logger.info(f"Loading data info from {path} ...")
        with open(path / _CATEGORY_INFO_JSON, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def update(self, field_name: str, unique_value: np.ndarray) -> None:
        if field_name not in self.field_names:
            self.field_names.append(field_name)
            self.unique_values[field_name] = unique_value.tolist()
        else:
            if unique_value.dtype not in ("uint8", "uint16", "uint32", "uint64"):
                raise ValueError
            if unique_value.max() >= len(self.unique_values[field_name]):
                raise ValueError
            old_unique_value = self.unique_values[field_name]
            self.unique_values[field_name] = [old_unique_value[i] for i in unique_value]


_DATA_INFO_JSON = "data_info.json"
_CATEGORY_INFO_JSON = "category_info.json"
