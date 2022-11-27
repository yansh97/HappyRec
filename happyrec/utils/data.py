import io
import json
import multiprocessing as mp
import operator
from os import PathLike
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from PIL import Image
from rich.progress import track

from ..data import (
    CategoricalType,
    Data,
    Field,
    FieldType,
    Frame,
    ImageType,
    ItemType,
    NumericType,
    ObjectType,
    TextType,
)
from ..data.predefined_fields import FTYPES, IID, UID
from .logger import logger


def load_from_jsonline(
    path: str | PathLike,
    num_items: int | None = None,
    usecols: list[str] | None = None,
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> pd.DataFrame:
    """Load data from a jsonline file.

    :param path: Path to the jsonline file.
    :param usecols: List of columns to load.
    :param validator: A function to validate each record.
    :return: A pandas DataFrame.
    """
    path = Path(path)
    logger.info(f"Loading jsonline from {path.name} ...")
    data: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in track(
            file,
            description=f"Loading jsonline from {path.name} ...",
            total=num_items,
            transient=True,
        ):
            item: dict[str, Any] = json.loads(line)
            if validator is not None and not validator(item):
                continue
            if usecols is not None:
                item = {col: item.get(col, None) for col in usecols}
            data.append(item)
    return pd.DataFrame.from_records(data)


def load_from_dictline(
    path: str | PathLike,
    num_items: int | None = None,
    usecols: list[str] | None = None,
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> pd.DataFrame:
    """Load data from a jsonline file.

    :param path: Path to the jsonline file.
    :param usecols: List of columns to load.
    :param validator: A function to validate each record.
    :return: A pandas DataFrame.
    """
    path = Path(path)
    logger.info(f"Loading jsonline from {path.name} ...")
    data: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in track(
            file,
            description=f"Loading jsonline from {path.name} ...",
            total=num_items,
            transient=True,
        ):
            item: dict[str, Any] = eval(line)
            if validator is not None and not validator(item):
                continue
            if usecols is not None:
                item = {col: item.get(col, None) for col in usecols}
            data.append(item)
    return pd.DataFrame.from_records(data)


def convert_image_to_array(path: str | PathLike) -> np.ndarray:
    """Convert an image to a numpy array in jpeg format.

    :param path: The path to the image.
    :return: The converted numpy array.
    """
    path = Path(path)
    byteio = io.BytesIO()
    Image.open(path).convert("RGB").save(byteio, format="JPEG")
    return np.frombuffer(byteio.getvalue(), dtype=np.uint8)


# def str_to_cate_seq(series: pd.Series, sep: str) -> pd.Series:
#     return series.apply(
#         lambda s: np.array(s.split(sep) if pd.notna(s) else [None], dtype=np.str_)
#     )


# def str_to_timestamp(series: pd.Series, /) -> pd.Series:
#     return pd.to_datetime(series).astype(int) // 10**9


# def timestamp_to_cate(series: pd.Series, /) -> pd.Series:
#     return pd.to_datetime(series, unit="s").apply(
#         lambda dt: f"{dt.year}-{dt.month:02d}" if pd.notna(dt) else None
#     )


# def latitude_to_cate(series: pd.Series, /) -> pd.Series:
#     return series.apply(lambda x: int(x) // 10 if pd.notna(x) else None)


# def longitude_to_cate(series: pd.Series, /) -> pd.Series:
#     return series.apply(lambda x: int(x) // 10 if pd.notna(x) else None)


def parallelize(
    dataframe: pd.DataFrame, func: Callable[[pd.DataFrame], pd.DataFrame | Exception]
) -> pd.DataFrame:
    """Parallelize a function on a Pandas DataFrame.

    :param dataframe: The DataFrame.
    :param func: The function to apply.
    :return: The DataFrame with the function applied.
    """
    num_partitions = mp.cpu_count() // 2
    partitions: list[pd.DataFrame] = cast(
        list[pd.DataFrame], np.array_split(dataframe, num_partitions)
    )
    partitions = [partition.copy() for partition in partitions]
    del dataframe
    pool = mp.Pool(num_partitions)
    results = pool.map(func, partitions)
    pool.close()
    pool.join()
    ret_partitions: list[pd.DataFrame] = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        ret_partitions.append(result)
    return pd.concat(ret_partitions)


def convert_dataframe_to_frame(
    ftypes: dict[str, FieldType], dataframe: pd.DataFrame
) -> "Frame":
    """Convert a Pandas DataFrame to a frame.

    :param ftypes: The field types of the frame.
    :param dataframe: The Pandas DataFrame.
    :return: The converted frame.
    """
    fields: dict[str, Field] = {}
    for name, ftype in ftypes.items():
        if isinstance(ftype, TextType | ImageType | ObjectType):
            value = dataframe[name].to_numpy()
        elif isinstance(ftype, NumericType | CategoricalType):
            match ftype.item_type:
                case ItemType.SCALAR:
                    value = dataframe[name].to_numpy()
                case ItemType.ARRAY:
                    value = np.vstack(dataframe[name])  # type: ignore
                case ItemType.SEQUENCE:
                    value = np.fromiter(
                        dataframe[name], dtype=np.object_, count=len(dataframe[name])
                    )
        else:
            raise ValueError(f"Unknown field type {ftype}")
        fields[name] = Field(ftype, value)
    return Frame(fields)


def filter_unused_uids_and_iids(
    interaction_frame: Frame,
    user_frame: Frame | None = None,
    item_frame: Frame | None = None,
) -> tuple[Frame, Frame | None, Frame | None]:
    array_in: Callable[[Any, np.ndarray], np.ndarray] = np.vectorize(operator.contains)

    interaction_mask = np.full(interaction_frame.num_items, True)
    num_interactions = interaction_frame.num_items
    if user_frame is not None:
        uid_set = set(user_frame[UID].value)
        interaction_mask &= array_in(uid_set, interaction_frame[UID].value)
    if item_frame is not None:
        iid_set = set(item_frame[IID].value)
        interaction_mask &= array_in(iid_set, interaction_frame[IID].value)
    interaction_frame = interaction_frame.loc_items[interaction_mask]
    logger.debug(
        "Filtered interactions: %d -> %d", num_interactions, interaction_frame.num_items
    )

    if user_frame is not None:
        num_users = user_frame.num_items
        uid_set = set(interaction_frame[UID].value)
        user_frame = user_frame.loc_items[array_in(uid_set, user_frame[UID].value)]
        logger.debug("Filtered users: %d -> %d", num_users, user_frame.num_items)
    if item_frame is not None:
        num_items = item_frame.num_items
        iid_set = set(interaction_frame[IID].value)
        item_frame = item_frame.loc_items[array_in(iid_set, item_frame[IID].value)]
        logger.debug("Filtered items: %d -> %d", num_items, item_frame.num_items)

    return interaction_frame, user_frame, item_frame


def create_default_user_frame(interaction_frame: Frame) -> Frame:
    """Create a default user frame from an interaction frame.

    :param inter_frame: The interaction frame.
    :return: The user frame.
    """
    uid_value = np.sort(np.unique(interaction_frame[UID].value), kind="stable")
    return Frame({UID: Field(ftype=FTYPES[UID], value=uid_value)})


def create_default_item_frame(interaction_frame: Frame) -> Frame:
    """Create a default item frame from an interaction frame.

    :param inter_frame: The interaction frame.
    :return: The item frame.
    """
    iid_value = np.sort(np.unique(interaction_frame[IID].value), kind="stable")
    return Frame({IID: Field(ftype=FTYPES[IID], value=iid_value)})


def create_data(
    interaction_frame: Frame,
    user_frame: Frame | None = None,
    item_frame: Frame | None = None,
) -> Data:
    """Create a data object from interaction, user, and item frames.

    :param interaction_frame: The interaction frame.
    :param user_frame: The user frame.
    :param item_frame: The item frame.
    :return: The data object.
    """
    interaction_frame, user_frame, item_frame = filter_unused_uids_and_iids(
        interaction_frame, user_frame, item_frame
    )

    if user_frame is None:
        user_frame = create_default_user_frame(interaction_frame)

    if item_frame is None:
        item_frame = create_default_item_frame(interaction_frame)

    data = Data(interaction_frame, user_frame, item_frame)
    for frame in data.frames.values():
        frame.downcast()
    data = data.validate()

    return data
