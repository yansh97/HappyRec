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

from ..data import Data, Field, FieldType, Frame
from ..data.field_types import FixedSizeListFtype, ListFtype, ScalarFtype, category
from ..data.predefined_fields import IID, UID
from .asserts import assert_never_type
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


def convert_str_to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).astype(int) // 10**9


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
) -> Frame:
    """Convert a Pandas DataFrame to a frame.

    :param ftypes: The field types of the frame.
    :param dataframe: The Pandas DataFrame.
    :return: The converted frame.
    """
    fields: dict[str, Field] = {}
    for name, ftype in ftypes.items():
        if isinstance(ftype, ScalarFtype):
            value = dataframe[name].to_numpy()
        elif isinstance(ftype, FixedSizeListFtype):
            value = np.vstack(dataframe[name])  # type: ignore
        elif isinstance(ftype, ListFtype):
            value = np.fromiter(
                dataframe[name], dtype=object, count=len(dataframe[name])
            )
        else:
            assert_never_type(ftype)
        fields[name] = Field(ftype, value)
    return Frame(fields)


def filter_unused_uids_and_iids(
    interaction_frame: Frame,
    user_frame: Frame | None = None,
    item_frame: Frame | None = None,
) -> tuple[Frame, Frame | None, Frame | None]:
    array_in: Callable[[set, np.ndarray], np.ndarray] = np.vectorize(operator.contains)

    interaction_mask = np.full(interaction_frame.num_elements, True)
    num_interactions = interaction_frame.num_elements
    if user_frame is not None:
        uid_set = set(user_frame[UID].value)
        interaction_mask &= array_in(uid_set, interaction_frame[UID].value)
    if item_frame is not None:
        iid_set = set(item_frame[IID].value)
        interaction_mask &= array_in(iid_set, interaction_frame[IID].value)
    interaction_frame = interaction_frame.loc_elements[interaction_mask]
    logger.debug(
        "Filtered interactions: %d -> %d",
        num_interactions,
        interaction_frame.num_elements,
    )

    if user_frame is not None:
        num_users = user_frame.num_elements
        uid_set = set(interaction_frame[UID].value)
        user_frame = user_frame.loc_elements[array_in(uid_set, user_frame[UID].value)]
        logger.debug("Filtered users: %d -> %d", num_users, user_frame.num_elements)
    if item_frame is not None:
        num_items = item_frame.num_elements
        iid_set = set(interaction_frame[IID].value)
        item_frame = item_frame.loc_elements[array_in(iid_set, item_frame[IID].value)]
        logger.debug("Filtered items: %d -> %d", num_items, item_frame.num_elements)

    return interaction_frame, user_frame, item_frame


def create_default_user_frame(interaction_frame: Frame) -> Frame:
    """Create a default user frame from an interaction frame.

    :param inter_frame: The interaction frame.
    :return: The user frame.
    """
    uid_value = np.sort(np.unique(interaction_frame[UID].value), kind="stable")
    return Frame({UID: Field(category(), uid_value)})


def create_default_item_frame(interaction_frame: Frame) -> Frame:
    """Create a default item frame from an interaction frame.

    :param inter_frame: The interaction frame.
    :return: The item frame.
    """
    iid_value = np.sort(np.unique(interaction_frame[IID].value), kind="stable")
    return Frame({IID: Field(category(), iid_value)})


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

    data = Data.from_frames(interaction_frame, user_frame, item_frame)
    data = data.downcast()
    data.validate()

    return data
