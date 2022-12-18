import io
import json
import multiprocessing
import multiprocessing as mp
import subprocess
from os import PathLike
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from PIL import Image
from rich.progress import track

from ..constants import IID, UID
from ..data import CategoryInfo, Data, Field, FieldType, Frame
from ..data.field_types import (
    FixedSizeListFtype,
    ImageFtype,
    ListFtype,
    ScalarFtype,
    TextFtype,
    category,
)
from ..data.transforms import (
    Compose,
    DowncastDtype,
    FactorizeCategoryFields,
    FilterByUnusedUIDsAndIIDs,
)
from .asserts import assert_never_type
from .logger import logger


def load_from_jsonline(
    path: str | PathLike,
    num_items: int | None = None,
    usecols: list[str] | None = None,
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> pd.DataFrame:
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


def convert_image_to_jpeg(image: bytes) -> bytes:
    if image == b"":
        return b""
    bytes_in = io.BytesIO(image)
    bytes_out = io.BytesIO()
    Image.open(bytes_in).convert("RGB").save(bytes_out, format="JPEG")
    return bytes_out.getvalue()


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
        if isinstance(ftype, ScalarFtype | TextFtype | ImageFtype):
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


def create_default_user_frame(interaction_frame: Frame) -> Frame:
    uid_value = np.sort(np.unique(interaction_frame[UID].value), kind="stable")
    return Frame({UID: Field(category(), uid_value)})


def create_default_item_frame(interaction_frame: Frame) -> Frame:
    iid_value = np.sort(np.unique(interaction_frame[IID].value), kind="stable")
    return Frame({IID: Field(category(), iid_value)})


def create_data(
    interaction_frame: Frame,
    user_frame: Frame | None = None,
    item_frame: Frame | None = None,
) -> tuple[Data, CategoryInfo]:
    if user_frame is None:
        user_frame = create_default_user_frame(interaction_frame)
    if item_frame is None:
        item_frame = create_default_item_frame(interaction_frame)

    data = Data.from_frames(interaction_frame, user_frame, item_frame)
    factorizer = FactorizeCategoryFields(CategoryInfo())
    transforms = Compose.from_transforms(
        FilterByUnusedUIDsAndIIDs(),
        factorizer,
        DowncastDtype(),
    )
    data = transforms(data)
    category_info = cast(CategoryInfo, factorizer.category_info)

    return data, category_info


def compress(src: str | PathLike, dst: str | PathLike) -> None:
    src = Path(src)
    dst = Path(dst)
    logger.info(f"Compressing {src.name} to {dst.name} ...")
    cores = multiprocessing.cpu_count() // 2
    cmd = f"xz -zcf6v -T{cores} {src} > {dst}"
    subprocess.run(cmd, shell=True)
