from collections.abc import Sequence
from os import PathLike
from typing import IO

import numpy as np
import pandas as pd

from ..config.constants import FIELD_SEP
from ..utils.fileio import get_io_handle
from .column import get_col_type


def downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast each column in the frame.

    :param frame: The frame.
    :return: The downcasted frame.
    """
    ret = pd.DataFrame()
    for name in frame.columns:
        ret[name] = get_col_type(name).downcast(frame[name])
    return ret


def validate(frame: pd.DataFrame) -> None:
    """Validate the frame.

    :param frame: The frame.
    :raises ValueError: The frame must have at least one column.
    :raises ValueError: Some columns have invalid values.
    """
    if len(frame.columns) == 0:
        raise ValueError("The frame must have at least one column.")

    for name in frame.columns:
        if not get_col_type(name).check_value(frame[name]):
            raise ValueError(f"The column {name} has invalid values.")


def from_csv(file: str | PathLike | IO) -> pd.DataFrame:
    """Load the frame from a CSV file.

    :param file: The file path or file object.
    :return: The frame.
    """
    str_frame = pd.read_csv(
        file, sep=FIELD_SEP, header=0, index_col=False, dtype=np.str_
    )
    ret = pd.DataFrame()
    for name in str_frame.columns:
        ret[name] = get_col_type(name).from_str_series(str_frame[name])
    return ret


def to_csv(
    frame: pd.DataFrame, file: str | PathLike | IO, buffer_len: int = 10000
) -> None:
    """Save the frame to a file in the CSV format.

    :param file: The file path or file object.
    :param buffer_len: The length of the buffer used to write the file.
    :return: None.
    """
    total_rows = len(frame.index)
    start_idx: int = 0
    while start_idx < total_rows:
        end_idx = min(start_idx + buffer_len, total_rows)
        buffer_df = pd.DataFrame()
        for name in frame.columns:
            buffer_df[name] = get_col_type(name).to_str_series(
                frame[name][start_idx:end_idx]
            )
        mode = "w" if start_idx == 0 else "a"
        with get_io_handle(file, mode) as fout:
            buffer_df.to_csv(fout.io, sep=FIELD_SEP, index=False, header=start_idx == 0)
        start_idx = end_idx


def concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple frames into one.

    :param frames: The frames to concatenate.
    :raises ValueError: The frames have different column names.
    :return: The concatenated frame.
    """
    names: list[str] = sorted(frames[0].columns.to_list())
    for frame in frames[1:]:
        if sorted(frame.columns.to_list()) != names:
            raise ValueError("The frames have different column names.")
    return downcast(pd.concat(frames, ignore_index=True))
