import logging
from pathlib import Path

import numpy as np
import pandas as pd

from happyrec.constants import IID, LABEL, TIMESTAMP, UID
from happyrec.data import DataInfo, Frame
from happyrec.data import ftypes as ftp
from happyrec.utils.logger import logger
from happyrec.utils.preprocessing import convert_dataframe_to_frame, create_data

GOWALLA_DESCRIPTION = (
    "Gowalla is a location-based social networking website where users share their "
    "locations by checking-in. We have collected a total of 6,442,890 check-ins of "
    "these users over the period of Feb. 2009 - Oct. 2010.\n"
    "Note: The dataset contains duplicated user-item interactions with different "
    "timestamps."
)
GOWALLA_CITATION = (
    "@inproceedings{cho2011friendship,\n"
    "  title={Friendship and mobility: Friendship and mobility: User movement in "
    "location-based social networksfriendship and mobility},\n"
    "  author={Cho, Eunjoon and Myers, Seth A and Leskovec, Jure},\n"
    "  booktitle={User Movement in Location-Based Social Networks ACM SIGKDD "
    "International Conference on Knowledge Discovery and Data Mining (KDD)},\n"
    "  year={2011}\n"
    "}"
)
GOWALLA_HOMEPAGE = "https://snap.stanford.edu/data/loc-gowalla.html"

RAW_GOWALLA_DIR = Path("~/Datasets/raw-data/gowalla").expanduser()
GOWALLA_DIR = Path("~/Datasets/preprocessed-data/gowalla").expanduser()
GOWALLA_DIR.mkdir(parents=True, exist_ok=True)


def create_interaction_and_item_frame(interaction_file: Path) -> tuple[Frame, Frame]:
    frame = pd.read_csv(interaction_file, sep="\t", header=None)
    frame.columns = np.array([UID, TIMESTAMP, "latitude", "longitude", IID])
    frame[LABEL] = np.ones(len(frame), dtype=np.int8)

    interaction_frame = frame[[UID, IID, LABEL, TIMESTAMP]].copy()
    interaction_frame[TIMESTAMP] = (
        pd.to_datetime(interaction_frame[TIMESTAMP]).astype(int) // 10**9
    )
    interaction_frame = interaction_frame.sort_values(
        by=TIMESTAMP, kind="stable", ignore_index=True
    )

    item_frame = frame[[IID, "latitude", "longitude"]].copy()
    item_frame["latitude"] = item_frame["latitude"].clip(-90, 90).astype(np.float32)
    item_frame["longitude"] = item_frame["longitude"].clip(-180, 180).astype(np.float32)
    item_frame = item_frame.drop_duplicates(
        subset=IID, keep="first", ignore_index=True
    ).sort_values(by=IID, kind="stable", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            IID: ftp.category(),
            LABEL: ftp.int_(),
            TIMESTAMP: ftp.int_(),
        },
        interaction_frame,
    ), convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "latitude": ftp.float_(),
            "longitude": ftp.float_(),
        },
        item_frame,
    )


def preprocess() -> None:
    interaction_frame, item_frame = create_interaction_and_item_frame(
        RAW_GOWALLA_DIR / "loc-gowalla_totalCheckins.txt"
    )

    data, category_info = create_data(interaction_frame, None, item_frame)

    print(data)
    data.info()
    data.to_feather(GOWALLA_DIR)

    data_info = DataInfo.from_data_files(
        GOWALLA_DIR,
        description=GOWALLA_DESCRIPTION,
        citation=GOWALLA_CITATION,
        homepage=GOWALLA_HOMEPAGE,
    )
    data_info.to_json(GOWALLA_DIR)

    category_info.to_json(GOWALLA_DIR)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    preprocess()
