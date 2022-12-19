import logging
from pathlib import Path

import numpy as np
import pandas as pd

from happyrec.constants import IID, LABEL, TIMESTAMP, UID
from happyrec.data import DataInfo, Frame
from happyrec.data import ftypes as ftp
from happyrec.utils.logger import logger
from happyrec.utils.preprocessing import convert_dataframe_to_frame, create_data

XING_DESCRIPTION = (
    "The ACM RecSys Challenge 2017 is focussing on the problem of job recommendations "
    "on XING in a cold-start scenario. This is the dataset that was used in that "
    "challenge.\n"
    "Note: The dataset contains duplicated user-item interactions with different "
    "timestamps."
)
XING_CITATION = ""
XING_HOMEPAGE = "http://www.recsyschallenge.com/2017/"

RAW_XING_DIR = Path("~/Datasets/raw-data/xing").expanduser()
XING_DIR = Path("~/Datasets/preprocessed-data/xing").expanduser()
XING_DIR.mkdir(parents=True, exist_ok=True)


def str_to_cate_seq(item: str | None) -> np.ndarray:
    if isinstance(item, str) and len(item) > 0:
        return np.array(item.split(",")).astype(np.int64)
    return np.array([-1], dtype=np.int64)


def create_interaction_frame(interaction_file: Path) -> Frame:
    interaction_frame = pd.read_csv(interaction_file, sep="\t")
    interaction_frame.columns = np.array([UID, IID, LABEL, TIMESTAMP])

    interaction_frame[LABEL][interaction_frame[LABEL] == 4] = -4

    interaction_frame = interaction_frame.sort_values(
        by=TIMESTAMP, kind="stable", ignore_index=True
    )

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            IID: ftp.category(),
            LABEL: ftp.int_(),
            TIMESTAMP: ftp.int_(),
        },
        interaction_frame,
    )


def create_user_frame(user_file: Path) -> Frame:
    user_frame = pd.read_csv(user_file, sep="\t")
    user_frame.columns = np.array(
        [
            UID,
            "jobroles",
            "user_career_level",
            "user_discipline_id",
            "user_industry_id",
            "user_country",
            "user_region",
            "experience_n_entries_class",
            "experience_years_experience",
            "experience_years_in_current",
            "edu_degree",
            "edu_fieldofstudies",
            "wtcj",
            "premium",
        ]
    )

    user_frame["jobroles"] = user_frame["jobroles"].map(str_to_cate_seq)
    user_frame["edu_fieldofstudies"] = user_frame["edu_fieldofstudies"].map(
        str_to_cate_seq
    )

    user_frame = (
        user_frame[
            [
                UID,
                "jobroles",
                "user_career_level",
                "user_discipline_id",
                "user_industry_id",
                "user_country",
                "user_region",
                "experience_n_entries_class",
                "experience_years_experience",
                "experience_years_in_current",
                "edu_degree",
                "edu_fieldofstudies",
                "wtcj",
                "premium",
            ]
        ]
        .sort_values(by=UID, kind="stable", ignore_index=True)
        .drop_duplicates(subset=UID, keep="first", ignore_index=True)
    )

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            "jobroles": ftp.list_(ftp.category()),
            "user_career_level": ftp.category(),
            "user_discipline_id": ftp.category(),
            "user_industry_id": ftp.category(),
            "user_country": ftp.category(),
            "user_region": ftp.category(),
            "experience_n_entries_class": ftp.category(),
            "experience_years_experience": ftp.category(),
            "experience_years_in_current": ftp.category(),
            "edu_degree": ftp.category(),
            "edu_fieldofstudies": ftp.list_(ftp.category()),
            "wtcj": ftp.category(),
            "premium": ftp.category(),
        },
        user_frame,
    )


def create_item_frame(raw_item_path: Path) -> Frame:
    item_frame = pd.read_csv(raw_item_path, sep="\t")
    item_frame.columns = np.array(
        [
            IID,
            "title",
            "item_career_level",
            "item_discipline_id",
            "item_industry_id",
            "item_country",
            "is_paid",
            "item_region",
            "latitude",
            "longitude",
            "employment",
            "tags",
            "created_at",
        ]
    )
    item_frame = item_frame.drop(columns=["latitude", "longitude"])

    item_frame["title"] = item_frame["title"].map(str_to_cate_seq)
    item_frame["tags"] = item_frame["tags"].map(str_to_cate_seq)
    item_frame["created_at"] = item_frame["created_at"].fillna("-1").astype(np.int64)

    item_frame = (
        item_frame[
            [
                IID,
                "title",
                "item_career_level",
                "item_discipline_id",
                "item_industry_id",
                "item_country",
                "is_paid",
                "item_region",
                "employment",
                "tags",
                "created_at",
            ]
        ]
        .sort_values(by=IID, kind="stable", ignore_index=True)
        .drop_duplicates(subset=IID, keep="first", ignore_index=True)
    )

    return convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "title": ftp.list_(ftp.category()),
            "item_career_level": ftp.category(),
            "item_discipline_id": ftp.category(),
            "item_industry_id": ftp.category(),
            "item_country": ftp.category(),
            "is_paid": ftp.category(),
            "item_region": ftp.category(),
            "employment": ftp.category(),
            "tags": ftp.list_(ftp.category()),
            "created_at": ftp.int_(),
        },
        item_frame,
    )


def preprocess() -> None:
    interaction_frame = create_interaction_frame(RAW_XING_DIR / "interactions.csv")
    user_frame = create_user_frame(RAW_XING_DIR / "users.csv")
    item_frame = create_item_frame(RAW_XING_DIR / "items.csv")

    data, category_info = create_data(interaction_frame, user_frame, item_frame)

    print(data)
    data.info()
    data.to_feather(XING_DIR)

    data_info = DataInfo.from_data_files(
        XING_DIR,
        description=XING_DESCRIPTION,
        citation=XING_CITATION,
        homepage=XING_HOMEPAGE,
    )
    data_info.to_json(XING_DIR)

    category_info.to_json(XING_DIR)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    preprocess()
