import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from happyrec.data import DataInfo, Frame
from happyrec.data import field_types as ftp
from happyrec.data.predefined_fields import IID, LABEL, TIMESTAMP, UID
from happyrec.utils.data import convert_dataframe_to_frame, create_data
from happyrec.utils.logger import logger

MOVIELENS_25M_DESCRIPTION = (
    "MovieLens 25M movie ratings. Stable benchmark dataset. 25 million ratings from "
    "162,000 users on 62,000 movies. Released 12/2019."
)
MOVIELENS_25M_CITATION = (
    "@article{harper2015movielens,\n"
    "  title={The movielens datasets: History and context},\n"
    "  author={Harper, F Maxwell and Konstan, Joseph A},\n"
    "  journal={Acm transactions on interactive intelligent systems (tiis)},\n"
    "  volume={5},\n"
    "  number={4},\n"
    "  pages={1--19},\n"
    "  year={2015},\n"
    "  publisher={ACM New York, NY, USA}\n"
    "}"
)
MOVIELENS_25M_HOMEPAGE = "https://grouplens.org/datasets/movielens/25m/"

RAW_MOVIELENS_25M_DIR = Path("~/Datasets/raw-data/movielens/25m").expanduser()
MOVIELENS_25M_DIR = Path("~/Datasets/preprocessed-data/movielens/25m").expanduser()
MOVIELENS_25M_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(item: str | None) -> str | None:
    if isinstance(item, str) and item.strip() != "":
        return item.strip()
    return None


def str_to_cate_seq(item: str | None) -> np.ndarray:
    if isinstance(item, str):
        return np.array(item.split("|"), dtype="object")
    return np.array(["unknown"], dtype="object")


def create_interaction_frame(interaction_file: Path) -> Frame:
    interaction_frame = pd.read_csv(interaction_file, sep=",")
    interaction_frame.columns = np.array([UID, IID, LABEL, TIMESTAMP])

    interaction_frame[LABEL] = interaction_frame[LABEL].astype(np.float16)

    interaction_frame = interaction_frame.sort_values(
        by=TIMESTAMP, kind="stable", ignore_index=True
    ).drop_duplicates(subset=[UID, IID], keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            IID: ftp.category(),
            LABEL: ftp.float_(),
            TIMESTAMP: ftp.int_(),
        },
        interaction_frame,
    )


def create_item_frame(item_file: Path, links_file: Path) -> Frame:
    item_frame = pd.read_csv(item_file, sep=",")
    item_frame.columns = np.array([IID, "title", "genres"])
    links_frame = pd.read_csv(links_file, sep=",", dtype="object")
    links_frame.columns = np.array([IID, "imdb_id", "tmdb_id"])
    links_frame[IID] = links_frame[IID].astype(np.int64)
    links_frame["imdb_id"] = "tt" + links_frame["imdb_id"]
    links_frame["tmdb_id"] = links_frame["tmdb_id"].fillna("-1").astype(np.int64)

    item_frame["title"] = item_frame["title"].map(clean_text)
    item_frame["genres"] = item_frame["genres"].map(str_to_cate_seq)
    item_frame["movielens_id"] = deepcopy(item_frame[IID])
    item_frame = item_frame.merge(links_frame, on=IID, how="left")

    item_frame = item_frame.sort_values(
        by=IID, kind="stable", ignore_index=True
    ).drop_duplicates(subset=IID, keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "title": ftp.string(),
            "genres": ftp.list_(ftp.category()),
            "movielens_id": ftp.category(),
            "imdb_id": ftp.category(),
            "tmdb_id": ftp.category(),
        },
        item_frame,
    )


def preprocess() -> None:
    interaction_frame = create_interaction_frame(RAW_MOVIELENS_25M_DIR / "ratings.csv")
    item_frame = create_item_frame(
        RAW_MOVIELENS_25M_DIR / "movies.csv", RAW_MOVIELENS_25M_DIR / "links.csv"
    )

    data = create_data(interaction_frame, None, item_frame)

    print(data)
    data.info()
    data.to_pickle(MOVIELENS_25M_DIR)

    data_info = DataInfo.from_data_files(
        MOVIELENS_25M_DIR,
        description=MOVIELENS_25M_DESCRIPTION,
        citation=MOVIELENS_25M_CITATION,
        homepage=MOVIELENS_25M_HOMEPAGE,
    )
    print(data_info)
    data_info.to_json(MOVIELENS_25M_DIR)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    preprocess()
