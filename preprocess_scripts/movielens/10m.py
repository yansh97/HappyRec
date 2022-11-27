import logging
from pathlib import Path

import numpy as np
import pandas as pd

from happyrec.data import CategoricalType, DataInfo, Frame, ItemType, TextType
from happyrec.data.predefined_fields import FTYPES, IID, LABEL, TIMESTAMP, UID
from happyrec.utils.data import convert_dataframe_to_frame, create_data
from happyrec.utils.logger import logger

MOVIELENS_10M_DESCRIPTION = (
    "MovieLens 10M movie ratings. Stable benchmark dataset. 10 million ratings and "
    "100,000 tag applications applied to 10,000 movies by 72,000 users. Released "
    "1/2009."
)
MOVIELENS_10M_CITATION = (
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
MOVIELENS_10M_HOMEPAGE = "https://grouplens.org/datasets/movielens/10m/"

RAW_MOVIELENS_10M_DIR = Path("~/Datasets/raw-data/movielens/10m").expanduser()
MOVIELENS_10M_DIR = Path("~/Datasets/preprocessed-data/movielens/10m").expanduser()
MOVIELENS_10M_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(item: str | None) -> str | None:
    if isinstance(item, str) and item.strip() != "":
        return item.strip()
    return None


def str_to_cate_seq(item: str | None) -> np.ndarray:
    if isinstance(item, str):
        return np.array(item.split("|"), dtype="object")
    return np.array(["unknown"], dtype="object")


def create_interaction_frame(interaction_file: Path) -> Frame:
    interaction_frame = pd.read_csv(
        interaction_file, sep="::", header=None, engine="python"
    )
    interaction_frame.columns = np.array([UID, IID, LABEL, TIMESTAMP])

    interaction_frame[LABEL] = interaction_frame[LABEL].astype(np.float16)

    interaction_frame = interaction_frame.sort_values(
        by=TIMESTAMP, kind="stable", ignore_index=True
    ).drop_duplicates(subset=[UID, IID], keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            UID: FTYPES[UID],
            IID: FTYPES[IID],
            LABEL: FTYPES[LABEL],
            TIMESTAMP: FTYPES[TIMESTAMP],
        },
        interaction_frame,
    )


def create_item_frame(item_file: Path) -> Frame:
    item_frame = pd.read_csv(item_file, sep="::", header=None, engine="python")
    item_frame.columns = np.array([IID, "title", "genres"])

    item_frame["title"] = item_frame["title"].map(clean_text)
    item_frame["genres"] = item_frame["genres"].map(str_to_cate_seq)

    item_frame = item_frame.sort_values(
        by=IID, kind="stable", ignore_index=True
    ).drop_duplicates(subset=IID, keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            IID: FTYPES[IID],
            "title": TextType(),
            "genres": CategoricalType(ItemType.SEQUENCE),
        },
        item_frame,
    )


def preprocess() -> None:
    interaction_frame = create_interaction_frame(RAW_MOVIELENS_10M_DIR / "ratings.dat")
    item_frame = create_item_frame(RAW_MOVIELENS_10M_DIR / "movies.dat")

    data = create_data(interaction_frame, None, item_frame)

    print(data)
    data.info()
    data.to_npz(MOVIELENS_10M_DIR)

    data_info = DataInfo.from_data_files(
        MOVIELENS_10M_DIR,
        description=MOVIELENS_10M_DESCRIPTION,
        citation=MOVIELENS_10M_CITATION,
        homepage=MOVIELENS_10M_HOMEPAGE,
    )
    print(data_info)
    data_info.to_json(MOVIELENS_10M_DIR)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    preprocess()
