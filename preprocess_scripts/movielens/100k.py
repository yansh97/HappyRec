import logging
from pathlib import Path

import numpy as np
import pandas as pd

from happyrec.data import DataInfo, Frame
from happyrec.data import field_types as ftp
from happyrec.data.predefined_fields import IID, LABEL, TIMESTAMP, UID
from happyrec.utils.data import convert_dataframe_to_frame, create_data
from happyrec.utils.logger import logger

MOVIELENS_100K_DESCRIPTION = (
    "MovieLens 100K movie ratings. Stable benchmark dataset. 100,000 ratings from 1000 "
    "users on 1700 movies. Released 4/1998."
)
MOVIELENS_100K_CITATION = (
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
MOVIELENS_100K_HOMEPAGE = "https://grouplens.org/datasets/movielens/100k/"

RAW_MOVIELENS_100K_DIR = Path("~/Datasets/raw-data/movielens/100k").expanduser()
MOVIELENS_100K_DIR = Path("~/Datasets/preprocessed-data/movielens/100k").expanduser()
MOVIELENS_100K_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(item: str | None) -> str | None:
    if isinstance(item, str) and item.strip() != "":
        return item.strip()
    return None


def create_interaction_frame(interaction_file: Path) -> Frame:
    interaction_frame = pd.read_csv(interaction_file, sep="\t", header=None)
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


def create_user_frame(user_file: Path) -> Frame:
    user_frame = pd.read_csv(user_file, sep="|", header=None)
    user_frame.columns = np.array([UID, "age", "gender", "occupation", "zip_code"])

    user_frame = user_frame.sort_values(
        by=UID, kind="stable", ignore_index=True
    ).drop_duplicates(subset=UID, keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            "age": ftp.int_(),
            "gender": ftp.category(),
            "occupation": ftp.category(),
            "zip_code": ftp.category(),
        },
        user_frame,
    )


def create_item_frame(item_file: Path) -> Frame:
    item_frame = pd.read_csv(item_file, sep="|", header=None, encoding="ISO-8859-1")
    item_frame = item_frame.drop(columns=[3, 4])
    item_frame.columns = np.array(
        [
            IID,
            "title",
            "release_date",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
    )

    columns = item_frame.columns[3:]

    def create_genres(row) -> np.ndarray:
        genres = []
        for column in columns:
            if row[column] == 1:
                genres.append(str(column))
        if len(genres) == 0:
            genres.append("unknown")
        return np.array(genres, dtype="object")

    item_frame["title"] = item_frame["title"].map(clean_text)
    item_frame["release_date"] = item_frame["release_date"].map(clean_text)
    item_frame["genres"] = item_frame.apply(create_genres, axis=1)  # type: ignore

    item_frame = (
        item_frame[[IID, "title", "release_date", "genres"]]
        .sort_values(by=IID, kind="stable", ignore_index=True)
        .drop_duplicates(subset=IID, keep="first", ignore_index=True)
    )

    return convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "title": ftp.string(),
            "release_date": ftp.string(),
            "genres": ftp.list_(ftp.category()),
        },
        item_frame,
    )


def preprocess() -> None:
    interaction_frame = create_interaction_frame(RAW_MOVIELENS_100K_DIR / "u.data")
    user_frame = create_user_frame(RAW_MOVIELENS_100K_DIR / "u.user")
    item_frame = create_item_frame(RAW_MOVIELENS_100K_DIR / "u.item")

    data = create_data(interaction_frame, user_frame, item_frame)

    print(data)
    data.info()
    data.to_pickle(MOVIELENS_100K_DIR)

    data_info = DataInfo.from_data_files(
        MOVIELENS_100K_DIR,
        description=MOVIELENS_100K_DESCRIPTION,
        citation=MOVIELENS_100K_CITATION,
        homepage=MOVIELENS_100K_HOMEPAGE,
    )
    print(data_info)
    data_info.to_json(MOVIELENS_100K_DIR)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    preprocess()
