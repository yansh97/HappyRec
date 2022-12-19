import logging
from pathlib import Path

import pandas as pd
from rich.progress import track

from happyrec.constants import IID, LABEL, TIMESTAMP, UID
from happyrec.data import DataInfo, Frame
from happyrec.data import ftypes as ftp
from happyrec.utils.logger import logger
from happyrec.utils.preprocessing import convert_dataframe_to_frame, create_data

NETFLIX_DESCRIPTION = (
    "Netflix held the Netflix Prize open competition for the best algorithm to predict "
    "user ratings for films. This is the dataset that was used in that competition."
)
NETFLIX_CITATION = ""
NETFLIX_HOMEPAGE = "https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/"

RAW_NETFLIX_DIR = Path("~/Datasets/raw-data/netflix").expanduser()
NETFLIX_DIR = Path("~/Datasets/preprocessed-data/netflix").expanduser()
NETFLIX_DIR.mkdir(parents=True, exist_ok=True)


def create_interaction_frame(interaction_files: list[tuple[Path, int]]) -> Frame:
    interaction_dict = {UID: [], IID: [], LABEL: [], TIMESTAMP: []}

    iid = -1
    for path, num_lines in interaction_files:
        with open(path, "r", encoding="utf-8") as file:
            for line in track(
                file, description=f"Reading {path} ...", total=num_lines, transient=True
            ):
                line = line.strip()
                if line.endswith(":"):
                    iid = int(line[:-1])
                else:
                    uid, rating, date = line.split(",")
                    interaction_dict[UID].append(int(uid))
                    interaction_dict[IID].append(iid)
                    interaction_dict[LABEL].append(int(rating))
                    interaction_dict[TIMESTAMP].append(date)

    interaction_frame = pd.DataFrame(interaction_dict)
    interaction_frame[TIMESTAMP] = (
        pd.to_datetime(interaction_frame[TIMESTAMP]).astype(int) // 10**9
    )

    interaction_frame = interaction_frame.sort_values(
        by=TIMESTAMP, kind="stable", ignore_index=True
    ).drop_duplicates(subset=[UID, IID], keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            IID: ftp.category(),
            LABEL: ftp.int_(),
            TIMESTAMP: ftp.int_(),
        },
        interaction_frame,
    )


def create_item_frame(item_file: Path, num_items: int) -> Frame:
    item_dict = {IID: [], "year": [], "title": []}

    with open(item_file, "r", encoding="iso-8859-1") as file:
        for line in track(
            file,
            description=f"Reading {item_file} ...",
            total=num_items,
            transient=True,
        ):
            iid, year, title = line.strip().split(",", 2)
            item_dict[IID].append(int(iid))
            item_dict["year"].append(-1 if year == "NULL" else int(year))
            item_dict["title"].append(title if len(title) > 0 else None)

    item_frame = pd.DataFrame(item_dict)

    item_frame = item_frame.sort_values(
        by=IID, kind="stable", ignore_index=True
    ).drop_duplicates(subset=IID, keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "year": ftp.int_(),
            "title": ftp.text(),
        },
        item_frame,
    )


def preprocess() -> None:
    interaction_frame = create_interaction_frame(
        [
            (RAW_NETFLIX_DIR / "combined_data_1.txt", 24058263),
            (RAW_NETFLIX_DIR / "combined_data_2.txt", 26982302),
            (RAW_NETFLIX_DIR / "combined_data_3.txt", 22605786),
            (RAW_NETFLIX_DIR / "combined_data_4.txt", 26851926),
        ]
    )
    item_frame = create_item_frame(RAW_NETFLIX_DIR / "movie_titles.csv", 17770)

    data, category_info = create_data(interaction_frame, None, item_frame)

    print(data)
    data.info()
    data.to_feather(NETFLIX_DIR)

    data_info = DataInfo.from_data_files(
        NETFLIX_DIR,
        description=NETFLIX_DESCRIPTION,
        citation=NETFLIX_CITATION,
        homepage=NETFLIX_HOMEPAGE,
    )
    data_info.to_json(NETFLIX_DIR)

    category_info.to_json(NETFLIX_DIR)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    preprocess()
