import gc
import logging
import operator
import re
from collections import defaultdict
from functools import partial
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from selectolax.parser import HTMLParser

from happyrec.constants import IID, LABEL, TIMESTAMP, UID
from happyrec.data import DataInfo, Frame
from happyrec.data import field_types as ftp
from happyrec.utils.logger import logger
from happyrec.utils.preprocessing import (
    convert_dataframe_to_frame,
    convert_image_to_jpeg,
    create_data,
    load_from_dictline,
    parallelize,
)

AMAZON_2014_DESCRIPTION = (
    "Amazon Review Data includes reviews (ratings and text) and product metadata "
    "(descriptions, category information, brand, and other features). This version of "
    "the dataset is released in 2014. Note that this dataset have been filtered by "
    "5-core files on the homepage."
)
AMAZON_2014_CITATION = (
    "@inproceedings{he2016ups,\n"
    "  title={Ups and downs: Modeling the visual evolution of fashion trends with one-"
    "class collaborative filtering},\n"
    "  author={He, Ruining and McAuley, Julian},\n"
    "  booktitle={proceedings of the 25th international conference on world wide web}"
    ",\n"
    "  pages={507--517},\n"
    "  year={2016}\n"
    "}"
)
AMAZON_2014_HOMEPAGE = "http://jmcauley.ucsd.edu/data/amazon/index_2014.html"

RAW_AMAZON_2014_DIR = Path("~/Datasets/raw-data/amazon2014-5").expanduser()
AMAZON_2014_DIR = Path("~/Datasets/preprocessed-data/amazon2014-5").expanduser()
AMAZON_2014_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS: dict[str, tuple[int, int]] = {
    "amazon-instant-video": (37126, 1685),
    "apps-for-android": (752937, 13209),
    "automotive": (20473, 1835),
    "baby": (160792, 7050),
    "beauty": (198502, 12101),
    "books": (8898041, 367982),
    "cds-and-vinyl": (1097592, 64443),
    "cell-phones-and-accessories": (194439, 10429),
    "clothing-shoes-and-jewelry": (278677, 23033),
    "digital-music": (64706, 3568),
    "electronics": (1689188, 63001),
    "grocery-and-gourmet-food": (151254, 8713),
    "health-and-personal-care": (346355, 18534),
    "home-and-kitchen": (551682, 28237),
    "kindle-store": (982619, 61934),
    "movies-and-tv": (1697533, 50052),
    "musical-instruments": (10261, 900),
    "office-products": (53258, 2420),
    "patio-lawn-and-garden": (13272, 962),
    "pet-supplies": (157836, 8510),
    "sports-and-outdoors": (296337, 18357),
    "tools-and-home-improvement": (134476, 10217),
    "toys-and-games": (167597, 11924),
    "video-games": (231780, 10672),
}

BLANK_PATTERN = re.compile(r"\s+")


def clean_text(item: str | None) -> str:
    if isinstance(item, str):
        return re.sub(BLANK_PATTERN, " ", HTMLParser(item).text()).strip()
    return ""


def clean_image_url(image_filenames: set[str], url: str | None) -> str:
    if isinstance(url, str) and len(url) > 0:
        if url.rsplit("/", 1)[-1] in image_filenames:
            return url
    return ""


def process_interaction_frame(dataframe: pd.DataFrame) -> pd.DataFrame | Exception:
    try:
        dataframe[UID] = dataframe[UID].str.strip()
        dataframe[IID] = dataframe[IID].str.strip()
        dataframe[LABEL] = dataframe[LABEL].astype(np.int8).clip(1, 5)
        dataframe["summary_review"] = dataframe["summary_review"].map(clean_text)
        dataframe["num_helpful_votes"] = dataframe["helpful"].map(
            operator.itemgetter(0)
        )
        dataframe["num_total_votes"] = dataframe["helpful"].map(operator.itemgetter(1))
        return dataframe
    except Exception as e:
        return e


def create_interaction_frame(
    interaction_file: Path, num_interactions: int | None = None
) -> Frame:
    interaction_frame: pd.DataFrame = load_from_dictline(
        interaction_file,
        num_items=num_interactions,
        usecols=[
            "reviewerID",
            "asin",
            "overall",
            "unixReviewTime",
            "summary",
            "helpful",
        ],
    )
    interaction_frame.columns = np.array(
        [UID, IID, LABEL, TIMESTAMP, "summary_review", "helpful"]
    )
    gc.collect()

    interaction_frame = parallelize(interaction_frame, process_interaction_frame)

    interaction_frame = (
        interaction_frame[
            [
                UID,
                IID,
                LABEL,
                TIMESTAMP,
                "summary_review",
                "num_helpful_votes",
                "num_total_votes",
            ]
        ]
        .sort_values(by=TIMESTAMP, kind="stable", ignore_index=True)
        .drop_duplicates(subset=[UID, IID], keep="first", ignore_index=True)
    )

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            IID: ftp.category(),
            LABEL: ftp.int_(),
            TIMESTAMP: ftp.int_(),
            "summary_review": ftp.text(),
            "num_helpful_votes": ftp.int_(),
            "num_total_votes": ftp.int_(),
        },
        interaction_frame,
    )


def process_item_frame(
    image_filenames: set[str], dataframe: pd.DataFrame
) -> pd.DataFrame | Exception:
    _clean_image_url = partial(clean_image_url, image_filenames)
    try:
        dataframe[IID] = dataframe[IID].str.strip()
        dataframe["title"] = dataframe["title"].map(clean_text)
        dataframe["brand"] = dataframe["brand"].map(clean_text)
        dataframe["description"] = dataframe["description"].map(clean_text)
        dataframe["image_url"] = dataframe["image"].map(_clean_image_url)
        dataframe["image_filename"] = dataframe["image_url"].map(
            lambda url: url.rsplit("/", 1)[-1]
        )
        del dataframe["image"]
        return dataframe
    except Exception as e:
        return e


def process_item_image(dataframe: pd.DataFrame) -> pd.DataFrame | Exception:
    try:
        dataframe["image"] = dataframe["image"].map(convert_image_to_jpeg)
        return dataframe
    except Exception as e:
        return e


def create_item_frame(
    item_file: Path,
    image_file: Path,
    interaction_frame: Frame,
    num_items: int | None = None,
) -> Frame:
    item_context_frame: pd.DataFrame = load_from_dictline(
        item_file,
        num_items=num_items,
        usecols=["asin", "title", "brand", "description", "imUrl"],
    )
    item_context_frame.columns = np.array(
        [IID, "title", "brand", "description", "image"]
    )
    item_frame = pd.DataFrame({IID: np.unique(interaction_frame[IID].value)})
    item_frame = item_frame.merge(item_context_frame, on=IID, how="left")
    del item_context_frame
    gc.collect()

    image_table = feather.read_table(image_file, columns=["filename"])
    image_filenames = set(image_table.column("filename").to_numpy())
    _process_item_frame = partial(process_item_frame, image_filenames)
    item_frame = parallelize(item_frame, _process_item_frame)
    del image_filenames, image_table

    image_table = feather.read_table(image_file)
    images: dict[str, bytes] = defaultdict(
        bytes,
        zip(
            image_table.column("filename").to_numpy(),
            image_table.column("content").to_numpy(),
        ),
    )
    item_frame["image"] = item_frame["image_filename"].map(images.__getitem__)
    del item_frame["image_filename"], images, image_table
    gc.collect()

    item_frame = parallelize(item_frame, process_item_image)

    item_frame = (
        item_frame[[IID, "title", "brand", "description", "image", "image_url"]]
        .sort_values(by=IID, kind="stable", ignore_index=True)
        .drop_duplicates(subset=[IID], keep="first", ignore_index=True)
    )

    return convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "title": ftp.text(),
            "brand": ftp.text(),
            "description": ftp.text(),
            "image": ftp.image(),
            "image_url": ftp.text(),
        },
        item_frame,
    )


def preprocess(
    input_dir: str | PathLike,
    output_dir: str | PathLike,
    num_interactions: int | None = None,
    num_items: int | None = None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_frame = create_interaction_frame(
        next(input_dir.glob("reviews_*.json")), num_interactions
    )
    item_frame = create_item_frame(
        next(input_dir.glob("meta_*.json")),
        next(input_dir.glob("images_*")),
        interaction_frame,
        num_items,
    )

    data, category_info = create_data(interaction_frame, None, item_frame)

    print(data)
    data.info()
    data.to_feather(output_dir)

    del data
    gc.collect()

    data_info = DataInfo.from_data_files(
        output_dir,
        description=AMAZON_2014_DESCRIPTION,
        citation=AMAZON_2014_CITATION,
        homepage=AMAZON_2014_HOMEPAGE,
    )
    data_info.to_json(output_dir)

    category_info.to_json(output_dir)

    gc.collect()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    for subset in SUBSETS:
        preprocess(
            RAW_AMAZON_2014_DIR / subset,
            AMAZON_2014_DIR / subset,
            SUBSETS[subset][0],
            SUBSETS[subset][1],
        )
