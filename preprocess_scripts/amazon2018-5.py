import gc
import logging
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
from happyrec.data import ftypes as ftp
from happyrec.utils.asserts import is_typed_list
from happyrec.utils.logger import logger
from happyrec.utils.preprocessing import (
    convert_dataframe_to_frame,
    convert_image_to_jpeg,
    create_data,
    load_from_jsonline,
    parallelize,
)

AMAZON_2018_DESCRIPTION = (
    "Amazon Review Data includes reviews (ratings and text) and product metadata "
    "(descriptions, category information, brand, and other features). This version of "
    "the dataset is released in 2018. Note that this dataset have been filtered by "
    "5-core files on the homepage."
)
AMAZON_2018_CITATION = (
    "@inproceedings{ni2019justifying,\n"
    "  title={Justifying recommendations using distantly-labeled reviews and "
    "fine-grained aspects},\n"
    "  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},\n"
    "  booktitle={Proceedings of the 2019 conference on empirical methods in natural "
    "language processing and the 9th international joint conference on natural "
    "language processing (EMNLP-IJCNLP)},\n"
    "  pages={188--197},\n"
    "  year={2019}\n"
    "}"
)
AMAZON_2018_HOMEPAGE = "https://nijianmo.github.io/amazon/index.html"


RAW_AMAZON_2018_DIR = Path("~/Datasets/raw-data/amazon2018-5").expanduser()
AMAZON_2018_DIR = Path("~/Datasets/preprocessed-data/amazon2018-5").expanduser()
AMAZON_2018_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS: dict[str, tuple[int, int]] = {
    "all-beauty": (5269, 85),
    "amazon-fashion": (3176, 31),
    "appliances": (2277, 47),
    "arts-crafts-and-sewing": (494485, 22855),
    "automotive": (1711519, 79317),
    "books": (27164983, 704023),
    "cds-and-vinyl": (1443755, 67602),
    "cell-phones-and-accessories": (1128437, 48172),
    "clothing-shoes-and-jewelry": (11285464, 376858),
    "digital-music": (169781, 185),
    "electronics": (6739590, 159935),
    "gift-cards": (2972, 148),
    "grocery-and-gourmet-food": (1143860, 41280),
    "home-and-kitchen": (6898955, 189040),
    "industrial-and-scientific": (77071, 5327),
    "kindle-store": (2222983, 98382),
    "luxury-beauty": (34278, 1577),
    "magazine-subscriptions": (2375, 151),
    "movies-and-tv": (3410019, 60110),
    "musical-instruments": (231392, 10611),
    "office-products": (800357, 27932),
    "patio-lawn-and-garden": (798415, 32869),
    "pet-supplies": (2098325, 42498),
    "prime-pantry": (137788, 4968),
    "software": (12805, 802),
    "sports-and-outdoors": (2839940, 104566),
    "tools-and-home-improvement": (2070831, 73548),
    "toys-and-games": (1828971, 78698),
    "video-games": (497577, 17389),
}

BLANK_PATTERN = re.compile(r"\s+")


def clean_text(item: str | None) -> str:
    if isinstance(item, str):
        return re.sub(BLANK_PATTERN, " ", HTMLParser(item).text()).strip()
    return ""


def clean_text_list(item: str | list[str] | None) -> str:
    if is_typed_list(item, str):
        item = ". ".join([i.strip() for i in item])
    if isinstance(item, str):
        item = re.sub(BLANK_PATTERN, " ", HTMLParser(item).text()).strip()
    return ""


def clean_image_url(image_filenames: set[str], item: str | list[str] | None) -> str:
    if isinstance(item, str):
        item = [item]
    if is_typed_list(item, str) and len(item) > 0:
        for url in item:
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
        return dataframe
    except Exception as e:
        return e


def create_interaction_frame(
    interaction_file: Path, num_interactions: int | None = None
) -> Frame:
    interaction_frame: pd.DataFrame = load_from_jsonline(
        interaction_file,
        num_items=num_interactions,
        usecols=["reviewerID", "asin", "overall", "unixReviewTime", "summary"],
    )
    interaction_frame.columns = np.array([UID, IID, LABEL, TIMESTAMP, "summary_review"])
    gc.collect()

    interaction_frame = parallelize(interaction_frame, process_interaction_frame)

    interaction_frame = (
        interaction_frame[[UID, IID, LABEL, TIMESTAMP, "summary_review"]]
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
        dataframe["description"] = dataframe["description"].map(clean_text_list)
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
    item_context_frame: pd.DataFrame = load_from_jsonline(
        item_file,
        num_items=num_items,
        usecols=["asin", "title", "brand", "description", "imageURLHighRes"],
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
        description=AMAZON_2018_DESCRIPTION,
        citation=AMAZON_2018_CITATION,
        homepage=AMAZON_2018_HOMEPAGE,
    )
    data_info.to_json(output_dir)

    category_info.to_json(output_dir)

    gc.collect()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    for subset in SUBSETS:
        preprocess(
            RAW_AMAZON_2018_DIR / subset,
            AMAZON_2018_DIR / subset,
            SUBSETS[subset][0],
            SUBSETS[subset][1],
        )
