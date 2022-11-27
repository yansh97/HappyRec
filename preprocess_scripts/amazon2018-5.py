import gc
import logging
import operator
import re
from functools import partial
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from selectolax.parser import HTMLParser

from happyrec.data import DataInfo, Frame, ImageType, TextType
from happyrec.data.predefined_fields import FTYPES, IID, LABEL, TIMESTAMP, UID
from happyrec.utils.data import (
    convert_dataframe_to_frame,
    convert_image_to_array,
    create_data,
    load_from_jsonline,
    parallelize,
)
from happyrec.utils.logger import logger

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


def clean_text(item: str | None) -> str | None:
    text = None
    if isinstance(item, str):
        item = re.sub(BLANK_PATTERN, " ", HTMLParser(item).text()).strip()
        if len(item) > 0:
            text = item
    return text


def clean_text_list(item: str | list[str] | None) -> str | None:
    text = None
    if isinstance(item, list):
        item = ". ".join(item)
    if isinstance(item, str):
        item = re.sub(BLANK_PATTERN, " ", HTMLParser(item).text()).strip()
        if len(item) > 0:
            text = item
    return text


def read_image(
    image_dir: Path, item: str | list[str] | None
) -> tuple[np.ndarray | None, str | None]:
    if isinstance(item, str):
        item = [item]
    if isinstance(item, list) and len(item) > 0:
        for url in item:
            if isinstance(url, str) and len(url) > 0:
                path = image_dir / url.rsplit("/", 1)[-1]
                if path.exists():
                    return convert_image_to_array(path), url
    return None, None


def process_interaction_frame(dataframe: pd.DataFrame) -> pd.DataFrame | Exception:
    try:
        dataframe[UID] = dataframe[UID].str.strip()
        dataframe[IID] = dataframe[IID].str.strip()
        dataframe[LABEL] = dataframe[LABEL].astype(np.float16).clip(1, 5)
        dataframe["summary_review"] = dataframe["summary_review"].apply(clean_text)  # type: ignore # noqa: E501
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
            UID: FTYPES[UID],
            IID: FTYPES[IID],
            LABEL: FTYPES[LABEL],
            TIMESTAMP: FTYPES[TIMESTAMP],
            "summary_review": TextType(),
        },
        interaction_frame,
    )


def process_item_frame(
    image_dir: Path, dataframe: pd.DataFrame
) -> pd.DataFrame | Exception:
    _read_image = partial(read_image, image_dir)
    try:
        dataframe[IID] = dataframe[IID].str.strip()
        dataframe["title"] = dataframe["title"].apply(clean_text)  # type: ignore
        dataframe["brand"] = dataframe["brand"].apply(clean_text)  # type: ignore
        dataframe["description"] = dataframe["description"].apply(clean_text_list)  # type: ignore # noqa: E501
        images = dataframe["image"].apply(_read_image)  # type: ignore
        dataframe["image"] = images.apply(operator.itemgetter(0))
        dataframe["image_url"] = images.apply(operator.itemgetter(1))
        return dataframe
    except Exception as e:
        return e


def create_item_frame(
    item_file: Path,
    interaction_frame: Frame,
    image_dir: Path,
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

    _process_item_frame = partial(process_item_frame, image_dir)
    item_frame = parallelize(item_frame, _process_item_frame)

    item_frame = item_frame[
        [IID, "title", "brand", "description", "image", "image_url"]
    ]
    item_frame = item_frame.sort_values(by=IID, kind="stable", ignore_index=True)
    item_frame = item_frame.drop_duplicates(
        subset=[IID], keep="first", ignore_index=True
    )

    return convert_dataframe_to_frame(
        {
            IID: FTYPES[IID],
            "title": TextType(),
            "brand": TextType(),
            "description": TextType(),
            "image": ImageType(),
            "image_url": TextType(),
        },
        item_frame,
    )


def preprocess(
    input_dir: str | PathLike,
    output_dir: str | PathLike,
    num_interactions: int | None = None,
    num_items: int | None = None,
) -> None:
    """Preprocess the Amazon2018 dataset.

    :param input_dir: The directory containing the raw data.
    :param output_dir: The directory to save the preprocessed data.
    """
    input_dir = Path(input_dir)
    image_dir = next(input_dir.glob("images_*"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_frame = create_interaction_frame(
        next(input_dir.glob("reviews_*.json")), num_interactions
    )
    item_frame = create_item_frame(
        next(input_dir.glob("meta_*.json")), interaction_frame, image_dir, num_items
    )

    data = create_data(interaction_frame, None, item_frame)

    print(data)
    data.info()
    data.to_npz(output_dir)

    data_info = DataInfo.from_data_files(
        output_dir,
        description=AMAZON_2018_DESCRIPTION,
        citation=AMAZON_2018_CITATION,
        homepage=AMAZON_2018_HOMEPAGE,
    )
    print(data_info)
    data_info.to_json(output_dir)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    for subset in SUBSETS:
        preprocess(
            RAW_AMAZON_2018_DIR / subset,
            AMAZON_2018_DIR / subset,
            SUBSETS[subset][0],
            SUBSETS[subset][1],
        )
