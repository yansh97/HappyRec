import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from happyrec.data import DataInfo, Frame
from happyrec.data import field_types as ftp
from happyrec.data.fields import IID, LABEL, TIMESTAMP, UID
from happyrec.utils.logger import logger
from happyrec.utils.preprocessing import convert_dataframe_to_frame, create_data

ZHIHUREC_DESCRIPTION = (
    "ZhihuRec dataset is collected from a knowledge-sharing platform (Zhihu), which is "
    "composed of interactions collected within 10 days. There are also descriptions of "
    "users, answers, which are anonymous."
    "Note: The dataset contains duplicated user-item interactions with different "
    "timestamps."
)
ZHIHUREC_CITATION = (
    "@article{hao2021large,\n"
    "  title={A Large-Scale Rich Context Query and Recommendation Dataset in Online "
    "Knowledge-Sharing},\n"
    "  author={Hao, Bin and Zhang, Min and Ma, Weizhi and Shi, Shaoyun and Yu, Xinxing "
    "and Shan, Houzhi and Liu, Yiqun and Ma, Shaoping},\n"
    "  journal={arXiv preprint arXiv:2106.06467},\n"
    "  year={2021}\n"
    "}"
)
ZHIHUREC_HOMEPAGE = "https://github.com/THUIR/ZhihuRec-Dataset/"

RAW_ZHIHUREC_DIR = Path("~/Datasets/raw-data/zhihurec").expanduser()
ZHIHUREC_DIR = Path("~/Datasets/preprocessed-data/zhihurec").expanduser()
ZHIHUREC_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = {
    "1m": {"interaction": 999970, "user": 7974, "item": 81563},
    "20m": {"interaction": 19999857, "user": 159642, "item": 343103},
    "100m": {"interaction": 99978523, "user": 798086, "item": 554976},
}


def create_interaction_frame(interaction_file: Path, nrows: int) -> Frame:
    interaction_frame = pd.read_csv(interaction_file, sep=",", header=None, nrows=nrows)
    interaction_frame.columns = np.array([UID, IID, TIMESTAMP, LABEL])
    interaction_frame = interaction_frame[[UID, IID, LABEL, TIMESTAMP]]

    interaction_frame[LABEL][interaction_frame[LABEL] > 0] = 1
    interaction_frame[LABEL][interaction_frame[LABEL] <= 0] = 0

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


def create_user_frame(user_file: Path, nrows: int) -> Frame:
    user_frame = pd.read_csv(
        user_file,
        sep=",",
        header=None,
        nrows=nrows,
        usecols=list(range(26)),
    )
    user_frame.columns = np.array(
        [
            UID,
            "register_timestamp",
            "gender",
            "login_frequency",
            "num_followers",
            "num_followed_topics",
            "num_followed_questions",
            "num_created_answers",
            "num_created_questions",
            "num_created_comments",
            "num_received_thanks",
            "num_received_comments",
            "num_received_likes",
            "num_received_dislikes",
            "register_type",
            "regis_platfrom",
            "from_android",
            "from_iphone",
            "from_ipad",
            "from_pc",
            "from_mobile_web",
            "device_model",
            "device_brand",
            "platform",
            "province",
            "city",
        ]
    )

    user_frame = user_frame.sort_values(
        by=UID, kind="stable", ignore_index=True
    ).drop_duplicates(subset=UID, keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            UID: ftp.category(),
            "register_timestamp": ftp.int_(),
            "gender": ftp.category(),
            "login_frequency": ftp.category(),
            "num_followers": ftp.int_(),
            "num_followed_topics": ftp.int_(),
            "num_followed_questions": ftp.int_(),
            "num_created_answers": ftp.int_(),
            "num_created_questions": ftp.int_(),
            "num_created_comments": ftp.int_(),
            "num_received_thanks": ftp.int_(),
            "num_received_comments": ftp.int_(),
            "num_received_likes": ftp.int_(),
            "num_received_dislikes": ftp.int_(),
            "register_type": ftp.category(),
            "regis_platfrom": ftp.category(),
            "from_android": ftp.category(),
            "from_iphone": ftp.category(),
            "from_ipad": ftp.category(),
            "from_pc": ftp.category(),
            "from_mobile_web": ftp.category(),
            "device_model": ftp.category(),
            "device_brand": ftp.category(),
            "platform": ftp.category(),
            "province": ftp.category(),
            "city": ftp.category(),
        },
        user_frame,
    )


def create_item_frame(item_file: Path, nrows: int) -> Frame:
    item_frame = pd.read_csv(
        item_file,
        sep=",",
        header=None,
        nrows=nrows,
        usecols=list(range(16)),
    )
    item_frame.columns = np.array(
        [
            IID,
            "question_id",
            "is_anonymous",
            "author_id",
            "is_high_quality",
            "is_recommended",
            "create_timestamp",
            "contains_pics",
            "contains_videos",
            "num_thanks",
            "num_likes",
            "num_comments",
            "num_collections",
            "num_dislikes",
            "num_reports",
            "num_helpness",
        ]
    )

    item_frame["question_id"] = item_frame["question_id"].fillna(-1).astype(int)
    item_frame["author_id"] = item_frame["author_id"].fillna(-1).astype(int)

    item_frame = item_frame.sort_values(
        by=IID, kind="stable", ignore_index=True
    ).drop_duplicates(subset=IID, keep="first", ignore_index=True)

    return convert_dataframe_to_frame(
        {
            IID: ftp.category(),
            "question_id": ftp.category(),
            "is_anonymous": ftp.category(),
            "author_id": ftp.category(),
            "is_high_quality": ftp.category(),
            "is_recommended": ftp.category(),
            "create_timestamp": ftp.int_(),
            "contains_pics": ftp.category(),
            "contains_videos": ftp.category(),
            "num_thanks": ftp.int_(),
            "num_likes": ftp.int_(),
            "num_comments": ftp.int_(),
            "num_collections": ftp.int_(),
            "num_dislikes": ftp.int_(),
            "num_reports": ftp.int_(),
            "num_helpness": ftp.int_(),
        },
        item_frame,
    )


def preprocess(
    input_dir: str | PathLike,
    output_dir: str | PathLike,
    num_interactions: int,
    num_users: int,
    num_items: int,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    pd.set_option("display.max_columns", None)
    interaction_frame = create_interaction_frame(
        input_dir / "inter_impression.csv", num_interactions
    )
    user_frame = create_user_frame(input_dir / "info_user.csv", num_users)
    item_frame = create_item_frame(input_dir / "info_answer.csv", num_items)

    data = create_data(interaction_frame, user_frame, item_frame)

    print(data)
    data.info()
    data.to_pickle(output_dir)

    data_info = DataInfo.from_data_files(
        output_dir,
        description=ZHIHUREC_DESCRIPTION,
        citation=ZHIHUREC_CITATION,
        homepage=ZHIHUREC_HOMEPAGE,
    )
    data_info.to_json(output_dir)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    for subset in SUBSETS:
        preprocess(
            RAW_ZHIHUREC_DIR,
            ZHIHUREC_DIR / subset,
            SUBSETS[subset]["interaction"],
            SUBSETS[subset]["user"],
            SUBSETS[subset]["item"],
        )
