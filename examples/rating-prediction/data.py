import logging

from happyrec.data import Data, DataInfo
from happyrec.data.splitters import RatioSplitter
from happyrec.utils.logger import logger

logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    # check the data infomation
    data_info = DataInfo.from_json("datasets/movielens100k")
    print(data_info)
    data_info.info()

    # load the data
    data = Data.from_feather("datasets/movielens100k")
    print(data)
    data.info()

    # split into training, validation and test sets
    data = RatioSplitter(ratio=(0.8, 0.1, 0.1))(data)

    # save the data
    print(data)
    data.info()
    data.to_feather("examples/rating-prediction/data")

    # save the data information
    data_info = DataInfo.from_data_files("examples/rating-prediction/data")
    print(data_info)
    data_info.info()
    data_info.to_json("examples/rating-prediction/data")
