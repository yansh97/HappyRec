import logging

from happyrec.data import Data, DataInfo
from happyrec.data.samplers import EvalNegativeSampler, RecSampler
from happyrec.data.splitters import LeaveOneOutSplitter
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
    data = LeaveOneOutSplitter()(data)

    # sample evaluation negative items for each user
    data = EvalNegativeSampler(size=100, sampler=RecSampler())(data)

    # save the data
    print(data)
    data.info()
    data.to_feather("examples/top-n-ranking/data")

    # save the data information
    data_info = DataInfo.from_data_files("examples/top-n-ranking/data")
    print(data_info)
    data_info.info()
    data_info.to_json("examples/top-n-ranking/data")
