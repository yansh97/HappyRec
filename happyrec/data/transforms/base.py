from collections.abc import Sequence

from ..data import Data


class DataTransform:
    """The base class of all data transforms."""

    def __call__(self, data: Data) -> Data:
        """Transform the data and validate the transformed data.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        if data.is_splitted:
            raise ValueError("The data is already splitted.")
        if data.has_eval_negative_samples:
            raise ValueError("The data has already generated negative samples.")
        data = self.transform(data)
        data = data.validate()
        return data

    def transform(self, data: Data) -> Data:
        """Transform the data.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        raise NotImplementedError


class Compose(DataTransform):
    """Compose multiple data transforms together."""

    def __init__(self, transforms: Sequence[DataTransform]) -> None:
        """Initialize the Compose data transform.

        :param transforms: The transforms to compose.
        """
        self.transforms = list(transforms)

    def transform(self, data: Data) -> Data:
        """Transform the data.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        for transform in self.transforms:
            data = transform.transform(data)
        return data
