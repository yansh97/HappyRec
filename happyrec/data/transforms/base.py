from collections.abc import Sequence

from ..data import Data


class DataTransform:
    """The base class of all data transforms."""

    def __call__(self, data: Data) -> Data:
        """Check the data, transform the data and validate the transformed data.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        self.check(data)
        return self.transform(data).validate()

    def check(self, data: Data) -> None:
        """Check the data.

        :param data: The data to be checked.
        """
        pass

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

    def __call__(self, data: Data) -> Data:
        """Check the data, transform the data and validate the transformed data.

        :param data: The data to be transformed.
        :return: The transformed data.
        """
        for transform in self.transforms:
            transform.check(data)
            data = transform.transform(data)
        return data.validate()
