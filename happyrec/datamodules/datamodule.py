from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from ..models import Model


class DataModule(LightningDataModule):
    def __init__(
        self,
        datasets: tuple[Model.Dataset, Model.Dataset, Model.Dataset],
        batch_size: int = 128,
        eval_batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        sampler = RandomSampler(self.train_dataset)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=None,
            sampler=batch_sampler,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = SequentialSampler(self.val_dataset)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.eval_batch_size, drop_last=False
        )
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=None,
            sampler=batch_sampler,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = SequentialSampler(self.test_dataset)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.eval_batch_size, drop_last=False
        )
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=None,
            sampler=batch_sampler,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
