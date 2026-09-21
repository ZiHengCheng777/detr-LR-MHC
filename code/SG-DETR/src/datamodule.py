"""Module defines the DataModule class, which is used to prepare and load data for the MomentRetrieval dataset."""

from typing import Literal, Type

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.dataset.collate import custom_collate
from src.dataset.qvhighlights import QVHighlights as MomentRetrievalDataset
from src.utils.basic_utils import guess_num_workers

Segment = Literal["train", "val", "test"]
VAL_BATCH_SIZE: int = 512


class MomentRetrievalDataModule(LightningDataModule):  # noqa: WPS230
    """
    A PyTorch Lightning DataModule for the MomentRetrieval dataset.

    This class provides methods for preparing the data, creating datasets, and
    defining data loaders for training, validation, and testing.
    """

    # pylint: disable=unused-argument
    def __init__(  # noqa: WPS211
        self,
        dataset: Type[MomentRetrievalDataset],
        batch_size: int,
        annotation_path_train: str,
        annotation_path_val: str,
        annotation_path_test: str,
        query_feat_dir_train: str,
        query_feat_dir_val: str,
        query_feat_dir_test: str,
        video_feat_dir_train: str,
        video_feat_dir_val: str,
        video_feat_dir_test: str,
        audio_feat_dir_train: str,
        audio_feat_dir_val: str,
        audio_feat_dir_test: str,
        num_workers: int,
        val_batch_size: int = VAL_BATCH_SIZE,
    ) -> None:
        """
        Initialize the MomentRetrievalDataModule.

        Args:
            dataset (Type[MomentRetrievalDataset]): The dataset class to use.
            batch_size (int): The batch size to use for the data loaders.
            annotation_path_train (str): Path to train annotation file dir.
            annotation_path_val (str): Path to val annotation file dir.
            annotation_path_test (str): Path to test annotation file dir.
            query_feat_dir_train (str): Path to train caption embeddings file dir.
            query_feat_dir_val (str): Path to val caption embeddings file dir.
            query_feat_dir_test (str): Path to test caption embeddings file dir.
            video_feat_dir_train (str): Path to train video embeddings file dir.
            video_feat_dir_val (str): Path to val video embeddings file dir.
            video_feat_dir_test (str): Path to test video embeddings file dir.
            audio_feat_dir_train (str): Path to train audio embeddings file dir.
            audio_feat_dir_val (str): Path to val audio embeddings file dir.
            audio_feat_dir_test (str): Path to test audio embeddings file dir.
            num_workers (int): The number of workers to use for the data loaders.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        # annotation files
        self.annotation_path_train = annotation_path_train
        self.annotation_path_val = annotation_path_val
        self.annotation_path_test = annotation_path_test

        # query dirs
        self.query_feat_dir_train = query_feat_dir_train
        self.query_feat_dir_val = query_feat_dir_val
        self.query_feat_dir_test = query_feat_dir_test

        # video features dirs
        self.video_feat_dir_train = video_feat_dir_train
        self.video_feat_dir_val = video_feat_dir_val
        self.video_feat_dir_test = video_feat_dir_test

        # audio features dirs
        self.audio_feat_dir_train = audio_feat_dir_train
        self.audio_feat_dir_val = audio_feat_dir_val
        self.audio_feat_dir_test = audio_feat_dir_test

    def get_dataloader(self, segment: Segment) -> DataLoader[MomentRetrievalDataset]:
        """
        Get the data loader for the given segment.

        Args:
            segment (Segment): Which segment to fetch data loader for (train, val, or test).

        Returns:
            DataLoader: DataLoader object for the given segment.
        """
        num_workers = self.num_workers if self.num_workers is not None else guess_num_workers()

        if segment == "train":
            anno_path = self.annotation_path_train
            query_feat_dir = self.query_feat_dir_train
            video_feat_dir = self.video_feat_dir_train
            audio_feat_dir = self.audio_feat_dir_train
        elif segment == "val":
            anno_path = self.annotation_path_val
            query_feat_dir = self.query_feat_dir_val
            video_feat_dir = self.video_feat_dir_val
            audio_feat_dir = self.audio_feat_dir_val
        else:
            anno_path = self.annotation_path_test
            query_feat_dir = self.query_feat_dir_test
            video_feat_dir = self.video_feat_dir_test
            audio_feat_dir = self.audio_feat_dir_test

        dataset = self.dataset(
            data_path=anno_path,
            video_feat_dir=video_feat_dir,
            query_feat_dir=query_feat_dir,
            audio_feat_dir=audio_feat_dir,
        )
        # There are losses that use outputs from different batch objects. Therefore, for the train, if the last batch
        # has a length of 1, it will be deleted
        dataset_length = len(dataset)
        if segment == "train" and (dataset_length % self.batch_size == 1):
            drop_last = True
        else:
            drop_last = False

        worker_options = {}
        if num_workers > 0:
            # Forking after CUDA/model initialization copies the parent address
            # space into every worker. Spawn keeps feature loaders CPU-only.
            worker_options = {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "prefetch_factor": 2,
            }

        return DataLoader(
            dataset=dataset,
            shuffle=(segment == "train"),
            batch_size=self.batch_size if segment == "train" else self.val_batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
            collate_fn=custom_collate,
            **worker_options,
        )

    def train_dataloader(self) -> DataLoader[MomentRetrievalDataset]:
        """
        Get the data loader for the training set.

        Returns:
            DataLoader[MomentRetrievalDataset]: Data loader for the training set.
        """
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader[MomentRetrievalDataset]:
        """
        Get the data loader for the validation set.

        Returns:
            DataLoader[MomentRetrievalDataset]: Data loader for the val set.
        """
        return self.get_dataloader("val")

    def test_dataloader(self) -> DataLoader[MomentRetrievalDataset]:
        """
        Get the data loader for the testing set.

        Returns:
            DataLoader[MomentRetrievalDataset]: Data loader for the testing set.
        """
        return self.get_dataloader("test")
