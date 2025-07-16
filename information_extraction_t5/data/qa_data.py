"""Implement DataModule"""
import os
from typing import Optional
import configargparse

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers.data.processors.squad import SquadV1Processor

from information_extraction_t5.data.convert_squad_to_t5 import squad_convert_examples_to_t5_format

class QADataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

    def setup(self, stage: Optional[str] = None):
        # 1. 确定并创建数据特征缓存目录
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else "./.cache"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"INFO: Using data feature cache directory: {cache_dir}")
        model_name_suffix = list(filter(None, self.hparams.model_name_or_path.split('/'))).pop()

        # 2. 准备训练和验证数据集
        if stage == 'fit' or stage is None:
            cached_train_file = os.path.join(cache_dir, f"cached_train_{model_name_suffix}")
            cached_valid_file = os.path.join(cache_dir, f"cached_valid_{model_name_suffix}")

            if os.path.exists(cached_train_file) and os.path.exists(
                    cached_valid_file) and not self.hparams.overwrite_cache:
                print(f"Loading features from cached files in {cache_dir}")
                self.train_dataset = torch.load(cached_train_file)["dataset"]
                self.valid_dataset = torch.load(cached_valid_file)["dataset"]
            else:
                print("Creating features from dataset files...")
                processor = SquadV1Processor()
                examples_train = processor.get_dev_examples(None, filename=self.hparams.train_file)
                examples_valid = processor.get_dev_examples(None, filename=self.hparams.valid_file)

                _, _, self.train_dataset = squad_convert_examples_to_t5_format(examples=examples_train,
                                                                               use_sentence_id=self.hparams.use_sentence_id,
                                                                               evaluate=False,
                                                                               negative_ratio=self.hparams.negative_ratio,
                                                                               return_dataset=True)
                _, _, self.valid_dataset = squad_convert_examples_to_t5_format(examples=examples_valid,
                                                                               use_sentence_id=self.hparams.use_sentence_id,
                                                                               evaluate=True, negative_ratio=0,
                                                                               return_dataset=True)

                print(f"Saving features into cached file {cached_train_file}")
                torch.save({"dataset": self.train_dataset}, cached_train_file)
                print(f"Saving features into cached file {cached_valid_file}")
                torch.save({"dataset": self.valid_dataset}, cached_valid_file)

            print(f'>> train-dataset: {len(self.train_dataset)} samples')
            print(f'>> valid-dataset: {len(self.valid_dataset)} samples')

        # 3. 准备测试数据集
        if stage == 'test' or stage is None:
            assert self.hparams.test_file, 'test_file must be specificed'
            cached_test_file = os.path.join(cache_dir, f"cached_test_{model_name_suffix}")

            if os.path.exists(cached_test_file) and not self.hparams.overwrite_cache:
                print(f"Loading features from cached file {cached_test_file}")
                self.test_dataset = torch.load(cached_test_file)["dataset"]
            else:
                print("Creating features from dataset file...")
                processor = SquadV1Processor()
                examples_test = processor.get_dev_examples(None, filename=self.hparams.test_file)
                _, _, self.test_dataset = squad_convert_examples_to_t5_format(examples=examples_test,
                                                                              use_sentence_id=self.hparams.use_sentence_id,
                                                                              evaluate=True, negative_ratio=0,
                                                                              return_dataset=True)
                print(f"Saving features into cached file {cached_test_file}")
                torch.save({"dataset": self.test_dataset}, cached_test_file)

            print(f'>> test-dataset: {len(self.test_dataset)} samples')

    def get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_dataset, batch_size=self.hparams.train_batch_size,
                                   shuffle=self.hparams.shuffle_train, num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.valid_dataset, batch_size=self.hparams.val_batch_size, shuffle=False,
                                   num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset, batch_size=self.hparams.val_batch_size, shuffle=False,
                                   num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = configargparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--cache_dir", default=None, type=str,
                            help="Path to directory to store the cached datasets.")
        parser.add_argument("--data_dir", default=None, type=str,
                            help="The input data dir. Should contain the .json files for the task.")
        parser.add_argument("--train_file", default=None, type=str, help="The input training file.")
        parser.add_argument("--valid_file", default=None, type=str, help="The input evaluation file.")
        parser.add_argument("--test_file", default=None, type=str, help="The input test file.")
        parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--shuffle_train", action="store_true", help="Shuffle the train dataset")
        parser.add_argument("--negative_ratio", default=0, type=int,
                            help="Set the positive-negative ratio of the training dataset.")
        parser.add_argument("--use_sentence_id", action="store_true",
                            help="Set this flag if you are using the approach that breaks the contexts into sentences")
        parser.add_argument("--overwrite_cache", action="store_true",
                            help="Overwrite the cached training and evaluation sets")
        return parser