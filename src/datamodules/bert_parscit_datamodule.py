from typing import Optional, Tuple

import datasets
from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from src.datamodules.components.class_label import LABEL_LIST, label2id
from src.datamodules.components.preprocess import preprocess, tokenize_and_align_labels
from src.models.components.bert_tokenizer import bert_tokenizer

SYNTHETIC_DATASET_REPO = "myvision/yuanchuan-synthetic-dataset-final"

DATA_CACHE_DIR = "/data3/jiahe/synthetic-final/"


class BERTParsCitDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        data_repo: str = SYNTHETIC_DATASET_REPO,
        train_val_split: Tuple[int, int] = (50000, 5000),
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_cache_dir: str = DATA_CACHE_DIR,
        seed: int = 777
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_collator = DataCollatorForTokenClassification(tokenizer=bert_tokenizer)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return len(LABEL_LIST)

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            raw_trainset = datasets.load_dataset(
                self.hparams.data_repo,
                split="train",
                cache_dir=self.hparams.data_cache_dir
            )

            raw_testset = datasets.load_dataset(
                self.hparams.data_repo,
                split="test",
                cache_dir=self.hparams.data_cache_dir
            )

            shuffled_raw_trainset = raw_trainset.shuffle(seed=self.hparams.seed)
            selected_indices = list(range(sum(self.hparams.train_val_split)))
            selected_train_data = shuffled_raw_trainset.select(selected_indices[:self.hparams.train_val_split[0]])
            selected_val_data = shuffled_raw_trainset.select(selected_indices[self.hparams.train_val_split[0]:])

            dataset_dict = DatasetDict()
            dataset_dict['train'] = selected_train_data
            dataset_dict['val'] = selected_val_data
            dataset_dict['test'] = raw_testset

            processed_datasets = dataset_dict.map(
                preprocess,
                batched=True,
                remove_columns=dataset_dict["train"].column_names,
                load_from_cache_file=True
            )

            tokenized_datasets = processed_datasets.map(
                lambda x: tokenize_and_align_labels(x, label2id),
                batched=True,
                remove_columns=processed_datasets["train"].column_names,
                load_from_cache_file=True
            )   

            self.data_train = tokenized_datasets["train"]
            self.data_val = tokenized_datasets["val"]
            self.data_test = tokenized_datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
            shuffle=False,
        )


