from dataclasses import dataclass
from typing import Optional, cast

import lightning as L
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, DataCollatorWithPadding, LlamaTokenizer  # type: ignore

SEED = 2357


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "aholovko/go-critic-style"
    max_length: int = 4096
    batch_size: int = 8
    num_workers: int = 4


class GoCriticStyleDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
        tokenizer: LlamaTokenizer,
        max_length: int,
    ) -> None:
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        label_feature = hf_dataset.features["labels"].feature
        self.num_classes = label_feature.num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hf = cast(HFDataset, self.dataset)
        raw = hf[idx]

        encoding: BatchEncoding = self.tokenizer(
            raw["code"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = cast(torch.Tensor, encoding["input_ids"]).squeeze(0)
        attention_mask = cast(torch.Tensor, encoding["attention_mask"]).squeeze(0).to(torch.float)

        # one‑hot encode labels
        idxs = torch.as_tensor(raw["labels"], dtype=torch.long)
        labels = F.one_hot(idxs, num_classes=self.num_classes).sum(dim=0).to(torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class GoCriticStyleDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: LlamaTokenizer,
        config: DataConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("config")
        self.tokenizer = tokenizer
        self.config = config

        self.train_dataset: Optional[GoCriticStyleDataset] = None
        self.val_dataset: Optional[GoCriticStyleDataset] = None
        self.test_dataset: Optional[GoCriticStyleDataset] = None

        self.collator = DataCollatorWithPadding(tokenizer, padding="longest")

    def prepare_data(self) -> None:
        load_dataset(self.config.dataset_name)

    def setup(self, stage: Optional[str] = None) -> None:
        raw = load_dataset(self.config.dataset_name)
        ds = cast(DatasetDict, raw)

        self.train_dataset = GoCriticStyleDataset(ds["train"], self.tokenizer, self.config.max_length)
        self.val_dataset = GoCriticStyleDataset(ds["validation"], self.tokenizer, self.config.max_length)
        self.test_dataset = GoCriticStyleDataset(ds["test"], self.tokenizer, self.config.max_length)

    def train_dataloader(self) -> DataLoader:
        ds = self.train_dataset
        assert ds is not None, "setup() must be called before train_dataloader()"
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collator,
            generator=torch.Generator().manual_seed(SEED),
        )

    def val_dataloader(self) -> DataLoader:
        ds = self.val_dataset
        assert ds is not None, "setup() must be called before val_dataloader()"
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        ds = self.test_dataset
        assert ds is not None, "setup() must be called before test_dataloader()"
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collator,
        )
