from typing import Optional, cast

import lightning as L
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, DataCollatorWithPadding, LlamaTokenizer  # type: ignore

from src.go_ast_tokenizer.tokenizer import GoASTTokenizer
from src.go_ast_tokenizer.utils import get_tokenizer

TOKENIZER_MODEL_ID = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "aholovko/go-critic-style"
MAX_LENGTH = 4096
NUM_LABELS = 8


class GoCriticStyleDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
        tokenizer: LlamaTokenizer,
        max_length: int,
    ) -> None:
        super().__init__()

        self._go_ast_tokenizer: Optional[GoASTTokenizer] = None

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

        # lazy init per worker
        if self._go_ast_tokenizer is None:
            self._go_ast_tokenizer = GoASTTokenizer()

        tokenized_code = self._go_ast_tokenizer.tokenize(raw["code"])

        encoding: BatchEncoding = self.tokenizer(
            tokenized_code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = cast(torch.Tensor, encoding["input_ids"]).squeeze(0)
        attention_mask = cast(torch.Tensor, encoding["attention_mask"]).squeeze(0).to(torch.float)

        # one‑hot encode labels
        indices = torch.as_tensor(raw["labels"], dtype=torch.long)
        labels = F.one_hot(indices, num_classes=self.num_classes).sum(dim=0).to(torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class GoCriticStyleDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = get_tokenizer(TOKENIZER_MODEL_ID)
        self.dataset_name = DATASET_NAME
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = MAX_LENGTH

        self.train_dataset: Optional[GoCriticStyleDataset] = None
        self.val_dataset: Optional[GoCriticStyleDataset] = None
        self.test_dataset: Optional[GoCriticStyleDataset] = None

        self.collator = DataCollatorWithPadding(self.tokenizer, padding="longest")

    def prepare_data(self) -> None:
        load_dataset(self.dataset_name)

    def setup(self, stage: Optional[str] = None) -> None:
        raw = load_dataset(self.dataset_name)
        ds = cast(DatasetDict, raw)

        self.train_dataset = GoCriticStyleDataset(ds["train"], self.tokenizer, MAX_LENGTH)
        self.val_dataset = GoCriticStyleDataset(ds["validation"], self.tokenizer, MAX_LENGTH)
        self.test_dataset = GoCriticStyleDataset(ds["test"], self.tokenizer, MAX_LENGTH)

    def train_dataloader(self) -> DataLoader:
        ds = self.train_dataset
        assert ds is not None, "setup() must be called before train_dataloader()"
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        ds = self.val_dataset
        assert ds is not None, "setup() must be called before val_dataloader()"
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collator,
        )
