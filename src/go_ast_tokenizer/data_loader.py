"""
Data loader.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import lightning as L
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

SEED = 2357
DATASET_ID = "yahma/alpaca-cleaned"


class AlpacaDataset(Dataset):
    def __init__(self, data: HFDataset, tokenizer: Any, max_length: int = 512) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def pad_and_truncate(self, sequence: List[int], pad_value: int) -> List[int]:
        if len(sequence) > self.max_length:
            return sequence[: self.max_length]
        return sequence + [pad_value] * (self.max_length - len(sequence))

    @staticmethod
    def build_prompt(instruction: str, input_text: str) -> str:
        user_message = f"{instruction}\n{input_text}" if input_text else instruction
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            "You are a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{user_message}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        prompt = self.build_prompt(instruction, input_text)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        output_ids = self.tokenizer.encode(output, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + output_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + output_ids

        input_ids = self.pad_and_truncate(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self.pad_and_truncate(attention_mask, 0)
        labels = self.pad_and_truncate(labels, -100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class AlpacaDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: Any,
        batch_size: int = 16,
        num_workers: int = 4,
        max_length: int = 512,
        max_samples: int = 2000,
        train_val_test_split: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.max_samples = max_samples

        self.train_val_test_split = train_val_test_split or [0.8, 0.1, 0.1]
        if len(self.train_val_test_split) != 3 or sum(self.train_val_test_split) != 1.0:
            raise ValueError("Split ratios must contain exactly 3 values summing to 1.")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        load_dataset(DATASET_ID)

    def _split_dataset(self, full_dataset: HFDataset) -> Tuple[HFDataset, HFDataset, HFDataset]:
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_val_test_split[0])
        val_size = int(total_size * self.train_val_test_split[1])
        test_size = total_size - train_size - val_size

        splits = full_dataset.train_test_split(test_size=(val_size + test_size), seed=SEED)
        train_data = splits["train"]
        temp_splits = splits["test"].train_test_split(test_size=test_size / (val_size + test_size), seed=SEED)
        return train_data, temp_splits["train"], temp_splits["test"]

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_dataset(DATASET_ID)
        if "train" not in dataset:
            raise ValueError("No 'train' split found.")

        full_dataset = cast(HFDataset, dataset["train"])  # type: ignore
        if self.max_samples and len(full_dataset) > self.max_samples:
            full_dataset = full_dataset.shuffle(seed=SEED).select(range(self.max_samples))

        train_data, val_data, test_data = self._split_dataset(full_dataset)
        self.train_dataset = AlpacaDataset(train_data, self.tokenizer, self.max_length)
        self.val_dataset = AlpacaDataset(val_data, self.tokenizer, self.max_length)
        self.test_dataset = AlpacaDataset(test_data, self.tokenizer, self.max_length)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
