from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from transformers import AutoConfig, LlamaForSequenceClassification  # type: ignore

from src.go_ast_tokenizer.dataset import NUM_LABELS
from src.go_ast_tokenizer.utils import get_tokenizer

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"


class Llama3Classifier(L.LightningModule):
    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_id = BASE_MODEL_ID
        self.learning_rate = learning_rate

        tokenizer = get_tokenizer(self.model_id)

        config = AutoConfig.from_pretrained(
            self.model_id,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
        config.pad_token_id = tokenizer.pad_token_id

        model = LlamaForSequenceClassification.from_pretrained(
            self.model_id,
            config=config,
            device_map="auto",
        )

        # freeze everything except the head
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("score")

        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=NUM_LABELS)
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=NUM_LABELS, average="macro")

        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=NUM_LABELS)
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=NUM_LABELS, average="macro")

        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=NUM_LABELS)
        self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=NUM_LABELS, average="macro")

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, "test")

    def _run_step(self, batch: dict[str, torch.Tensor], stage: Literal["train", "val", "test"]) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"].to(logits.dtype)

        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)

        metrics = {
            "train": {"acc": self.train_acc, "f1_macro": self.train_f1},
            "val": {"acc": self.val_acc, "f1_macro": self.val_f1},
            "test": {"acc": self.test_acc, "f1_macro": self.test_f1},
        }[stage]

        for name, metric in metrics.items():
            metric.update(preds, batch["labels"])
            metric_name = f"{stage}/{name}" if not name.startswith("f1") else f"{stage}/{name}"
            self.log(metric_name, metric, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
