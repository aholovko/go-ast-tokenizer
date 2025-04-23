from typing import Literal

import lightning as L
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
from transformers import AutoConfig, LlamaForSequenceClassification  # type: ignore

from src.go_ast_tokenizer.dataset import NUM_LABELS
from src.go_ast_tokenizer.utils import get_tokenizer

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"


class Llama3Classifier(L.LightningModule):
    def __init__(self, learning_rate: float = 1e-5) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate: float = learning_rate

        tokenizer = get_tokenizer(BASE_MODEL_ID)
        config = AutoConfig.from_pretrained(
            BASE_MODEL_ID,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
        config.pad_token_id = tokenizer.pad_token_id

        self.model = LlamaForSequenceClassification.from_pretrained(
            BASE_MODEL_ID,
            config=config,
            device_map="auto",
        )

        # freeze all parameters except the classification head
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("score")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metrics = {
            stage: {
                "acc": Accuracy(task="multilabel", num_labels=NUM_LABELS),
                "f1": F1Score(task="multilabel", num_labels=NUM_LABELS, average="macro"),
            }
            for stage in ("train", "val", "test")
        }

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"].to(logits.dtype)

        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits)

        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)

        self.metrics[stage]["acc"].update(preds, batch["labels"])
        self.metrics[stage]["f1"].update(preds, batch["labels"])
        return loss

    def _epoch_end(self, stage: str):
        acc = self.metrics[stage]["acc"].compute()
        f1 = self.metrics[stage]["f1"].compute()

        self.log(f"{stage}/acc", acc, prog_bar=True)
        self.log(f"{stage}/f1_macro", f1, prog_bar=True)

        self.metrics[stage]["acc"].reset()
        self.metrics[stage]["f1"].reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
