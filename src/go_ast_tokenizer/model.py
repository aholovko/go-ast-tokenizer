from typing import Literal

import lightning as L
import torch
from torch import nn
from torchmetrics import F1Score, MeanMetric, MetricCollection, Precision, Recall
from transformers import AutoConfig, LlamaForSequenceClassification  # type: ignore

from src.go_ast_tokenizer.dataset import NUM_LABELS
from src.go_ast_tokenizer.tokenizer import SPECIAL_TOKENS
from src.go_ast_tokenizer.utils import get_tokenizer

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"


class Llama3Classifier(L.LightningModule):
    def __init__(self, learning_rate: float = 1e-5) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        tokenizer = get_tokenizer(BASE_MODEL_ID)
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})  # type: ignore

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
        self.model.resize_token_embeddings(len(tokenizer))

        # memory optimization: discards most forward activations and recomputes them on the backward pass
        self.model.gradient_checkpointing_enable()

        # freeze all parameters except the classification head
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("score")

        # unfreeze the embedding layer so new tokens can be learned
        embedding = self.model.get_input_embeddings()
        embedding.weight.requires_grad_(True)

        # loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_metrics = nn.ModuleDict({f"{stage}_loss": MeanMetric() for stage in ("train", "val", "test")})

        # macro-averaged precision, recall, and f1
        cls_metrics = MetricCollection(
            {
                "precision": Precision(task="multilabel", num_labels=NUM_LABELS, average="macro"),
                "recall": Recall(task="multilabel", num_labels=NUM_LABELS, average="macro"),
                "f1": F1Score(task="multilabel", num_labels=NUM_LABELS, average="macro"),
            }
        )
        self.classification_metrics = nn.ModuleDict(
            {f"{stage}_cls": cls_metrics.clone(prefix=f"{stage}/") for stage in ("train", "val", "test")}
        )

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # memory optimization: disable KV-cache so we don't store huge key/value tensors
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        return outputs.logits

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
        labels_float = batch["labels"].to(logits.dtype)
        labels_int = batch["labels"].int()

        loss = self.loss_fn(logits, labels_float)
        preds = torch.sigmoid(logits)

        self.loss_metrics[f"{stage}_loss"].update(loss)  # type: ignore
        self.classification_metrics[f"{stage}_cls"].update(preds, labels_int)  # type: ignore

        return loss

    def _epoch_end(self, stage: str):
        loss = self.loss_metrics[f"{stage}_loss"].compute()  # type: ignore
        self.log(f"{stage}/loss", loss, prog_bar=True)

        cls_metrics = self.classification_metrics[f"{stage}_cls"].compute()  # type: ignore
        self.log_dict(cls_metrics, prog_bar=True)

        self.loss_metrics[f"{stage}_loss"].reset()  # type: ignore
        self.classification_metrics[f"{stage}_cls"].reset()  # type: ignore

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
