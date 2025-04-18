from dataclasses import dataclass
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoConfig, AutoTokenizer, LlamaForSequenceClassification, LlamaTokenizer  # type: ignore

from src.go_ast_tokenizer.data_loader import SEED, DataConfig, GoCriticStyleDataModule


@dataclass(frozen=True)
class HParams:
    model_id: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 4096
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-5
    max_epochs: int = 3
    num_labels: int = 8
    seed: int = SEED


class Llama3Classifier(L.LightningModule):
    def __init__(self, params: HParams) -> None:
        super().__init__()

        self.save_hyperparameters("params")
        self.params: HParams = params

        # --- tokenizer + model setup ---
        tokenizer = AutoTokenizer.from_pretrained(params.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer: LlamaTokenizer = tokenizer

        config = AutoConfig.from_pretrained(
            params.model_id,
            num_labels=params.num_labels,
            problem_type="multi_label_classification",
        )
        config.pad_token_id = tokenizer.pad_token_id

        model = LlamaForSequenceClassification.from_pretrained(
            params.model_id,
            config=config,
            device_map="auto",
            # torch_dtype=torch.float16,
        )

        # freeze everything except the head
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("score")

        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=params.num_labels)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=params.num_labels)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, "val")

    def _run_step(self, batch: dict[str, torch.Tensor], stage: Literal["train", "val"]):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"].to(logits.dtype)

        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        metric = self.train_acc if stage == "train" else self.val_acc
        metric.update(preds, batch["labels"])
        self.log(f"{stage}/acc", metric, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.params.lr)


def main():
    params = HParams()
    L.seed_everything(params.seed)

    model = Llama3Classifier(params)

    config = DataConfig(
        max_length=params.max_length,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
    )

    data_module = GoCriticStyleDataModule(
        tokenizer=model.tokenizer,
        config=config,
    )

    trainer = L.Trainer(
        max_epochs=params.max_epochs,
        accelerator="auto",
        devices="auto",
        # precision=16,
        deterministic=True,
        logger=CSVLogger("./reports"),
        callbacks=[ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True)],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)


if __name__ == "__main__":
    main()
