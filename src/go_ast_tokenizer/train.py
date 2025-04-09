"""
Fine-tuning Llama 3.1 model.
"""

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaForCausalLM  # type: ignore

from src.go_ast_tokenizer.data_loader import SEED, AlpacaDataModule
from src.go_ast_tokenizer.utils import load_config

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


class Llama3Model(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        learning_rate: float,
        lora_r: int,
        lora_alpha: int,
        lora_target_modules: list[str],
        lora_dropout: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            print(f"Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if len(tokenizer) > model.get_input_embeddings().weight.size(0):  # type: ignore
            print("Resizing the embedding matrix to match the tokenizer vocabulary size!")
            model.resize_token_embeddings(len(tokenizer))

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_config)

        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_loss_metric = torchmetrics.MeanMetric()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs.loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self(batch["input_ids"], batch["attention_mask"])
        self.train_loss_metric.update(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.train_loss_metric.compute()
        perplexity = torch.exp(avg_loss)
        self.log("avg_train_loss", avg_loss, prog_bar=True)
        self.log("train_perplexity", perplexity, prog_bar=True)
        self.train_loss_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self(batch["input_ids"], batch["attention_mask"])
        self.val_loss_metric.update(loss.detach())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        avg_loss = self.val_loss_metric.compute()
        perplexity = torch.exp(avg_loss)
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        self.log("val_perplexity", perplexity, prog_bar=True)
        self.val_loss_metric.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, foreach=True)


def main() -> None:
    config = load_config()
    training_config = config["training"]
    lora_config = config["lora"]

    L.seed_everything(SEED)

    model = Llama3Model(
        model_id=MODEL_ID,
        learning_rate=float(training_config["learning_rate"]),
        lora_r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_target_modules=lora_config["target_modules"],
        lora_dropout=float(lora_config["dropout"]),
    )

    data = AlpacaDataModule(tokenizer=model.tokenizer)
    logger = CSVLogger(save_dir="./reports")

    trainer = L.Trainer(
        fast_dev_run=True,
        logger=logger,
        max_epochs=int(training_config["max_epochs"]),
        accelerator=training_config["accelerator"],
        devices="auto",
        deterministic=True,
    )

    trainer.print("Starting training ...")
    trainer.fit(model, datamodule=data)
    trainer.print("Training successfully completed!")


if __name__ == "__main__":
    main()
