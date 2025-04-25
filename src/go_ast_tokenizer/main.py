import torch
from lightning.pytorch.cli import LightningCLI

from src.go_ast_tokenizer.dataset import GoCriticStyleDataModule
from src.go_ast_tokenizer.model import Llama3Classifier

# trade a bit of FP32 precision for faster, more memory-efficient matmuls
torch.set_float32_matmul_precision("medium")


def main() -> None:
    LightningCLI(Llama3Classifier, GoCriticStyleDataModule, save_config_callback=None)


if __name__ == "__main__":
    main()
