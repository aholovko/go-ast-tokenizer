import wandb
from lightning.pytorch.cli import LightningCLI

from src.go_ast_tokenizer.dataset import GoCriticStyleDataModule
from src.go_ast_tokenizer.model import Llama3Classifier


def main() -> None:
    wandb.login(force=True)
    LightningCLI(Llama3Classifier, GoCriticStyleDataModule, save_config_callback=None)


if __name__ == "__main__":
    main()
