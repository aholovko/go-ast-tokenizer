# Syntax-Aware Tokenizer for Go Code Style Analysis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checker: pyright](https://img.shields.io/badge/type%20checker-pyright-3775A9.svg)](https://github.com/microsoft/pyright)

## Requirements

- Python 3.12
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Go 1.24

## Project Structure

```
├── data/                          <- Raw data
├── notebooks/                     <- Jupyter notebooks
├── src/
│   └── go_ast_tokenizer/
│       ├── checker/
│       │   ├── checker.go
│       │   ├── checker_test.go
│       │   ├── export.go
│       │   ├── go.mod
│       │   └── go.sum
│       ├── tokenizer/
│       │   ├── go.mod
│       │   ├── tokenizer.go
│       │   └── tokenizer_test.go
│       ├── __init__.py
│       ├── dataset.py             <- Dataset and data module
│       ├── dataset_builder.py     <- Dataset builder
│       ├── dataset_card.py        <- Dataset info card for Hugging Face
│       ├── go_style_checker.py    <- Style checker wrapper
│       ├── main.py                <- Entry point
│       ├── model.py               <- Model definition
│       ├── tokenizer.py           <- Wrapper for Go tokenizer
│       └── utils.py
├── tests/                         <- Unit tests
├── config.yaml                    <- Configuration file for LightningCLI
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Usage

### Configuration

The project uses a `config.yaml` file to configure model training parameters:

```yaml
seed_everything: 2357         # Random seed for reproducibility
model:
  learning_rate: 1.0e-05      # Learning rate for optimizer
data:
  batch_size: 8               # Batch size for training
  num_workers: 4              # DataLoader workers
trainer:
  precision: "bf16-mixed"     # Training precision (bf16, 16, 32)
  max_epochs: 10              # Maximum training epochs
  # More options in the file...
```

You can modify these parameters in the configuration file to adjust training behavior.

### Model Fine-tuning

Run the Llama 3 fine-tuning with:

```bash
make fit
```

To evaluate the fine-tuned model on test data:

```bash
uv run --env-file .env -m src.go_ast_tokenizer.main test --config config.yaml --ckpt_path <path_to_checkpoint>
```

## Development

### Code Quality Checks

Run all code quality checks with:

```bash
make checks
```

This command runs the following checks in sequence:

1. **Dependencies**: `make uv-lock` - Locks dependencies
2. **Linting**: `make lint` - Lints the code using Ruff with auto-fixes 
3. **Formatting**: `make format` - Formats code using Ruff formatter
4. **Type checking**: `make typecheck` - Performs static type checking with Pyright

You can also run each check individually as needed.

### Run Tests

To run unit tests:

```bash
make unit-test
```
or
```bash
uv run pytest tests/ -v
```

### Build Style Checker

Build and test the Go style checker with:

```bash
make checker
```

This command:
1. Runs the Go tests for the checker package
2. Builds the checker as a shared library for use by Python

### Build Dataset

Generate the dataset with:

```bash
make dataset
```

This command:
1. Pulls the "Go" split of [bigcode/the‑stack‑v2‑dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)
2. Runs go‑critic (style group) → labels each snippet
3. Pushes dataset and README to 🤗 ${HF_USERNAME}/go-critic-style

**Note:** Required in `.env`: AWS_PROFILE_NAME, AWS_ROLE_ARN, AWS_SESSION_NAME, HF_USERNAME, HF_TOKEN

### Jupyter Notebook

#### Setup Jupyter Kernel

Install a dedicated Jupyter kernel for this project:

```bash
make jupyter-kernel
```

#### Run Jupyter Lab

Start Jupyter Lab:

```bash
make lab
```

This launches Jupyter Lab with the ./notebooks directory as the root.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
