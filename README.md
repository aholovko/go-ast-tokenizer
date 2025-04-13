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
├── checkpoints/                   <- Model's checkpoints
├── data/                          <- Dataset
├── reports/                       <- Generated reports
│   └── figures/                   <- Generated figures
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
│       ├── config.yaml
│       ├── data_loader.py
│       ├── dataset_builder.py
│       ├── tokenizer.py           <- wrapper for Go tokenizer
│       ├── train.py
│       └── utils.py
├── tests/                         <- Unit tests
│   └── test_tokenizer.py
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Usage

### Model Fine-tuning

Run the Llama 3 fine-tuning with:

```bash
make tune
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

Run the test suite with:

```bash
make test
```
or
```bash
python -m pytest tests/ -v
```

### Build Style Checker

Build and test the Go style checker with:

```bash
make checker
```

This command:
1. Runs the Go tests for the checker package
2. Builds the checker as a shared library for use by Python

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
