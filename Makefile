.PHONE: fit
fit:
	@echo "Running Llama 3 fine-tuning..."
	@uv run --env-file .env -m src.go_ast_tokenizer.main fit --config config.yaml

.PHONE: test
test:
	@echo "Evaluating Llama 3 on test data..."
	@uv run --env-file .env -m src.go_ast_tokenizer.main test --config config.yaml --ckpt_path best

.PHONY: unit-test
unit-test:
	@echo "Running tests..."
	@uv run pytest tests/ -v

.PHONY: checks
checks: uv-lock lint format typecheck

.PHONY: uv-lock
uv-lock:
	@echo "Locking dependencies..."
	@uv lock

.PHONY: lint
lint:
	@echo "Linting code..."
	@uv run ruff check --fix .

.PHONY: format
format:
	@echo "Formatting code..."
	@uv run ruff format .

.PHONY: typecheck
typecheck:
	@echo "Type checking code..."
	@uv run pyright

.PHONY: checker
checker:
	@echo "Building and testing style checker..."
	@cd src/go_ast_tokenizer/checker && go test -v . && go build -o _checkstyle.so -buildmode=c-shared .

.PHONY: dataset
dataset: checker
	@echo "Building go-critic-style dataset..."
	@uv run --env-file .env -m src.go_ast_tokenizer.dataset_builder

.PHONY: jupyter-kernel
jupyter-kernel:
	@echo "Installing Jupyter kernel..."
	@uv run -- ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=go-ast-tokenizer
	@echo "Start Jupyter Lab with 'make lab'"

.PHONY: lab
lab:
	@echo "Starting Jupyter Lab..."
	@uv run --env-file .env --with jupyter jupyter lab --no-browser --notebook-dir=./notebooks
