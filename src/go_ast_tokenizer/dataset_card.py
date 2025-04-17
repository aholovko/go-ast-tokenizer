"""
Dataset info card for `go-critic-style` dataset.
"""

DATASET_CARD = """
---
tags:
  - code
  - go
  - code-style-analysis
  - multi-label-classification
license: mit
language:
  - go
source_datasets:
  - bigcode/the-stack-v2
task_categories:
  - text-classification
task_ids:
  - multi-label-classification
dataset_info:
  features:
    - name: code
      dtype: string
      description: A snippet of Go source code.
    - name: labels
      dtype:
        sequence:
          class_label:
            names:
              - assignOp
              - builtinShadow
              - captLocal
              - commentFormatting
              - elseif
              - ifElseChain
              - paramTypeCombine
              - singleCaseSwitch
      description: >
        One or more style-rule violations detected by the go‑critic linter's "style" checker group.
  splits:
    - name: train
      num_examples: 1536
    - name: validation
      num_examples: 222
    - name: test
      num_examples: 448
  dataset_size: 2206
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
---
# go-critic-style

A **multi‑label** dataset of Go code snippets annotated with style violations from the [go‑critic linter's "style" group](https://go-critic.com/overview.html#checkers-from-the-style-group).
Curated from the [bigcode/the‑stack‑v2‑dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) "Go" split, filtered to examples of manageable length.

## Label Set

List of style violations covered by this dataset:

| ID | Label               | Description                                                         |
|--:|----------------------|---------------------------------------------------------------------|
| 0 | `assignOp`           | Could use `+=`, `-=`, `*=`, etc.                                    |
| 1 | `builtinShadow`      | Shadows a predeclared identifier.                                   |
| 2 | `captLocal`          | Local variable name begins with an uppercase letter.                |
| 3 | `commentFormatting`  | Comment is non‑idiomatic or badly formatted.                        |
| 4 | `elseif`             | Nested `if` statement that can be replaced with `else-if`.          |
| 5 | `ifElseChain`        | Repeated `if-else` statements can be replaced with `switch`.        |
| 6 | `paramTypeCombine`   | Function parameter types that can be combined (e.g. `x, y int`).    |
| 7 | `singleCaseSwitch`   | Statement `switch` that could be better written as `if`.            |

## Splits

The dataset is partitioned into training, validation, and test subsets in a 70/10/20 ratio:

| Split          | # Examples | Approx. % |
|---------------:|-----------:|----------:|
| **train**      | 1536       | 70%       |
| **validation** | 222        | 10%       |
| **test**       | 448        | 20%       |

"""  # noqa: E501
