"""
Dataset builder.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import boto3
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import HfApi, create_repo, upload_file
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from smart_open import open as smart_open_file
from transformers import AutoTokenizer  # type: ignore

from src.go_ast_tokenizer.dataset_card import DATASET_CARD
from src.go_ast_tokenizer.go_style_checker import GoStyleChecker

MODEL_ID = "meta-llama/Llama-3.2-1B"

TOKENS_LENGTH_CUTOFF = 4000
LABELS_NUMBER_CUTOFF = 300

DATASET_SIZE = 2400
DATASET_OUTPUT_FILE = "dataset-go-critic-style.jsonl"
DATASET_NAME = "go-critic-style"

SEED = 2357

LABEL_NAMES = [
    "assignOp",
    "builtinShadow",
    "captLocal",
    "commentFormatting",
    "elseif",
    "ifElseChain",
    "paramTypeCombine",
    "singleCaseSwitch",
]
ALLOWED_LABELS = set(LABEL_NAMES)
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}


class GoCriticStyleDatasetBuilder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.s3_client = self._create_s3_client()
        self.style_checker = GoStyleChecker()

    @staticmethod
    def _create_s3_client():
        sts_client = boto3.Session(profile_name=os.environ["AWS_PROFILE_NAME"]).client("sts")
        credentials = sts_client.assume_role(
            RoleArn=os.environ["AWS_ROLE_ARN"],
            RoleSessionName=os.environ["AWS_SESSION_NAME"],
            DurationSeconds=900,
        ).get("Credentials")

        session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        return session.client("s3")

    def _download_content(self, blob_id: str, encoding: str) -> str:
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        with smart_open_file(
            s3_url,
            "rb",
            compression=".gz",
            transport_params={"client": self.s3_client},
        ) as fin:
            return fin.read().decode(encoding)

    def _download_with_retry(self, blob_id: str, encoding: str) -> str:
        try:
            return self._download_content(blob_id, encoding)
        except Exception:
            print("Refreshing credentials...")
            self.s3_client = self._create_s3_client()
            time.sleep(1)
            return self._download_content(blob_id, encoding)

    def _process_row(self, row: dict, stats: dict, label_counter: Counter) -> dict | None:
        stats["num_total_seen"] += 1

        blob_id = str(row.get("blob_id"))
        encoding = str(row.get("src_encoding"))

        snippet = self._download_with_retry(blob_id, encoding)

        if len(self.tokenizer.encode(snippet)) > TOKENS_LENGTH_CUTOFF:
            stats["num_too_long"] += 1
            return None

        try:
            warnings = self.style_checker.check_style(snippet)
        except Exception:
            warnings = []

        allowed_warnings = [w for w in warnings if w in ALLOWED_LABELS]
        if not allowed_warnings:
            stats["num_no_warnings"] += 1
            return None

        if any(label_counter[label] >= LABELS_NUMBER_CUTOFF for label in allowed_warnings):
            stats["num_labels_cutoff"] += 1
            return None

        for label in allowed_warnings:
            label_counter[label] += 1

        return {"code": snippet, "labels": allowed_warnings}

    @staticmethod
    def _print_stats(stats: dict, label_counter: Counter) -> None:
        print(
            f"Total: {stats['num_total_seen']}; Too Long: {stats['num_too_long']}; "
            f"No Warnings: {stats['num_no_warnings']}; Labels Cutoff: {stats['num_labels_cutoff']}; "
            f"Records Written: {stats['num_records_written']}"
        )
        print("Labels:")
        for label, count in label_counter.items():
            print(f"  {label}: {count}")

    def download_dataset(self) -> None:
        dataset = load_dataset("bigcode/the-stack-v2-dedup", "Go", split="train", streaming=True)

        output_dir = Path("./data")
        output_path = output_dir / DATASET_OUTPUT_FILE

        stats = {
            "num_total_seen": 0,
            "num_too_long": 0,
            "num_no_warnings": 0,
            "num_labels_cutoff": 0,
            "num_records_written": 0,
        }
        label_counter = Counter()

        with output_path.open("w", encoding="utf-8") as outfile:
            for row in dataset:
                record = self._process_row(dict(row), stats, label_counter)
                if not record:
                    continue

                outfile.write(json.dumps(record) + "\n")
                stats["num_records_written"] += 1

                if stats["num_records_written"] % 100 == 0:
                    print(f"Progress: {stats['num_records_written']}/{DATASET_SIZE}")

                if stats["num_records_written"] >= DATASET_SIZE:
                    break

        self._print_stats(stats, label_counter)
        print("Done!")

    def prepare_dataset(self, dataset_path: Path) -> DatasetDict:
        df = pd.read_json(dataset_path, lines=True)
        df = self._map_labels(df)
        train_df, val_df, test_df = self._split_data(df)
        return self._to_hf_datasets(train_df, val_df, test_df)

    @staticmethod
    def _map_labels(df: pd.DataFrame) -> pd.DataFrame:
        df["labels"] = df["labels"].apply(lambda labels: [LABEL_TO_ID[label] for label in labels])
        return df

    @staticmethod
    def _split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        mlb = MultiLabelBinarizer()
        binary_labels = mlb.fit_transform(df["labels"])

        # split into train and temporary sets (70:30)
        split1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)  # type: ignore
        train_idx, temp_idx = next(split1.split(df, binary_labels))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)
        temp_binary_labels = binary_labels[temp_idx]

        # split the temporary set into validation and test sets (validation:test = 1:2)
        split2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=2 / 3, random_state=SEED)  # type: ignore
        val_idx, test_idx = next(split2.split(temp_df, temp_binary_labels))
        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    @staticmethod
    def _to_hf_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetDict:
        features = Features({"code": Value("string"), "labels": Sequence(feature=ClassLabel(names=LABEL_NAMES))})

        train_ds = Dataset.from_pandas(train_df, features=features)
        val_ds = Dataset.from_pandas(val_df, features=features)
        test_ds = Dataset.from_pandas(test_df, features=features)

        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds,
            }
        )

    @staticmethod
    def dataset_exists_on_hf() -> bool:
        hf_api = HfApi()
        try:
            repo_id = f"{os.environ['HF_USERNAME']}/{DATASET_NAME}"
            hf_api.dataset_info(repo_id=repo_id)
            return True
        except Exception:
            return False

    @staticmethod
    def upload_to_hf(dataset_dict: DatasetDict) -> None:
        # create the dataset repository
        repo_id = f"{os.environ['HF_USERNAME']}/{DATASET_NAME}"
        create_repo(repo_id, token=os.environ["HF_TOKEN"], repo_type="dataset", exist_ok=True)

        # save dataset_dict to the hub
        dataset_dict.push_to_hub(repo_id)

        # save the dataset card to the README.md file
        with open("README.md", "w") as f:
            f.write(DATASET_CARD)

        # upload README.md to the repo
        upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )


def main() -> None:
    builder = GoCriticStyleDatasetBuilder()

    dataset_path = Path("./data") / DATASET_OUTPUT_FILE
    if not dataset_path.exists():
        print("Dataset not found. Downloading...")
        builder.download_dataset()

    if builder.dataset_exists_on_hf():
        print("Dataset already exists on Hugging Face.")
    else:
        print("Preparing dataset...")
        dataset_dic = builder.prepare_dataset(dataset_path)
        print("Uploading dataset to Hugging Face...")
        builder.upload_to_hf(dataset_dic)


if __name__ == "__main__":
    main()
