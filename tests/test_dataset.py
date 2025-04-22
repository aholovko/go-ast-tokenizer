import pytest
import torch
from datasets import ClassLabel, Features, Sequence, Value
from datasets import Dataset as HFDataset

from src.go_ast_tokenizer.dataset import GoCriticStyleDataset
from src.go_ast_tokenizer.utils import get_tokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
MAX_LEN = 8


@pytest.fixture
def sample_dataset():
    data = {
        "code": ["print(1)", "x = 2", "y = x + 1"],
        "labels": [[0, 2], [1], [0, 1, 2]],
    }
    features = Features(
        {
            "code": Value("string"),
            "labels": Sequence(feature=ClassLabel(num_classes=3), length=-1),
        }
    )
    hf = HFDataset.from_dict(data, features=features)
    tokenizer = get_tokenizer(MODEL_ID)
    return GoCriticStyleDataset(hf_dataset=hf, tokenizer=tokenizer, max_length=MAX_LEN)


@pytest.mark.skip(reason="skip for CI")
def test_length(sample_dataset):
    assert len(sample_dataset) == 3


@pytest.mark.skip(reason="skip for CI")
@pytest.mark.parametrize(
    "idx,expected",
    [
        (0, [1, 0, 1]),
        (1, [0, 1, 0]),
        (2, [1, 1, 1]),
    ],
)
def test_one_hot(sample_dataset, idx, expected):
    labels = sample_dataset[idx]["labels"]
    expected = torch.tensor(expected, dtype=torch.float32)
    assert labels.shape == (sample_dataset.num_classes,)
    assert torch.allclose(labels, expected)
