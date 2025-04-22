"""
Utility functions.
"""

from transformers import AutoTokenizer, LlamaTokenizer  # type: ignore


def get_tokenizer(model_id: str) -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
