"""
Utility functions.
"""

import os
from typing import Any, Dict

import yaml


def load_config(config_filename: str = "config.yaml") -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), config_filename)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
