# scalex/data/__init__.py
import importlib
import os
from typing import List
from types import ModuleType

from basicsr.utils import scandir

# Explicitly import key dataset classes to make them available via `from scalex.data import ...`
from .ffhq_degradation_dataset import FFHQDegradationDataset

# This list controls `from scalex.data import *`
__all__ = [
    "FFHQDegradationDataset",
]

# Dynamic import for registration (e.g., with basicsr.DATASET_REGISTRY)
data_folder_path: str = os.path.dirname(os.path.abspath(__file__))
dataset_filenames: List[str] = [
    os.path.splitext(os.path.basename(file_path))[0]
    for file_path in scandir(data_folder_path)
    if file_path.endswith("_dataset.py") and not file_path.startswith("__")
]

_dataset_modules_loaded: List[ModuleType] = []
for file_name in dataset_filenames:
    try:
        module = importlib.import_module(f".{file_name}", package=__package__)
        _dataset_modules_loaded.append(module)
    except ImportError as e:
        print(
            f"Warning: Could not import dataset module 'scalex.data.{file_name}'. Error: {e}"
        )
