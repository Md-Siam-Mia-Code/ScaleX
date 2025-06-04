# scalex/models/__init__.py
import importlib
import os
from typing import List
from types import ModuleType

from basicsr.utils import scandir

# Explicitly import key model classes
# Assuming your model file is named 'scalex_model.py' and contains 'ScaleXModel'
# If it's still 'gfpgan_model.py', change the import accordingly.
from .scalex_model import ScaleXModel  # Or from .gfpgan_model import ScaleXModel

# This list controls `from scalex.models import *`
__all__ = [
    "ScaleXModel",
]

# Dynamic import for registration (e.g., with basicsr.MODEL_REGISTRY)
models_folder_path: str = os.path.dirname(os.path.abspath(__file__))
model_filenames: List[str] = [
    os.path.splitext(os.path.basename(file_path))[0]
    for file_path in scandir(models_folder_path)
    if file_path.endswith("_model.py") and not file_path.startswith("__")
]

_model_modules_loaded: List[ModuleType] = []
for file_name in model_filenames:
    try:
        module = importlib.import_module(f".{file_name}", package=__package__)
        _model_modules_loaded.append(module)
    except ImportError as e:
        print(
            f"Warning: Could not import model module 'scalex.models.{file_name}'. Error: {e}"
        )
