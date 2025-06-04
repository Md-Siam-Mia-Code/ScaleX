# scalex/archs/__init__.py
import importlib
import os
from typing import List
from types import ModuleType

from basicsr.utils import scandir

# For explicit import like `from scalex.archs import GFPGANv1`, etc.
from .gfpgan_bilinear_arch import GFPGANBilinear
from .gfpganv1_arch import GFPGANv1
from .gfpganv1_clean_arch import GFPGANv1Clean
from .restoreformer_arch import RestoreFormer
from .stylegan2_bilinear_arch import StyleGAN2GeneratorBilinear
from .stylegan2_clean_arch import StyleGAN2GeneratorClean
from .arcface_arch import ResNetArcFace

# This list controls `from scalex.archs import *`.
__all__ = [
    "GFPGANBilinear",
    "GFPGANv1",
    "GFPGANv1Clean",
    "RestoreFormer",
    "StyleGAN2GeneratorBilinear",
    "StyleGAN2GeneratorClean",
    "ResNetArcFace",
    # Add other key archs if necessary, but for now these are the main ones.
    # Helper components (EqualLinear, etc.) would be imported from their specific files.
]

# Dynamic import for registration (e.g., with basicsr.ARCH_REGISTRY)
arch_folder_path: str = os.path.dirname(os.path.abspath(__file__))
arch_filenames: List[str] = [
    os.path.splitext(os.path.basename(file_path))[0]
    for file_path in scandir(arch_folder_path)
    if file_path.endswith("_arch.py") and not file_path.startswith("__")
]

_arch_modules_loaded: List[ModuleType] = []  # Keep track of loaded modules
for file_name in arch_filenames:
    try:
        module = importlib.import_module(f".{file_name}", package=__package__)
        _arch_modules_loaded.append(module)
    except ImportError as e:
        print(
            f"Warning: Could not import architecture module 'scalex.archs.{file_name}'. Error: {e}"
        )
