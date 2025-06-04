# scalex/__init__.py
from .utils import ScaleXEnhancer

from .models.scalex_model import ScaleXModel
from .data.ffhq_degradation_dataset import FFHQDegradationDataset

from . import archs
from . import data
from . import models


# --- Version Information ---
try:
    from .version import __version__, __gitsha__, version_info
except ImportError:
    __version__ = "0.0.0.unknown"
    __gitsha__ = "unknown"
    version_info = (
        0,
        0,
        0,
        "unknown",
        "unknown",
    )  # Add a placeholder for releaselevel if needed

# --- Public API Definition (`__all__`) ---
__all__ = [
    "ScaleXEnhancer",
    "ScaleXModel",
    "FFHQDegradationDataset",
    "archs",  # Makes scalex.archs accessible
    "data",  # Makes scalex.data accessible
    "models",  # Makes scalex.models accessible
    'utils',  # If you have a utils module to export
    "__version__",
    "__gitsha__",
    "version_info",
]
