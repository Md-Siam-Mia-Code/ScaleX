# scalex/train.py

# flake8: noqa: E402 (Module level import not at top of file - basicsr might have specific needs)
# This script serves as the main entry point for training ScaleX models using basicsr's pipeline.

import os.path as osp

# Import basicsr's training pipeline function
from basicsr.train import train_pipeline

# Import ScaleX (formerly GFPGAN) specific modules to ensure their components
# (architectures, datasets, models) are registered with basicsr's registries.
# These imports trigger the __init__.py files in each respective submodule.
import scalex.archs # For architectures like ScaleXGenerator, discriminators, etc.
import scalex.data  # For datasets like FFHQDegradationDataset
import scalex.models # For the main ScaleXModel (training logic)

if __name__ == '__main__':
    # Determine the root path of the project.
    # This assumes train.py is located at `scalex/train.py`.
    # __file__ -> scalex/train.py
    # osp.pardir (1st) -> scalex/
    # osp.pardir (2nd) -> root directory containing the 'scalex' package and config files.
    # Example: if project root is /path/to/ScaleXProject/, and train.py is /path/to/ScaleXProject/scalex/train.py
    # then root_path will be /path/to/ScaleXProject/
    current_file_path = osp.abspath(__file__)
    scalex_package_dir = osp.dirname(current_file_path) # Gets 'scalex/' directory
    project_root_path = osp.dirname(scalex_package_dir) # Gets the parent of 'scalex/'

    # Call basicsr's training pipeline, passing the determined project root path.
    # The training pipeline will use this root_path to find configuration files (e.g., train.yml).
    train_pipeline(project_root_path)